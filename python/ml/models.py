"""
python/ml/models.py

Three ML models for Artemis II trajectory analysis:
  1. AnomalyDetector   — Isolation Forest on orbital residuals (finds TCM burns)
  2. LSTMForecaster    — Sequence model: window of states → predict next N steps
  3. LandingPredictor  — Gradient boosted regressor: EI state → landing lat/lon
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[ml] sklearn not available; anomaly detection disabled")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[ml] PyTorch not available; LSTM forecaster disabled")

# ── Constants ─────────────────────────────────────────────────────────────────
MU_EARTH = 398_600.4418  # km³/s²
R_EARTH  = 6_371.0       # km

# =============================================================================
# 1. ANOMALY DETECTOR  (Isolation Forest on orbital residuals)
# =============================================================================

class AnomalyDetector:
    """
    Detects trajectory correction maneuvers (TCMs) and sensor anomalies
    by fitting an Isolation Forest on orbital feature residuals.

    Features used:
        - speed_residual   (measured speed − vis-viva, should ≈0 on free coast)
        - energy_residual  (energy drift from t=0)
        - jerk_mag         (third derivative of position; spikes at burns)
        - delta_energy     (inter-sample energy change)
        - accel_mag        (non-gravitational acceleration proxy)
    """

    FEATURES = [
        "speed_residual", "energy_residual",
        "jerk_mag", "delta_energy", "accel_mag",
    ]

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self._model: Optional[Pipeline] = None
        self._trained  = False

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        if not HAS_SKLEARN:
            print("[AnomalyDetector] sklearn unavailable")
            return self

        X = df[self.FEATURES].fillna(0).values

        # Rolling normalisation: subtract rolling median to handle slow drifts
        X_norm = self._rolling_detrend(X)

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("isoforest", IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self._model.fit(X_norm)
        self._trained = True
        print(f"[AnomalyDetector] Fitted on {len(df)} samples, "
              f"contamination={self.contamination}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with added columns: anomaly_if (bool), anomaly_score_if (float)."""
        result = df.copy()
        if not self._trained or not HAS_SKLEARN:
            result["anomaly_if"]       = False
            result["anomaly_score_if"] = 0.0
            return result

        X = df[self.FEATURES].fillna(0).values
        X_norm = self._rolling_detrend(X)

        # Isolation Forest: -1 = anomaly, +1 = normal
        labels = self._model.predict(X_norm)
        scores = self._model.decision_function(X_norm)  # higher = more normal
        # Normalise to [0,1] where 1 = most anomalous
        scores_norm = 1.0 - (scores - scores.min()) / (np.ptp(scores) + 1e-9)

        result["anomaly_if"]       = labels == -1
        result["anomaly_score_if"] = scores_norm
        return result

    @staticmethod
    def _rolling_detrend(X: np.ndarray, window: int = 30) -> np.ndarray:
        """Subtract rolling median to remove slow secular trends."""
        result = np.zeros_like(X)
        for j in range(X.shape[1]):
            ser = pd.Series(X[:, j])
            med = ser.rolling(window, center=True, min_periods=1).median()
            result[:, j] = X[:, j] - med.values
        return result

    def summary(self, df_pred: pd.DataFrame) -> dict:
        anomalies = df_pred[df_pred["anomaly_if"]]
        return {
            "n_total":    len(df_pred),
            "n_anomalies": len(anomalies),
            "anomaly_rate": f"{100*len(anomalies)/len(df_pred):.2f}%",
            "top_events": anomalies.nlargest(5, "anomaly_score_if")[
                ["epoch", "met_hours", "anomaly_score_if", "delta_energy", "speed_residual"]
            ].to_dict(orient="records"),
        }


# =============================================================================
# 2. LSTM FORECASTER
# =============================================================================

class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, horizon: int):
        super().__init__()
        self.lstm  = nn.LSTM(n_features, hidden, n_layers,
                             batch_first=True, dropout=0.2 if n_layers > 1 else 0)
        self.head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_features * horizon),
        )
        self.n_features = n_features
        self.horizon    = horizon

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.horizon, self.n_features)


class LSTMForecaster:
    """
    Sequence-to-sequence LSTM for next-state prediction.

    Inputs:  window of (dist_km, speed_kms, energy, h_mag, ecc, fpa_deg)
    Outputs: next `horizon` steps of the same 6 features

    Useful for:
      - Predicting where Orion will be in 30–60 min
      - Residual analysis to detect off-nominal maneuvers
    """

    STATE_FEATURES = [
        "dist_km", "speed_kms", "energy", "h_mag", "ecc", "fpa_deg",
        "pos_x_km", "pos_y_km", "pos_z_km",
    ]

    def __init__(
        self,
        window:   int = 30,   # input context window (steps)
        horizon:  int = 10,   # prediction horizon (steps)
        hidden:   int = 128,
        n_layers: int = 2,
        epochs:   int = 60,
        lr:       float = 3e-4,
    ):
        self.window   = window
        self.horizon  = horizon
        self.hidden   = hidden
        self.n_layers = n_layers
        self.epochs   = epochs
        self.lr       = lr
        self._model: Optional[_LSTMNet] = None
        self._scaler  = None
        self._trained = False

    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) sliding-window arrays."""
        from sklearn.preprocessing import StandardScaler
        data = df[self.STATE_FEATURES].fillna(method="ffill").values.astype(np.float32)

        if self._scaler is None:
            self._scaler = StandardScaler()
            data = self._scaler.fit_transform(data)
        else:
            data = self._scaler.transform(data)

        X, y = [], []
        for i in range(len(data) - self.window - self.horizon):
            X.append(data[i : i + self.window])
            y.append(data[i + self.window : i + self.window + self.horizon])
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame) -> "LSTMForecaster":
        if not HAS_TORCH:
            print("[LSTMForecaster] PyTorch unavailable")
            return self

        X, y = self._prepare(df)
        n_feat = len(self.STATE_FEATURES)

        self._model = _LSTMNet(n_feat, self.hidden, self.n_layers, self.horizon)
        opt     = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(X)
        yt = torch.tensor(y)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        self._model.train()
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item() * len(xb)
            sched.step()
            epoch_loss /= len(Xt)
            losses.append(epoch_loss)
            if epoch % 10 == 0:
                print(f"[LSTMForecaster] epoch {epoch:3d}/{self.epochs}  loss={epoch_loss:.6f}")

        self._trained = True
        self.train_losses = losses
        print(f"[LSTMForecaster] Training complete. Final loss={losses[-1]:.6f}")
        return self

    def predict(self, df: pd.DataFrame, start_idx: int = -1) -> pd.DataFrame:
        """Predict next `horizon` steps from the last `window` steps in df."""
        if not self._trained or self._model is None:
            return pd.DataFrame()

        from sklearn.preprocessing import StandardScaler
        data  = df[self.STATE_FEATURES].fillna(method="ffill").values.astype(np.float32)
        data_s = self._scaler.transform(data)

        if start_idx < 0:
            start_idx = len(data_s) - self.window

        x = torch.tensor(data_s[start_idx : start_idx + self.window]).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            pred_s = self._model(x).squeeze(0).numpy()

        pred = self._scaler.inverse_transform(pred_s)
        return pd.DataFrame(pred, columns=self.STATE_FEATURES)

    def rolling_forecast_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 1-step-ahead forecast error across the full trajectory."""
        if not self._trained or self._model is None:
            return pd.DataFrame()

        data  = df[self.STATE_FEATURES].fillna(method="ffill").values.astype(np.float32)
        data_s = self._scaler.transform(data)
        n     = len(data_s)

        errors = []
        self._model.eval()
        with torch.no_grad():
            for i in range(self.window, n - self.horizon):
                x = torch.tensor(data_s[i-self.window:i]).unsqueeze(0)
                pred = self._model(x).squeeze(0).numpy()
                actual = data_s[i : i + self.horizon]
                rmse = float(np.sqrt(np.mean((pred - actual)**2)))
                errors.append({"idx": i, "met_hours": df["met_hours"].iloc[i], "rmse": rmse})

        return pd.DataFrame(errors)


# =============================================================================
# 3. LANDING ZONE PREDICTOR
# =============================================================================

class LandingPredictor:
    """
    Estimates splashdown location from the Entry Interface state vector.

    Method:
      1. Generate a Monte Carlo ensemble of EI states by perturbing the
         real EI state (±1σ in position/velocity from nav uncertainty).
      2. Propagate each perturbed state through a simplified 3-DOF
         ballistic entry model to compute landing lat/lon.
      3. Train a GradientBoosting regressor on {EI features → (lat, lon)}.
      4. At inference, run the model on the actual EI state to produce
         a predicted landing point + uncertainty ellipse.

    EI definition: altitude = 121.9 km, speed ≈ 11 km/s
    """

    EI_ALT_KM = 121.9

    def __init__(self, n_samples: int = 5000):
        self.n_samples = n_samples
        self._model_lat: Optional[Pipeline] = None
        self._model_lon: Optional[Pipeline] = None
        self._trained   = False

    def fit(self, ei_state: dict) -> "LandingPredictor":
        """
        ei_state: {x, y, z, vx, vy, vz} in km, km/s (EME2000).
        Generates Monte Carlo ensemble and trains regressors.
        """
        if not HAS_SKLEARN:
            return self

        print(f"[LandingPredictor] Generating {self.n_samples} Monte Carlo samples...")
        X_mc, lat_mc, lon_mc = self._monte_carlo(ei_state)

        for name, model_attr, targets in [
            ("lat", "_model_lat", lat_mc),
            ("lon", "_model_lon", lon_mc),
        ]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )),
            ])
            model.fit(X_mc, targets)
            setattr(self, model_attr, model)

            # Quick CV score
            cv = cross_val_score(model, X_mc, targets, cv=5, scoring="r2")
            print(f"[LandingPredictor] {name} R²={cv.mean():.4f} ± {cv.std():.4f}")

        self._trained = True
        return self

    def predict(self, ei_state: dict) -> dict:
        """Returns predicted landing {lat, lon} + 1σ uncertainty radius."""
        if not self._trained:
            return {"lat": None, "lon": None, "sigma_km": None}

        X = np.array(self._ei_features(ei_state)).reshape(1, -1)
        lat = float(self._model_lat.predict(X)[0])
        lon = float(self._model_lon.predict(X)[0])

        # Bootstrap uncertainty
        n_boot = 200
        lat_b, lon_b = [], []
        rng = np.random.default_rng(0)
        pos_sig = 0.5; vel_sig = 0.001   # nav 1σ
        for _ in range(n_boot):
            pert = {
                "x":  ei_state["x"]  + rng.normal(0, pos_sig),
                "y":  ei_state["y"]  + rng.normal(0, pos_sig),
                "z":  ei_state["z"]  + rng.normal(0, pos_sig),
                "vx": ei_state["vx"] + rng.normal(0, vel_sig),
                "vy": ei_state["vy"] + rng.normal(0, vel_sig),
                "vz": ei_state["vz"] + rng.normal(0, vel_sig),
            }
            Xp = np.array(self._ei_features(pert)).reshape(1, -1)
            lat_b.append(self._model_lat.predict(Xp)[0])
            lon_b.append(self._model_lon.predict(Xp)[0])

        sig_lat = np.std(lat_b) * 111.0  # deg→km
        sig_lon = np.std(lon_b) * 111.0 * np.cos(np.radians(lat))
        sigma_km = (sig_lat**2 + sig_lon**2)**0.5

        return {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "sigma_km": round(sigma_km, 2),
            "lat_1sigma_deg": round(np.std(lat_b), 4),
            "lon_1sigma_deg": round(np.std(lon_b), 4),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ei_features(s: dict) -> list:
        """13 features derived from EI state vector."""
        r  = np.array([s["x"], s["y"], s["z"]])
        v  = np.array([s["vx"], s["vy"], s["vz"]])
        r_mag = float(np.linalg.norm(r))
        v_mag = float(np.linalg.norm(v))
        h = np.cross(r, v)
        h_mag = float(np.linalg.norm(h))
        energy = 0.5 * v_mag**2 - MU_EARTH / r_mag
        ecc = (1 + 2 * energy * h_mag**2 / MU_EARTH**2)
        ecc = float(np.sqrt(max(ecc, 0)))
        # Entry angle (FPA below horizon)
        fpa = float(np.arcsin(np.dot(v, r / r_mag) / v_mag) * 180 / np.pi)
        # Azimuth
        az = float(np.arctan2(s["vy"], s["vx"]) * 180 / np.pi)
        lat = float(np.arcsin(s["z"] / r_mag) * 180 / np.pi)
        lon = float(np.arctan2(s["y"], s["x"]) * 180 / np.pi)
        return [r_mag, v_mag, energy, h_mag, ecc, fpa, az,
                lat, lon, s["vx"], s["vy"], s["vz"], h[2]]

    def _monte_carlo(self, ei_state: dict):
        """Propagate MC ensemble through simplified ballistic re-entry."""
        rng = np.random.default_rng(42)
        # Nav uncertainties (1σ)
        POS_SIG = 1.0   # km
        VEL_SIG = 0.005  # km/s

        X_features = []
        lats, lons  = [], []

        for _ in range(self.n_samples):
            pert = {
                "x":  ei_state["x"]  + rng.normal(0, POS_SIG),
                "y":  ei_state["y"]  + rng.normal(0, POS_SIG),
                "z":  ei_state["z"]  + rng.normal(0, POS_SIG),
                "vx": ei_state["vx"] + rng.normal(0, VEL_SIG),
                "vy": ei_state["vy"] + rng.normal(0, VEL_SIG),
                "vz": ei_state["vz"] + rng.normal(0, VEL_SIG),
            }
            lat, lon = self._propagate_entry(pert)
            X_features.append(self._ei_features(pert))
            lats.append(lat)
            lons.append(lon)

        return np.array(X_features), np.array(lats), np.array(lons)

    @staticmethod
    def _propagate_entry(s: dict, dt: float = 10.0) -> Tuple[float, float]:
        """
        Simplified 3-DOF Keplerian+drag ballistic re-entry propagator.
        Integrates from EI (121 km) to ~10 km with Euler steps.
        Good enough for landing ellipse estimation — not precision GNC.
        """
        # Earth rotation rate [rad/s]
        OMEGA_E = 7.2921150e-5
        CD_A_M  = 1.1e-3   # ballistic coefficient proxy (km²/kg equivalent)
        rho0    = 1.225e-12  # sea-level density in km/km³ ≈ 1.225 kg/m³ (SI)
        H_scale = 8.5        # km scale height

        r = np.array([s["x"], s["y"], s["z"]], dtype=float)
        v = np.array([s["vx"], s["vy"], s["vz"]], dtype=float)

        alt = np.linalg.norm(r) - R_EARTH
        step = 0
        max_steps = 10000

        while alt > 0.01 and step < max_steps:
            r_mag = np.linalg.norm(r)
            alt   = r_mag - R_EARTH

            # Gravity
            a_grav = -MU_EARTH / r_mag**3 * r

            # Aerodynamic drag (exponential atmosphere)
            rho = rho0 * np.exp(-alt / H_scale)
            v_rel = v - np.cross(np.array([0, 0, OMEGA_E]), r)
            v_rel_mag = np.linalg.norm(v_rel)
            a_drag = -0.5 * CD_A_M * rho * v_rel_mag * v_rel

            # Euler step
            a_total = a_grav + a_drag
            v = v + a_total * dt
            r = r + v * dt
            step += 1

            if alt < 5.0:
                break

        # Convert final position to lat/lon
        r_mag = np.linalg.norm(r)
        lat   = float(np.arcsin(r[2] / r_mag) * 180 / np.pi)
        lon   = float(np.arctan2(r[1], r[0]) * 180 / np.pi)
        return lat, lon


# =============================================================================
# Public convenience function
# =============================================================================

def run_full_pipeline(
    features_csv: str,
    summary_json: str,
    out_dir: str = "outputs",
) -> dict:
    """
    Load Rust-computed features, run all 3 ML models, return results dict.
    """
    Path(out_dir).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("  ORION-WATCH  ML PIPELINE")
    print(f"{'='*60}")

    df = pd.read_csv(features_csv)
    print(f"[pipeline] Loaded {len(df)} feature rows from {features_csv}")

    with open(summary_json) as f:
        summary = json.load(f)
    print(f"[pipeline] Mission: {summary['object_name']}  "
          f"Duration: {summary['duration_days']:.3f} days")

    results = {"summary": summary, "models": {}}

    # ── 1. Anomaly Detection ──────────────────────────────────────────────────
    print("\n[1/3] Anomaly Detection (Isolation Forest)")
    detector = AnomalyDetector(contamination=0.03)
    detector.fit(df)
    df = detector.predict(df)
    anom_summary = detector.summary(df)
    results["models"]["anomaly"] = anom_summary
    print(f"      Found {anom_summary['n_anomalies']} anomalies "
          f"({anom_summary['anomaly_rate']})")
    for ev in anom_summary["top_events"]:
        print(f"      → T+{ev['met_hours']:.2f}h  score={ev['anomaly_score_if']:.3f}"
              f"  Δenergy={ev['delta_energy']:.6f}")

    # ── 2. LSTM Forecaster ────────────────────────────────────────────────────
    print("\n[2/3] LSTM Forecaster")
    if HAS_TORCH:
        forecaster = LSTMForecaster(window=30, horizon=10, epochs=40)
        forecaster.fit(df)

        # Predict next 10 steps from end of trajectory
        predicted_df = forecaster.predict(df)
        results["models"]["lstm"] = {
            "window_steps":  30,
            "horizon_steps": 10,
            "train_loss_final": float(forecaster.train_losses[-1]),
            "predicted_states": predicted_df.to_dict(orient="records"),
        }
        print(f"      Final training loss: {forecaster.train_losses[-1]:.6f}")
        print(f"      Predicted next 10 states (dist_km): "
              f"{[f'{x:.0f}' for x in predicted_df['dist_km'].tolist()]}")

        # Rolling forecast error for anomaly cross-check
        roll_err = forecaster.rolling_forecast_error(df)
        results["models"]["lstm"]["rolling_rmse_mean"] = float(roll_err["rmse"].mean()) \
            if not roll_err.empty else None
    else:
        print("      Skipped (PyTorch not available)")
        results["models"]["lstm"] = {"status": "unavailable"}

    # ── 3. Landing Zone Predictor ─────────────────────────────────────────────
    print("\n[3/3] Landing Zone Predictor")
    ei_row = df.iloc[-1]
    ei_state = {
        "x": ei_row["pos_x_km"], "y": ei_row["pos_y_km"], "z": ei_row["pos_z_km"],
        "vx": ei_row["vel_x_kms"], "vy": ei_row["vel_y_kms"], "vz": ei_row["vel_z_kms"],
    }
    predictor = LandingPredictor(n_samples=3000)
    predictor.fit(ei_state)
    landing  = predictor.predict(ei_state)
    results["models"]["landing"] = landing
    print(f"      Predicted splashdown:  {landing['lat']:.3f}°N, "
          f"{landing['lon']:.3f}°E  (±{landing['sigma_km']:.1f} km)")

    # ── Save annotated features ───────────────────────────────────────────────
    out_features = str(Path(out_dir) / "features_annotated.csv")
    df.to_csv(out_features, index=False)
    print(f"\n[pipeline] Saved annotated features → {out_features}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_json = str(Path(out_dir) / "ml_results.json")
    with open(out_json, "w") as fh:
        # Convert non-serialisable types
        def default(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, bool):
                return bool(o)
            raise TypeError(f"Not serialisable: {type(o)}")
        json.dump(results, fh, indent=2, default=default)
    print(f"[pipeline] Saved ML results → {out_json}")

    return results


if __name__ == "__main__":
    import sys
    feat = sys.argv[1] if len(sys.argv) > 1 else "data/features.csv"
    summ = sys.argv[2] if len(sys.argv) > 2 else feat.replace(".csv", "_summary.json")
    out  = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    run_full_pipeline(feat, summ, out)
