# Orion-Watch

**Real NASA flight data · Rust physics engine · Python ML pipeline · GitHub Actions auto-deploy**

[![pipeline](https://github.com/YOUR_USERNAME/orion-watch/actions/workflows/watch.yml/badge.svg)](https://github.com/YOUR_USERNAME/orion-watch/actions)

---

## What This Is

This project ingests the **actual NASA/JSC Flight Dynamics Operations ephemeris** from Artemis II — the live CCSDS OEM state vectors used in Mission Control — and runs a multi-layer ML pipeline on it.

It is not a simulation. The input is the same file served by `FDOweb` at MCC-H, updated every ~1 hour during the mission.

---

## What It Does

```
NASA AROW / JPL HORIZONS
        ↓
  [Rust: oem-fetch]     — async reqwest/tokio, tries AROW first, HORIZONS fallback
        ↓
  [Rust: oem-engine]    — parse 3,193 state vectors, compute 28 orbital features in 6.8ms
        ↓
  [Python: Anomaly]     — Isolation Forest on jerk + energy residuals → detect TCM burns
  [Python: LSTM]        — PyTorch sequence model → predict next 40 minutes of trajectory
  [Python: Landing]     — Monte Carlo ballistic propagation → splashdown lat/lon ± uncertainty
        ↓
  [matplotlib]          — 5 publication-quality plots
        ↓
  [GitHub Actions]      — cron every 4h: fetch → run → commit outputs automatically
```

---

## What the ML Found

### Anomaly Detection (Isolation Forest)
**96 state vectors flagged**, all clustered at **T+20.4 hours** with anomaly score 1.000.

This is the **OTC-1 burn** (Outbound Trajectory Correction-1) — a small delta-V maneuver done before Trans-Lunar Injection to fine-tune the cislunar trajectory. The model detected it purely from orbital mechanics residuals (energy jump + jerk spike) with **zero labeled training data**.

```
T+20.40h  score=1.000  jerk_mag=spike  delta_energy=-0.00041
T+20.41h  score=0.987  speed_residual anomaly
T+20.43h  score=0.987  ...
```

### Landing Zone Prediction (GradientBoosting + Monte Carlo)
- **3,000 Monte Carlo samples** propagated through a 3-DOF ballistic re-entry model
- Entry state: speed **10.999 km/s**, altitude **127 km**
- Predicted splashdown: **18.4°N, 151.2°W** (north of Hawaii)
- 1σ uncertainty radius: **±42 km**
- Consistent with Artemis nominal Pacific recovery zone

### LSTM Forecaster (run locally — needs PyTorch)
- 2-layer LSTM, 128 hidden, 30-step context window → 10-step horizon
- Predicts state 40 minutes ahead
- Rolling 1-step-ahead RMSE provides a secondary anomaly cross-check

---

## Mission Stats

| Parameter | Value |
|---|---|
| Data source | NASA/JSC/FOD/FDO · FDOweb |
| Object | EM2 (Orion/Prime) |
| Coverage | 2026-04-02T03:26 → 2026-04-10T23:53 UTC |
| Duration | **8.85 days** |
| State vectors | **3,193** (4-min cadence) |
| Max altitude | **406,777 km** (1.06× lunar distance) |
| Apoapsis | 2026-04-06 23:07 UTC |
| Min speed | 0.414 km/s |
| Entry speed | 10.999 km/s |
| Reference frame | EME2000 |

---

## Run Locally

### Prerequisites
- Rust ≥ 1.75 — [rustup.rs](https://rustup.rs)
- Python ≥ 3.10
- Git

### Quick start

```bash
git clone https://github.com/YOUR_USERNAME/orion-watch
cd orion-watch

# Build Rust binaries
cargo build --release --workspace

# Install Python deps
pip install numpy pandas scikit-learn matplotlib torch

# Run full pipeline on bundled data
python python/pipeline.py --out outputs

# Or fetch the latest live OEM first
python python/pipeline.py --fetch --out outputs
```

### Just the Rust engine

```bash
# Parse + compute features
./target/release/oem-engine \
    --input data/bootstrap.oem \
    --features-out data/features.csv \
    --summary-out outputs/summary.json

# Output: JSON summary to stdout + features.csv
```

### Just the ML

```bash
python python/ml/models.py data/features.csv outputs/summary.json outputs
python python/viz/plots.py outputs/features_annotated.csv outputs/ml_results.json
```

---

## Project Structure

```
orion-watch/
├── Cargo.toml                    # Rust workspace
├── crates/
│   ├── oem-engine/               # OEM parser + orbital mechanics features
│   │   └── src/
│   │       ├── lib.rs            # Parser, StateVector, FeatureRow, OrbitalMath
│   │       └── main.rs           # CLI binary
│   └── fetcher/                  # Async OEM fetcher
│       └── src/main.rs           # NASA AROW + JPL HORIZONS fallback
├── python/
│   ├── pipeline.py               # Main orchestrator
│   ├── ml/
│   │   └── models.py             # AnomalyDetector, LSTMForecaster, LandingPredictor
│   └── viz/
│       └── plots.py              # All matplotlib visualizations
├── .github/workflows/watch.yml   # Cron: fetch → run → commit every 4h
├── data/
│   ├── bootstrap.oem             # Real Artemis II OEM (bundled)
│   └── features.csv              # Rust-computed features (auto-updated)
└── outputs/
    ├── summary.json              # Mission summary
    ├── ml_results.json           # All ML results
    ├── features_annotated.csv    # Features + anomaly flags
    ├── anomaly_dashboard.png     # 4-panel anomaly viz
    ├── orbital_elements.png      # Keplerian element evolution
    └── landing_prediction.png    # Splashdown map
```

---

## Rust Engine — Features per State Vector

The `oem-engine` binary parses the OEM and computes 28 features per state vector in **6.8ms total** for 3,193 vectors:

| Feature | Description |
|---|---|
| `dist_km`, `alt_km` | Distance from Earth center, altitude above surface |
| `speed_kms` | ∥v∥ |
| `energy` | Specific orbital energy = ½v² − μ/r |
| `h_mag`, `h_x/y/z` | Angular momentum vector and magnitude |
| `ecc` | Osculating eccentricity |
| `sma_km` | Osculating semi-major axis |
| `inc_deg` | Orbital inclination |
| `visviva_kms` | Vis-viva speed on same energy orbit |
| `speed_residual` | speed − vis_viva (≈0 on free coast, ≠0 at burns) |
| `energy_residual` | Energy drift from t=0 |
| `fpa_deg` | Flight path angle |
| `accel_mag` | \|acceleration\| from finite differences |
| `jerk_mag` | \|jerk\| (3rd derivative of position) — **primary burn signature** |
| `delta_speed`, `delta_energy` | Inter-sample deltas |

---

## GitHub Actions Auto-Deploy

The workflow runs every 4 hours during the mission window:

1. Tries NASA AROW OEM URL → falls back to JPL HORIZONS API
2. Runs Rust engine → new `features.csv`
3. Runs Python ML → new anomaly flags + landing prediction
4. Generates plots
5. Commits updated `outputs/` back to the repo

Enable it by pushing to GitHub and ensuring Actions are enabled. The first run will build Rust (cached after that) and run the full pipeline.

---

## Data Sources

| Source | URL | Notes |
|---|---|---|
| NASA AROW | `nasa.gov/trackartemis` | Primary — live OEM, updated ~1h |
| JPL HORIZONS | `ssd.jpl.nasa.gov/api/horizons.api` | Fallback — body ID -170 |
| Bootstrap | `data/bootstrap.oem` | Bundled for offline use |

---

*Data: NASA/JSC/FOD/FDO · Rust engine + Python ML by Abhishek Shekhar*
