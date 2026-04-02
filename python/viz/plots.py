"""
python/viz/plots.py

Generates all static visualizations from the annotated feature CSV
and ML results JSON. Designed to run headlessly in GitHub Actions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

BG    = "#080812"
GOLD  = "#f5c842"
CYAN  = "#00e5ff"
PINK  = "#ff4da6"
GREEN = "#00ff9f"
WHITE = "#eef2ff"
CMAP  = LinearSegmentedColormap.from_list("traj", [CYAN, PINK, GOLD])

def _ax(ax, title="", xl="", yl="", grid=True):
    ax.set_facecolor("#0d0d22")
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    for s in ax.spines.values(): s.set_edgecolor("#2a2a4a")
    if title: ax.set_title(title, color=GOLD, fontsize=10, fontweight="bold", pad=5)
    if xl:    ax.set_xlabel(xl, fontsize=9)
    if yl:    ax.set_ylabel(yl, fontsize=9)
    if grid:  ax.grid(True, color="#1a1a3a", lw=0.5, alpha=0.7)


def plot_anomaly_dashboard(df: pd.DataFrame, out: str) -> None:
    """4-panel anomaly detection dashboard."""
    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.08)

    days = df["met_hours"] / 24
    anom = df.get("anomaly_if", pd.Series([False]*len(df)))
    score= df.get("anomaly_score_if", pd.Series([0.0]*len(df)))

    # 1. Speed residual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(days, df["speed_residual"], color=CYAN, lw=0.8, label="speed residual")
    ax1.fill_between(days, df["speed_residual"], alpha=0.12, color=CYAN)
    # Mark anomalies
    ax1.scatter(days[anom], df["speed_residual"][anom],
                color=PINK, s=30, zorder=6, label="anomaly", marker="^")
    _ax(ax1, "Speed Residual (measured − vis-viva)", "Days", "km/s")
    ax1.axhline(0, color=WHITE, lw=0.5, ls="--", alpha=0.4)
    ax1.legend(fontsize=7, facecolor="#0d0d22", edgecolor="#2a2a4a", labelcolor=WHITE)

    # 2. Energy residual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(days, df["energy_residual"], color=GOLD, lw=0.8)
    ax2.fill_between(days, df["energy_residual"], alpha=0.12, color=GOLD)
    ax2.scatter(days[anom], df["energy_residual"][anom],
                color=PINK, s=30, zorder=6, marker="^")
    _ax(ax2, "Specific Energy Residual (ΔE from t₀)", "Days", "km²/s²")
    ax2.axhline(0, color=WHITE, lw=0.5, ls="--", alpha=0.4)

    # 3. Jerk magnitude  (3rd deriv of position — spikes at burns)
    ax3 = fig.add_subplot(gs[1, 0])
    jerk_clipped = df["jerk_mag"].clip(upper=df["jerk_mag"].quantile(0.999))
    ax3.semilogy(days, jerk_clipped + 1e-12, color=GREEN, lw=0.8)
    ax3.scatter(days[anom], jerk_clipped[anom] + 1e-12,
                color=PINK, s=30, zorder=6, marker="^")
    _ax(ax3, "Jerk Magnitude  (TCM signature)", "Days", "km/s³  (log)")

    # 4. Anomaly score
    ax4 = fig.add_subplot(gs[1, 1])
    norm = Normalize(0, 1)
    for i in range(len(days)-1):
        c = PINK if anom.iloc[i] else CYAN
        ax4.fill_between(days.iloc[i:i+2],
                         score.iloc[i:i+2],
                         alpha=0.7 if anom.iloc[i] else 0.3,
                         color=c, linewidth=0)
    ax4.plot(days, score, color=GOLD, lw=0.8, alpha=0.9)
    ax4.axhline(0.5, color=PINK, lw=0.8, ls="--", alpha=0.6, label="anomaly threshold")
    _ax(ax4, "Isolation Forest Anomaly Score", "Days", "Score  (1 = most anomalous)")
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=7, facecolor="#0d0d22", edgecolor="#2a2a4a", labelcolor=WHITE)

    n_anom = int(anom.sum())
    fig.suptitle(
        f"ANOMALY DETECTION  |  Artemis II Orion  |  {n_anom} events flagged",
        color=GOLD, fontsize=14, fontweight="bold", y=0.97
    )
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[viz] Saved {out}")


def plot_orbital_elements(df: pd.DataFrame, out: str) -> None:
    """Keplerian element evolution over the mission."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    days = df["met_hours"] / 24

    params = [
        ("dist_km",   "Distance from Earth",  "km",    CYAN),
        ("speed_kms", "Speed",                "km/s",  PINK),
        ("ecc",       "Eccentricity",         "",      GOLD),
        ("inc_deg",   "Inclination",          "deg",   GREEN),
        ("h_mag",     "Angular Momentum |h|", "km²/s", "#9b7fff"),
        ("energy",    "Specific Energy",      "km²/s²","#ff9f00"),
    ]
    anom = df.get("anomaly_if", pd.Series([False]*len(df)))

    for ax, (col, title, unit, col_color) in zip(axes.flat, params):
        data = df[col]
        ax.plot(days, data, color=col_color, lw=1.0)
        ax.fill_between(days, data, alpha=0.1, color=col_color)
        ax.scatter(days[anom], data[anom], color=PINK, s=20, zorder=5, marker="^", alpha=0.7)
        lab = f"{title} [{unit}]" if unit else title
        _ax(ax, title, "Days", lab)

    fig.suptitle("ORBITAL ELEMENTS EVOLUTION  |  Artemis II Orion  |  EME2000",
                 color=GOLD, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[viz] Saved {out}")


def plot_landing_prediction(result: dict, out: str) -> None:
    """Landing zone prediction map."""
    land  = result["models"]["landing"]
    if land.get("lat") is None:
        print("[viz] No landing prediction available, skipping")
        return

    lat0, lon0 = land["lat"], land["lon"]
    sigma_km   = land.get("sigma_km", 50)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)

    # Simple equirectangular background
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_aspect("equal")

    # Draw simplified coastlines (ASCII approximation)
    # Draw Pacific splashdown zone reference box (Artemis nominal)
    box_lats = [-10, 10, 10, -10, -10]
    box_lons = [-160, -160, -120, -120, -160]
    ax.fill(box_lons, box_lats, alpha=0.07, color="#1a5fb4", label="Pacific target zone")
    ax.plot(box_lons, box_lats, color="#1a5fb4", lw=0.8, ls="--", alpha=0.5)

    # 1σ uncertainty ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        xy=(lon0, lat0),
        width=2 * land["lon_1sigma_deg"],
        height=2 * land["lat_1sigma_deg"],
        angle=0, edgecolor=PINK, facecolor=PINK, alpha=0.25, lw=2, label="1σ ellipse"
    )
    ax.add_patch(ellipse)

    # 3σ ellipse
    ellipse3 = Ellipse(
        xy=(lon0, lat0),
        width=6 * land["lon_1sigma_deg"],
        height=6 * land["lat_1sigma_deg"],
        angle=0, edgecolor=PINK, facecolor="none", alpha=0.4, lw=1, ls="--", label="3σ ellipse"
    )
    ax.add_patch(ellipse3)

    # Landing point
    ax.plot(lon0, lat0, "o", color=GOLD, ms=12, zorder=8, label=f"Predicted splashdown")
    ax.annotate(
        f"  {lat0:.2f}°N, {lon0:.2f}°E\n  ±{sigma_km:.0f} km (1σ)",
        xy=(lon0, lat0), fontsize=9, color=GOLD,
        xytext=(lon0+5, lat0+3)
    )

    # Grid
    for lon_g in range(-180, 181, 30):
        ax.axvline(lon_g, color="#1a1a3a", lw=0.5, alpha=0.6)
    for lat_g in range(-90, 91, 30):
        ax.axhline(lat_g, color="#1a1a3a", lw=0.5, alpha=0.6)
    ax.axhline(0, color="#2a2a5a", lw=0.8)
    ax.axvline(0, color="#2a2a5a", lw=0.8)

    ax.tick_params(colors=WHITE, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#2a2a4a")
    ax.set_xlabel("Longitude", color=WHITE, fontsize=9)
    ax.set_ylabel("Latitude", color=WHITE, fontsize=9)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    ax.legend(fontsize=9, facecolor="#0d0d22", edgecolor="#2a2a4a", labelcolor=WHITE, loc="lower left")
    ax.set_title(
        f"ARTEMIS II  PREDICTED SPLASHDOWN  |  Monte Carlo ({result.get('mc_samples', 3000)} samples)",
        color=GOLD, fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[viz] Saved {out}")


def plot_lstm_performance(df: pd.DataFrame, roll_err: pd.DataFrame, out: str) -> None:
    """LSTM rolling forecast error + predicted vs actual."""
    if roll_err is None or len(roll_err) == 0:
        print("[viz] No LSTM errors to plot, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    days_roll = roll_err["met_hours"] / 24
    axes[0].plot(days_roll, roll_err["rmse"], color=CYAN, lw=1.0)
    axes[0].fill_between(days_roll, roll_err["rmse"], alpha=0.12, color=CYAN)
    # Highlight high-error regions
    thresh = roll_err["rmse"].quantile(0.95)
    high_err = roll_err["rmse"] > thresh
    axes[0].scatter(days_roll[high_err], roll_err["rmse"][high_err],
                    color=PINK, s=20, zorder=5, label=f"Top 5% RMSE (>{thresh:.4f})")
    _ax(axes[0], "LSTM 1-Step-Ahead Forecast RMSE\n(spike = potential maneuver)",
        "Days", "Scaled RMSE")
    axes[0].legend(fontsize=7, facecolor="#0d0d22", edgecolor="#2a2a4a", labelcolor=WHITE)

    # RMSE distribution
    axes[1].hist(roll_err["rmse"], bins=50, color=GOLD, alpha=0.7, edgecolor="#0d0d22")
    axes[1].axvline(thresh, color=PINK, lw=1, ls="--", label=f"95th pct={thresh:.4f}")
    _ax(axes[1], "RMSE Distribution", "RMSE", "Count")
    axes[1].legend(fontsize=7, facecolor="#0d0d22", edgecolor="#2a2a4a", labelcolor=WHITE)

    fig.suptitle("LSTM FORECASTER PERFORMANCE  |  Artemis II",
                 color=GOLD, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[viz] Saved {out}")


def generate_all(
    features_csv:  str,
    ml_results_json: str,
    out_dir: str = "outputs",
) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    df = pd.read_csv(features_csv)
    with open(ml_results_json) as f:
        results = json.load(f)

    plot_anomaly_dashboard(df, f"{out_dir}/anomaly_dashboard.png")
    plot_orbital_elements(df, f"{out_dir}/orbital_elements.png")
    plot_landing_prediction(results, f"{out_dir}/landing_prediction.png")
    print("[viz] All plots generated.")


if __name__ == "__main__":
    import sys
    feat = sys.argv[1] if len(sys.argv) > 1 else "outputs/features_annotated.csv"
    res  = sys.argv[2] if len(sys.argv) > 2 else "outputs/ml_results.json"
    generate_all(feat, res)
