#!/usr/bin/env python3
"""
python/pipeline.py

Main orchestration script:
  1. Fetch latest OEM   (calls Rust oem-fetch binary OR downloads directly)
  2. Parse + features   (calls Rust oem-engine binary)
  3. ML pipeline        (Python: AnomalyDetector + LSTM + LandingPredictor)
  4. Visualisations     (Python: matplotlib)
  5. Emit GitHub Actions summary

Usage:
  python pipeline.py [--oem PATH] [--fetch] [--no-ml] [--out DIR]

CI usage (in GitHub Actions):
  python pipeline.py --fetch --out outputs
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"[pipeline] $ {' '.join(cmd)}")
    kwargs = {"check": check}
    if capture:
        kwargs.update({"capture_output": True, "text": True})
    return subprocess.run(cmd, **kwargs)


def ensure_rust_binaries() -> tuple[Path, Path]:
    """Build Rust workspace if binaries aren't present."""
    engine  = ROOT / "target" / "release" / "oem-engine"
    fetcher = ROOT / "target" / "release" / "oem-fetch"

    if not engine.exists() or not fetcher.exists():
        print("[pipeline] Building Rust workspace (release)...")
        run(["cargo", "build", "--release", "--workspace"],
            check=True)
    else:
        print(f"[pipeline] Using cached Rust binaries")

    return engine, fetcher


def fetch_oem(fetcher: Path, out_path: str, source: str = "auto") -> str:
    """Fetch latest OEM file. Falls back to bundled data if network fails."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    result = run(
        [str(fetcher), "--out", out_path, "--source", source],
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"[pipeline] Fetcher failed: {result.stderr}")
        fallback = str(ROOT / "data" / "bootstrap.oem")
        if Path(fallback).exists():
            print(f"[pipeline] Using bootstrap OEM: {fallback}")
            return fallback
        raise RuntimeError("No OEM available — fetcher failed and no bootstrap found")
    return out_path


def run_engine(engine: Path, oem_path: str, features_path: str) -> dict:
    """Run Rust engine on OEM file, return parsed summary."""
    summary_path = features_path.replace(".csv", "_summary.json")
    result = run(
        [str(engine), "--input", oem_path,
         "--features-out", features_path,
         "--summary-out", summary_path],
        check=True, capture=True,
    )
    # stdout is the JSON summary
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}


def run_ml(features_path: str, summary_path: str, out_dir: str) -> dict:
    sys.path.insert(0, str(ROOT / "python"))
    from ml.models import run_full_pipeline
    return run_full_pipeline(features_path, summary_path, out_dir)


def run_viz(features_annotated: str, ml_results: str, out_dir: str) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from viz.plots import generate_all
    generate_all(features_annotated, ml_results, out_dir)


def write_github_summary(results: dict, out_dir: str) -> None:
    """Write $GITHUB_STEP_SUMMARY markdown when running in Actions."""
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary:
        return

    s   = results.get("summary", {})
    ml  = results.get("models", {})
    anom= ml.get("anomaly", {})
    land= ml.get("landing", {})

    md = f"""## ORION-WATCH Run — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

### Mission
| Field | Value |
|---|---|
| Object | {s.get('object_name','—')} |
| Duration | {s.get('duration_days',0):.3f} days |
| Max altitude | {s.get('max_alt_km',0):.0f} km |
| Apoapsis at | {s.get('apoapsis_epoch','—')} |
| Entry speed | {s.get('entry_speed_kms',0):.4f} km/s |

### Anomaly Detection
- **{anom.get('n_anomalies','?')}** events flagged ({anom.get('anomaly_rate','?')})
- Top events:
"""
    for ev in anom.get("top_events", []):
        md += f"  - T+{ev.get('met_hours',0):.2f}h  score={ev.get('anomaly_score_if',0):.3f}\n"

    md += f"""
### Landing Prediction
| | |
|---|---|
| Lat | {land.get('lat','—')} |
| Lon | {land.get('lon','—')} |
| 1σ radius | {land.get('sigma_km','—')} km |

### Outputs
- `outputs/anomaly_dashboard.png`
- `outputs/orbital_elements.png`
- `outputs/landing_prediction.png`
- `outputs/features_annotated.csv`
- `outputs/ml_results.json`
"""
    with open(step_summary, "w") as f:
        f.write(md)
    print("[pipeline] GitHub Actions summary written")


def main():
    parser = argparse.ArgumentParser(description="Orion-Watch ML pipeline")
    parser.add_argument("--oem",     default="data/latest.oem",   help="OEM input file")
    parser.add_argument("--fetch",   action="store_true",          help="Fetch latest OEM")
    parser.add_argument("--source",  default="auto",               help="arow|horizons|auto")
    parser.add_argument("--no-ml",   action="store_true",          help="Skip ML (engine only)")
    parser.add_argument("--no-viz",  action="store_true",          help="Skip visualisations")
    parser.add_argument("--out",     default="outputs",            help="Output directory")
    parser.add_argument("--data",    default="data",               help="Data directory")
    args = parser.parse_args()

    out_dir       = str(ROOT / args.out)
    data_dir      = str(ROOT / args.data)
    oem_path      = str(ROOT / args.oem)
    features_path = str(Path(data_dir) / "features.csv")
    summary_path  = features_path.replace(".csv", "_summary.json")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ensure Rust binaries ──────────────────────────────────────────
    engine, fetcher = ensure_rust_binaries()

    # ── Step 2: Fetch OEM ─────────────────────────────────────────────────────
    if args.fetch:
        print(f"\n[pipeline] Step 1/4: Fetching OEM (source={args.source})")
        oem_path = fetch_oem(fetcher, oem_path, args.source)
    else:
        print(f"\n[pipeline] Step 1/4: Using existing OEM: {oem_path}")

    # ── Step 3: Rust engine ───────────────────────────────────────────────────
    print(f"\n[pipeline] Step 2/4: Rust feature engine")
    rust_summary = run_engine(engine, oem_path, features_path)
    if rust_summary:
        print(f"[pipeline]   Vectors:  {rust_summary.get('n_vectors')}")
        print(f"[pipeline]   Duration: {rust_summary.get('duration_days'):.3f} days")
        print(f"[pipeline]   Max alt:  {rust_summary.get('max_alt_km'):.0f} km")

    if args.no_ml:
        print("[pipeline] --no-ml set, exiting after engine step")
        return

    # ── Step 4: ML pipeline ───────────────────────────────────────────────────
    print(f"\n[pipeline] Step 3/4: ML pipeline")
    results = run_ml(features_path, summary_path, out_dir)

    # ── Step 5: Visualisations ────────────────────────────────────────────────
    if not args.no_viz:
        print(f"\n[pipeline] Step 4/4: Visualisations")
        feat_annotated = str(Path(out_dir) / "features_annotated.csv")
        ml_results_json= str(Path(out_dir) / "ml_results.json")
        run_viz(feat_annotated, ml_results_json, out_dir)

    # ── GitHub Actions summary ────────────────────────────────────────────────
    write_github_summary(results, out_dir)

    print(f"\n[pipeline] Done. Outputs in {out_dir}/")
    print("[pipeline] Key files:")
    for f in Path(out_dir).glob("*"):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
