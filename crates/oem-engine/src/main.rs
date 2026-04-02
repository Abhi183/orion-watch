// oem-engine/src/main.rs
//
// CLI: oem-engine [--input FILE] [--features-out PATH] [--summary-out PATH]
//
// Usage examples:
//   oem-engine --input data/latest.oem --features-out data/features.csv
//   oem-engine --input data/latest.oem --features-out data/features.csv --summary-out outputs/summary.json
//   echo "..." | oem-engine   (reads stdin if no --input given)

use anyhow::Result;
use oem_engine::{
    compute_features, mission_summary, parse_oem, write_features_csv, write_summary_json,
};
use std::env;
use std::io::Read;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut input_path: Option<String> = None;
    let mut features_path: Option<String> = None;
    let mut summary_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input_path = args.get(i).cloned();
            }
            "--features-out" => {
                i += 1;
                features_path = args.get(i).cloned();
            }
            "--summary-out" => {
                i += 1;
                summary_path = args.get(i).cloned();
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => {
                eprintln!("Unknown arg: {}", other);
            }
        }
        i += 1;
    }

    // Read OEM content
    let t_parse = Instant::now();
    let content = match &input_path {
        Some(path) => {
            eprintln!("[oem-engine] Reading {}", path);
            std::fs::read_to_string(path)?
        }
        None => {
            eprintln!("[oem-engine] Reading from stdin");
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let oem = parse_oem(&content)?;
    eprintln!(
        "[oem-engine] Parsed {} state vectors in {:.1}ms",
        oem.vectors.len(),
        t_parse.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!("[oem-engine] Object:    {}", oem.meta.object_name);
    eprintln!("[oem-engine] Frame:     {}", oem.meta.ref_frame);
    eprintln!(
        "[oem-engine] Coverage:  {} → {}",
        oem.meta.start_time, oem.meta.stop_time
    );

    // Compute features
    let t_feat = Instant::now();
    let features = compute_features(&oem);
    eprintln!(
        "[oem-engine] Features computed in {:.1}ms",
        t_feat.elapsed().as_secs_f64() * 1000.0
    );

    // Summary
    let summary = mission_summary(&oem, &features);
    eprintln!("[oem-engine] ─────────────────────────────────────");
    eprintln!(
        "[oem-engine] Duration:      {:.3} days",
        summary.duration_days
    );
    eprintln!("[oem-engine] Max altitude:  {:.0} km", summary.max_alt_km);
    eprintln!("[oem-engine] Apoapsis at:   {}", summary.apoapsis_epoch);
    eprintln!(
        "[oem-engine] Min speed:     {:.4} km/s",
        summary.min_speed_kms
    );
    eprintln!(
        "[oem-engine] Entry speed:   {:.4} km/s",
        summary.entry_speed_kms
    );
    eprintln!(
        "[oem-engine] Anomaly flags: {}/{}",
        summary.n_anomalies, summary.n_vectors
    );
    eprintln!("[oem-engine] ─────────────────────────────────────");

    // Write outputs
    let feat_path = features_path.as_deref().unwrap_or("data/features.csv");
    write_features_csv(&features, feat_path)?;
    eprintln!("[oem-engine] Features → {}", feat_path);

    if let Some(ref sp) = summary_path {
        write_summary_json(&summary, sp)?;
        eprintln!("[oem-engine] Summary → {}", sp);
    } else {
        // Always write summary alongside features
        let sp = feat_path.replace(".csv", "_summary.json");
        write_summary_json(&summary, &sp)?;
        eprintln!("[oem-engine] Summary → {}", sp);
    }

    // Print JSON summary to stdout for piping
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn print_help() {
    eprintln!(
        r#"
oem-engine — CCSDS OEM parser + orbital mechanics feature engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE:
    oem-engine [OPTIONS]

OPTIONS:
    --input FILE         OEM file to parse  (default: stdin)
    --features-out PATH  CSV output path    (default: data/features.csv)
    --summary-out PATH   JSON summary path  (auto-derived from features-out if omitted)
    --help, -h           Print this help

OUTPUT (CSV columns):
    epoch, met_hours, pos_x_km, pos_y_km, pos_z_km,
    vel_x_kms, vel_y_kms, vel_z_kms,
    dist_km, alt_km, speed_kms,
    energy, h_mag, h_x, h_y, h_z,
    ecc, sma_km, inc_deg,
    visviva_kms, speed_residual, energy_residual,
    fpa_deg, accel_mag, jerk_mag,
    delta_speed, delta_energy, anomaly_score

EXAMPLE:
    oem-engine --input data/latest.oem --features-out data/features.csv
    oem-engine --input data/latest.oem | jq .duration_days
"#
    );
}
