// fetcher/src/main.rs
//
// Fetches the latest Artemis II OEM from:
//   1. NASA AROW (primary) — polls the known public URL pattern
//   2. JPL HORIZONS API (fallback) — queries Orion by HORIZONS ID
//
// Usage:
//   oem-fetch --out data/latest.oem [--source arow|horizons|auto]
//             [--horizons-id -170] [--start 2026-04-01] [--stop 2026-04-11]

use anyhow::{anyhow, Result};
use chrono::{Duration, Utc};
use reqwest::Client;
use std::env;

// ── NASA AROW known URL patterns ──────────────────────────────────────────────
// These mirror what AROW serves publicly during active Artemis missions.
// The fetcher tries each in order.
const AROW_OEM_URLS: &[&str] = &[
    // Primary: current mission OEM (updated every ~1h by FDO)
    "https://www.nasa.gov/wp-content/uploads/artemis/artemis-ii/oem/EPH_OEM.txt",
    // Secondary: direct JSC FDOweb public mirror (when available)
    "https://www.nasa.gov/sites/default/files/atoms/files/artemis_ii_oem_latest.zip",
];

// JPL HORIZONS REST API  (Orion/Artemis II = -170, Artemis I was -166)
const HORIZONS_API: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut out_path = "data/latest.oem".to_string();
    let mut source = "auto".to_string();
    let mut horizons_id = "-170".to_string(); // Artemis II Orion
    let mut start_time = String::new();
    let mut stop_time = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--source" => {
                i += 1;
                source = args[i].clone();
            }
            "--horizons-id" => {
                i += 1;
                horizons_id = args[i].clone();
            }
            "--start" => {
                i += 1;
                start_time = args[i].clone();
            }
            "--stop" => {
                i += 1;
                stop_time = args[i].clone();
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // Default time window: now-1h to now+10d
    if start_time.is_empty() {
        let t = Utc::now() - Duration::hours(1);
        start_time = t.format("%Y-%m-%d").to_string();
    }
    if stop_time.is_empty() {
        let t = Utc::now() + Duration::days(10);
        stop_time = t.format("%Y-%m-%d").to_string();
    }

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let client = Client::builder()
        .user_agent("orion-watch/0.1 (github.com/your-org/orion-watch)")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let content = match source.as_str() {
        "arow" => fetch_arow(&client).await?,
        "horizons" => fetch_horizons(&client, &horizons_id, &start_time, &stop_time).await?,
        _ => {
            eprintln!("[fetcher] Trying NASA AROW first...");
            match fetch_arow(&client).await {
                Ok(c) => {
                    eprintln!("[fetcher] AROW: success ({} bytes)", c.len());
                    c
                }
                Err(e) => {
                    eprintln!("[fetcher] AROW failed: {}. Falling back to HORIZONS.", e);
                    fetch_horizons(&client, &horizons_id, &start_time, &stop_time).await?
                }
            }
        }
    };

    std::fs::write(&out_path, &content)?;
    eprintln!("[fetcher] Saved {} bytes → {}", content.len(), out_path);
    println!("{}", out_path);
    Ok(())
}

// ── AROW fetcher ──────────────────────────────────────────────────────────────
async fn fetch_arow(client: &Client) -> Result<String> {
    for url in AROW_OEM_URLS {
        eprintln!("[fetcher] GET {}", url);
        let resp = client.get(*url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let body = r.text().await?;
                // Validate it looks like OEM data
                if body.contains("CCSDS_OEM_VERS") || body.contains("OBJECT_NAME") {
                    return Ok(body);
                }
                // Handle zip response
                if url.ends_with(".zip") {
                    let bytes = body.into_bytes();
                    return extract_oem_from_zip(bytes);
                }
                // Might be HTML-wrapped OEM (like FDOweb serves)
                if body.contains("<pre>") && body.contains("OBJECT_NAME") {
                    return Ok(body);
                }
                eprintln!(
                    "[fetcher] Response from {} didn't look like OEM, trying next",
                    url
                );
            }
            Ok(r) => eprintln!("[fetcher] HTTP {} from {}", r.status(), url),
            Err(e) => eprintln!("[fetcher] Error from {}: {}", url, e),
        }
    }
    Err(anyhow!("All AROW URLs failed"))
}

fn extract_oem_from_zip(bytes: Vec<u8>) -> Result<String> {
    use std::io::Read;
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name().ends_with(".txt") || file.name().ends_with(".asc") {
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            if content.contains("OBJECT_NAME") {
                return Ok(content);
            }
        }
    }
    Err(anyhow!("No OEM file found in zip"))
}

// ── HORIZONS fetcher ──────────────────────────────────────────────────────────
// Queries the JPL HORIZONS REST API for vector ephemeris (OEM-compatible output).
async fn fetch_horizons(
    client: &Client,
    body_id: &str,
    start_time: &str,
    stop_time: &str,
) -> Result<String> {
    eprintln!(
        "[fetcher] Querying JPL HORIZONS for body {} ({} → {})",
        body_id, start_time, stop_time
    );

    let params = [
        ("format", "text"),
        ("COMMAND", body_id),
        ("OBJ_DATA", "NO"),
        ("MAKE_EPHEM", "YES"),
        ("EPHEM_TYPE", "VECTORS"),
        ("CENTER", "500@399"), // geocentric (Earth center)
        ("REF_FRAME", "ICRF"), // J2000 ≈ EME2000
        ("START_TIME", start_time),
        ("STOP_TIME", stop_time),
        ("STEP_SIZE", "4m"), // match FDO cadence
        ("VEC_TABLE", "2"),  // X,Y,Z, VX,VY,VZ
        ("VEC_CORR", "NONE"),
        ("OUT_UNITS", "KM-S"),
        ("CSV_FORMAT", "NO"),
    ];

    let resp = client.get(HORIZONS_API).query(&params).send().await?;

    if !resp.status().is_success() {
        return Err(anyhow!("HORIZONS HTTP {}", resp.status()));
    }

    let text = resp.text().await?;
    if text.contains("ERROR") || text.contains("No ephemeris") {
        return Err(anyhow!("HORIZONS error: {}", &text[..text.len().min(300)]));
    }

    // Convert HORIZONS VECTORS output to CCSDS OEM format
    convert_horizons_to_oem(&text, body_id)
}

fn convert_horizons_to_oem(horizons: &str, body_id: &str) -> Result<String> {
    // HORIZONS VECTORS table:  JDTDB, Cal, X, Y, Z, VX, VY, VZ, ...
    let mut header = format!(
        "CCSDS_OEM_VERS = 2.0\nCOMMENT Converted from JPL HORIZONS\nCREATION_DATE = {}\nORIGINATOR = JPL/HORIZONS\n\nMETA_START\nOBJECT_NAME = ORION/{}\nCENTER_NAME = EARTH\nREF_FRAME = EME2000\nTIME_SYSTEM = UTC\n",
        Utc::now().format("%Y-%m-%dT%H:%M:%S"),
        body_id,
    );

    let mut lines = Vec::new();
    let mut in_data = false;
    for line in horizons.lines() {
        let line = line.trim();
        if line.starts_with("$$SOE") {
            in_data = true;
            continue;
        }
        if line.starts_with("$$EOE") {
            break;
        }
        if !in_data {
            continue;
        }

        // Format: "JDTDB  Cal_Date  X  Y  Z  VX  VY  VZ  ..."
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            continue;
        }

        // Calendar date is like "A.D. 2026-Apr-02 03:26:56.0000"
        // Find the date/time part
        let epoch_str = parse_horizons_epoch(&parts)?;
        let x = parts[parts.len() - 6].parse::<f64>().unwrap_or(0.0);
        let y = parts[parts.len() - 5].parse::<f64>().unwrap_or(0.0);
        let z = parts[parts.len() - 4].parse::<f64>().unwrap_or(0.0);
        let vx = parts[parts.len() - 3].parse::<f64>().unwrap_or(0.0);
        let vy = parts[parts.len() - 2].parse::<f64>().unwrap_or(0.0);
        let vz = parts[parts.len() - 1].parse::<f64>().unwrap_or(0.0);

        lines.push(format!(
            "{} {:.15E} {:.15E} {:.15E} {:.15E} {:.15E} {:.15E}",
            epoch_str, x, y, z, vx, vy, vz
        ));
    }

    if lines.is_empty() {
        return Err(anyhow!("No data rows in HORIZONS response"));
    }

    header.push_str(&format!("META_STOP\n\n{}\n", lines.join("\n")));
    Ok(header)
}

fn parse_horizons_epoch(parts: &[&str]) -> Result<String> {
    // Parts contain: JD, "A.D.", "YYYY-Mon-DD", "HH:MM:SS.ffff", ...
    for (i, p) in parts.iter().enumerate() {
        if *p == "A.D." && i + 2 < parts.len() {
            let date_str = parts[i + 1];
            let time_str = parts[i + 2].trim_end_matches('0').trim_end_matches('.');
            // "2026-Apr-02" → "2026-04-02"
            let date_iso = chrono_month(date_str)?;
            return Ok(format!("{}T{}", date_iso, time_str));
        }
    }
    Err(anyhow!("Could not parse HORIZONS epoch from {:?}", parts))
}

fn chrono_month(s: &str) -> Result<String> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(anyhow!("bad date {}", s));
    }
    let mon = match parts[1] {
        "Jan" => "01",
        "Feb" => "02",
        "Mar" => "03",
        "Apr" => "04",
        "May" => "05",
        "Jun" => "06",
        "Jul" => "07",
        "Aug" => "08",
        "Sep" => "09",
        "Oct" => "10",
        "Nov" => "11",
        "Dec" => "12",
        other => return Err(anyhow!("unknown month {}", other)),
    };
    Ok(format!("{}-{}-{}", parts[0], mon, parts[2]))
}

fn print_help() {
    eprintln!(
        r#"
oem-fetch — NASA AROW + JPL HORIZONS OEM fetcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE:
    oem-fetch [OPTIONS]

OPTIONS:
    --out PATH         Output file path       (default: data/latest.oem)
    --source MODE      arow | horizons | auto  (default: auto)
    --horizons-id ID   HORIZONS body ID        (default: -170  = Artemis II Orion)
    --start DATE       YYYY-MM-DD              (default: today-1h)
    --stop DATE        YYYY-MM-DD              (default: today+10d)

NOTES:
    auto mode tries AROW first, falls back to HORIZONS on failure.
    HORIZONS body IDs: Artemis I Orion = -166, Artemis II = -170 (verify at ssd.jpl.nasa.gov)
"#
    );
}
