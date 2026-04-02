// oem-engine/src/lib.rs
//
// CCSDS OEM v2.0 parser and orbital mechanics feature engine.
// Parses NASA FDO/AROW ephemeris files, computes a rich feature set
// per state vector, and writes a CSV for downstream ML consumption.

use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────
pub const MU_EARTH: f64 = 398_600.441_8; // km³/s²
pub const R_EARTH: f64 = 6_371.0; // km
pub const R_MOON: f64 = 1_737.4; // km

// ── Raw state vector ─────────────────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVector {
    pub epoch: DateTime<Utc>,
    pub epoch_str: String,
    /// Position in EME2000 [km]
    pub pos: [f64; 3],
    /// Velocity in EME2000 [km/s]
    pub vel: [f64; 3],
}

impl StateVector {
    /// Distance from Earth center [km]
    #[inline]
    pub fn dist(&self) -> f64 {
        norm3(self.pos)
    }

    /// Speed [km/s]
    #[inline]
    pub fn speed(&self) -> f64 {
        norm3(self.vel)
    }

    /// Specific orbital energy [km²/s²]
    #[inline]
    pub fn energy(&self) -> f64 {
        0.5 * self.speed().powi(2) - MU_EARTH / self.dist()
    }

    /// Specific angular momentum vector [km²/s]
    pub fn h_vec(&self) -> [f64; 3] {
        cross3(self.pos, self.vel)
    }

    /// |h| magnitude
    pub fn h_mag(&self) -> f64 {
        norm3(self.h_vec())
    }

    /// Osculating eccentricity (scalar)
    pub fn eccentricity(&self) -> f64 {
        let e2 = 1.0 + 2.0 * self.energy() * self.h_mag().powi(2) / MU_EARTH.powi(2);
        if e2 < 0.0 {
            0.0
        } else {
            e2.sqrt()
        }
    }

    /// Osculating semi-major axis [km]  (negative → hyperbolic)
    pub fn sma(&self) -> f64 {
        -MU_EARTH / (2.0 * self.energy())
    }

    /// Inclination of osculating orbit [deg]
    pub fn inclination(&self) -> f64 {
        let h = self.h_vec();
        let h_norm = norm3(h);
        if h_norm < 1e-10 {
            return 0.0;
        }
        (h[2] / h_norm).acos().to_degrees()
    }

    /// Vis-viva speed at current radius [km/s]  (speed on same energy orbit)
    pub fn visviva_speed(&self) -> f64 {
        let a = self.sma();
        let inside = MU_EARTH * (2.0 / self.dist() - 1.0 / a);
        if inside < 0.0 {
            0.0
        } else {
            inside.sqrt()
        }
    }

    /// Flight path angle [deg]  (0 = circular, >0 = climbing)
    pub fn fpa(&self) -> f64 {
        let r_hat = unit3(self.pos);
        let rdot = dot3(self.vel, r_hat);
        rdot.atan2((self.speed().powi(2) - rdot.powi(2)).max(0.0).sqrt())
            .to_degrees()
    }
}

// ── Computed feature row (written to CSV) ────────────────────────────────────
#[derive(Debug, Clone, Serialize)]
pub struct FeatureRow {
    pub epoch: String,
    pub met_hours: f64, // Mission Elapsed Time from first vector
    pub pos_x_km: f64,
    pub pos_y_km: f64,
    pub pos_z_km: f64,
    pub vel_x_kms: f64,
    pub vel_y_kms: f64,
    pub vel_z_kms: f64,
    pub dist_km: f64,
    pub alt_km: f64, // dist - R_earth
    pub speed_kms: f64,
    pub energy: f64, // specific orbital energy
    pub h_mag: f64,  // angular momentum magnitude
    pub h_x: f64,
    pub h_y: f64,
    pub h_z: f64,
    pub ecc: f64,             // osculating eccentricity
    pub sma_km: f64,          // osculating semi-major axis
    pub inc_deg: f64,         // inclination
    pub visviva_kms: f64,     // vis-viva speed
    pub speed_residual: f64,  // speed - vis_viva  (≈0 on unperturbed orbit; ≠0 at burn)
    pub energy_residual: f64, // energy - initial_energy
    pub fpa_deg: f64,         // flight path angle
    pub accel_mag: f64,       // |acceleration| from finite diff [km/s²]
    pub jerk_mag: f64,        // |jerk| from finite diff [km/s³]
    pub delta_speed: f64,     // speed[i] - speed[i-1]
    pub delta_energy: f64,    // energy[i] - energy[i-1]
    pub anomaly_score: f64,   // rule-based anomaly score [0,1]
}

// ── OEM Metadata ─────────────────────────────────────────────────────────────
#[derive(Debug)]
pub struct OemMeta {
    pub object_name: String,
    pub object_id: String,
    pub ref_frame: String,
    pub time_system: String,
    pub start_time: String,
    pub stop_time: String,
    pub creation: String,
    pub originator: String,
}

// ── Parsed OEM ────────────────────────────────────────────────────────────────
pub struct ParsedOem {
    pub meta: OemMeta,
    pub vectors: Vec<StateVector>,
}

// ── Parser ────────────────────────────────────────────────────────────────────
pub fn parse_oem(content: &str) -> Result<ParsedOem> {
    let mut meta = OemMeta {
        object_name: String::new(),
        object_id: String::new(),
        ref_frame: String::new(),
        time_system: String::new(),
        start_time: String::new(),
        stop_time: String::new(),
        creation: String::new(),
        originator: String::new(),
    };
    let mut vectors: Vec<StateVector> = Vec::new();

    // Extract <pre>...</pre> block if HTML-wrapped
    let data_content = if content.contains("<pre>") {
        let start = content.find("<pre>").unwrap_or(0) + 5;
        let end = content.rfind("</pre>").unwrap_or(content.len());
        &content[start..end]
    } else {
        content
    };

    for line in data_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("CCSDS") {
            continue;
        }

        // Metadata
        if let Some(val) = kv(line, "OBJECT_NAME") {
            meta.object_name = val;
            continue;
        }
        if let Some(val) = kv(line, "OBJECT_ID") {
            meta.object_id = val;
            continue;
        }
        if let Some(val) = kv(line, "REF_FRAME") {
            meta.ref_frame = val;
            continue;
        }
        if let Some(val) = kv(line, "TIME_SYSTEM") {
            meta.time_system = val;
            continue;
        }
        if let Some(val) = kv(line, "START_TIME") {
            meta.start_time = val;
            continue;
        }
        if let Some(val) = kv(line, "STOP_TIME") {
            meta.stop_time = val;
            continue;
        }
        if let Some(val) = kv(line, "CREATION_DATE") {
            meta.creation = val;
            continue;
        }
        if let Some(val) = kv(line, "ORIGINATOR") {
            meta.originator = val;
            continue;
        }

        // State vector: EPOCH  X  Y  Z  Vx  Vy  Vz
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 7 && parts[0].contains('T') && parts[0].contains('-') {
            let epoch = parse_epoch(parts[0])?;
            let x = parts[1].parse::<f64>()?;
            let y = parts[2].parse::<f64>()?;
            let z = parts[3].parse::<f64>()?;
            let vx = parts[4].parse::<f64>()?;
            let vy = parts[5].parse::<f64>()?;
            let vz = parts[6].parse::<f64>()?;
            vectors.push(StateVector {
                epoch,
                epoch_str: parts[0].to_string(),
                pos: [x, y, z],
                vel: [vx, vy, vz],
            });
        }
    }

    if vectors.is_empty() {
        return Err(anyhow!("No state vectors found in OEM file"));
    }
    Ok(ParsedOem { meta, vectors })
}

// ── Feature computation ───────────────────────────────────────────────────────
pub fn compute_features(oem: &ParsedOem) -> Vec<FeatureRow> {
    let svs = &oem.vectors;
    let n = svs.len();
    let t0 = svs[0].epoch;
    let e0 = svs[0].energy();

    // Pre-compute time deltas and accelerations via finite differences
    let dt: Vec<f64> = (1..n)
        .map(|i| (svs[i].epoch - svs[i - 1].epoch).num_milliseconds() as f64 / 1000.0)
        .collect();

    // Acceleration from central differences (m/s² equivalent)
    let mut accel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut jerk: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    for i in 1..n - 1 {
        let h = 0.5 * (dt[i - 1] + dt[i]); // central step
        for k in 0..3 {
            accel[i][k] = (svs[i + 1].vel[k] - svs[i - 1].vel[k]) / (2.0 * h);
        }
    }
    // Forward/backward at ends
    if n > 1 {
        for k in 0..3 {
            accel[0][k] = (svs[1].vel[k] - svs[0].vel[k]) / dt[0];
            accel[n - 1][k] = (svs[n - 1].vel[k] - svs[n - 2].vel[k]) / dt[n - 2];
        }
    }
    for i in 1..n - 1 {
        let h = 0.5 * (dt[i - 1] + dt[i]);
        for k in 0..3 {
            jerk[i][k] = (accel[i + 1][k] - accel[i - 1][k]) / (2.0 * h);
        }
    }

    svs.iter()
        .enumerate()
        .map(|(i, sv)| {
            let met_h = (sv.epoch - t0).num_milliseconds() as f64 / 3_600_000.0;
            let energy = sv.energy();
            let h = sv.h_vec();
            let speed = sv.speed();
            let vv = sv.visviva_speed();

            // Rule-based anomaly: large speed residual or jerk
            let a_score = {
                let sr = (speed - vv).abs();
                let jm = norm3(jerk[i]);
                let er = (energy - e0).abs() / e0.abs().max(1e-9);
                let raw = (sr / 0.5).min(1.0) * 0.4
                    + (jm / 1e-4).min(1.0) * 0.4
                    + (er / 0.01).min(1.0) * 0.2;
                raw.min(1.0)
            };

            let prev_speed = if i > 0 { svs[i - 1].speed() } else { speed };
            let prev_energy = if i > 0 { svs[i - 1].energy() } else { energy };

            FeatureRow {
                epoch: sv.epoch_str.clone(),
                met_hours: met_h,
                pos_x_km: sv.pos[0],
                pos_y_km: sv.pos[1],
                pos_z_km: sv.pos[2],
                vel_x_kms: sv.vel[0],
                vel_y_kms: sv.vel[1],
                vel_z_kms: sv.vel[2],
                dist_km: sv.dist(),
                alt_km: sv.dist() - R_EARTH,
                speed_kms: speed,
                energy,
                h_mag: sv.h_mag(),
                h_x: h[0],
                h_y: h[1],
                h_z: h[2],
                ecc: sv.eccentricity(),
                sma_km: sv.sma(),
                inc_deg: sv.inclination(),
                visviva_kms: vv,
                speed_residual: speed - vv,
                energy_residual: energy - e0,
                fpa_deg: sv.fpa(),
                accel_mag: norm3(accel[i]),
                jerk_mag: norm3(jerk[i]),
                delta_speed: speed - prev_speed,
                delta_energy: energy - prev_energy,
                anomaly_score: a_score,
            }
        })
        .collect()
}

// ── Mission summary JSON ──────────────────────────────────────────────────────
#[derive(Serialize)]
pub struct MissionSummary {
    pub object_name: String,
    pub originator: String,
    pub start_time: String,
    pub stop_time: String,
    pub n_vectors: usize,
    pub duration_days: f64,
    pub max_dist_km: f64,
    pub max_alt_km: f64,
    pub apoapsis_epoch: String,
    pub min_speed_kms: f64,
    pub entry_speed_kms: f64,
    pub n_anomalies: usize,
    pub mean_energy: f64,
    pub energy_std: f64,
}

pub fn mission_summary(oem: &ParsedOem, features: &[FeatureRow]) -> MissionSummary {
    let n = features.len();
    let dur =
        (oem.vectors.last().unwrap().epoch - oem.vectors[0].epoch).num_seconds() as f64 / 86400.0;

    let max_dist_idx = features
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.dist_km.partial_cmp(&b.1.dist_km).unwrap())
        .unwrap()
        .0;
    let max_dist = features[max_dist_idx].dist_km;
    let apo_epoch = features[max_dist_idx].epoch.clone();

    let min_speed = features
        .iter()
        .map(|f| f.speed_kms)
        .fold(f64::INFINITY, f64::min);
    let entry_speed = features.last().map(|f| f.speed_kms).unwrap_or(0.0);

    let n_anom = features.iter().filter(|f| f.anomaly_score > 0.5).count();

    let mean_e = features.iter().map(|f| f.energy).sum::<f64>() / n as f64;
    let std_e = (features
        .iter()
        .map(|f| (f.energy - mean_e).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    MissionSummary {
        object_name: oem.meta.object_name.clone(),
        originator: oem.meta.originator.clone(),
        start_time: oem.meta.start_time.clone(),
        stop_time: oem.meta.stop_time.clone(),
        n_vectors: n,
        duration_days: dur,
        max_dist_km: max_dist,
        max_alt_km: max_dist - R_EARTH,
        apoapsis_epoch: apo_epoch,
        min_speed_kms: min_speed,
        entry_speed_kms: entry_speed,
        n_anomalies: n_anom,
        mean_energy: mean_e,
        energy_std: std_e,
    }
}

// ── Write helpers ─────────────────────────────────────────────────────────────
pub fn write_features_csv(features: &[FeatureRow], path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    for row in features {
        wtr.serialize(row)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn write_summary_json(summary: &MissionSummary, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(summary)?;
    std::fs::write(path, json)?;
    Ok(())
}

// ── Math helpers ──────────────────────────────────────────────────────────────
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}
#[inline]
pub fn unit3(v: [f64; 3]) -> [f64; 3] {
    let n = norm3(v).max(1e-30);
    [v[0] / n, v[1] / n, v[2] / n]
}
#[inline]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn kv(line: &str, key: &str) -> Option<String> {
    if line.starts_with(key) && line.contains('=') {
        let val = line.split_once('=')?.1.trim();
        Some(val.to_string())
    } else {
        None
    }
}

fn parse_epoch(s: &str) -> Result<DateTime<Utc>> {
    let s = s.trim().trim_end_matches('\r');
    // Handle both "2026-04-02T03:26:56.675" and "2026-04-02T03:26:56"
    let ndt = if s.contains('.') {
        let parts: Vec<&str> = s.splitn(2, '.').collect();
        let frac = parts.get(1).unwrap_or(&"0");
        let micro = format!("{:0<6}", &frac[..frac.len().min(6)]);
        NaiveDateTime::parse_from_str(&format!("{}.{}", parts[0], micro), "%Y-%m-%dT%H:%M:%S.%f")
    } else {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
    };
    ndt.map(|dt| Utc.from_utc_datetime(&dt))
        .map_err(|e| anyhow!("epoch parse error for '{}': {}", s, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn test_circular_orbit() {
        // ISS-like orbit at 400 km
        let r = R_EARTH + 400.0;
        let v_circ = (MU_EARTH / r).sqrt();
        let sv = StateVector {
            epoch: Utc::now(),
            epoch_str: "2026-01-01T00:00:00.000".to_string(),
            pos: [r, 0.0, 0.0],
            vel: [0.0, v_circ, 0.0],
        };
        let e = sv.eccentricity();
        assert!(e < 0.001, "circular orbit ecc={}", e);
        let speed_res = sv.speed() - sv.visviva_speed();
        assert!(speed_res.abs() < 0.001, "speed residual={}", speed_res);
    }

    #[test]
    fn test_parse_epoch() {
        let dt = parse_epoch("2026-04-02T03:26:56.675").unwrap();
        assert_eq!(dt.year(), 2026);
        assert_eq!(dt.month(), 4);
    }
}
