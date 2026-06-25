//! Mordor anomaly detection via TMAutoEncoder and TMCoalescedAutoEncoder.
//!
//! Trains an autoencoder on **benign-only** Sysmon events from the OTRF Mordor
//! Security-Datasets and uses per-sample reconstruction error as an anomaly score.
//! Attack events (mimikatz, PsExec, procdump, …) should be harder for the model
//! to reconstruct because it was only trained on normal OS-process patterns.
//!
//! ## What this does
//!
//! 1. Downloads and parses Mordor host datasets (same pipeline as sysmon_mordor).
//! 2. Fits an encoder vocabulary on ALL data so both benign and attack tokens are
//!    represented; trains the autoencoder on **benign-only** events (no labels).
//! 3. Computes per-sample reconstruction error = fraction of bits wrong → anomaly score.
//! 4. Reports anomaly detection metrics (accuracy/precision/recall) by scanning
//!    thresholds over reconstruction error.
//! 5. Shows which ECS token features are hardest to reconstruct for attack events.
//! 6. Repeats with TMCoalescedAutoEncoder for comparison.
//!
//! ## Run
//! ```text
//! cargo run --release --example mordor_autoencoder
//! ```

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, hive_of};

use std::{collections::HashSet, fs, path::Path};
use tmu_rs::{Encoder, Rng, TMAutoEncoder, TMCoalescedAutoEncoder};

// ── dataset discovery ─────────────────────────────────────────────────────────

const DATA_DIR: &str = "mordor_data";

const TACTIC_DIRS: &[&str] = &[
    "execution",
    "credential_access",
    "defense_evasion",
    "discovery",
    "lateral_movement",
    "persistence",
    "privilege_escalation",
    "collection",
];

// ── attack heuristic ──────────────────────────────────────────────────────────

const ATTACK_PROCS: &[&str] = &[
    "SharpView.exe",
    "powershell.exe",
    "netsh.exe",
    "wscript.exe",
    "python.exe",
    "whoami.exe",
    "cmd.exe",
    "mshta.exe",
    "vbc.exe",
    "vbscript.dll",
    "mimikatz.exe",
    "mimilib.dll",
    "Rubeus.exe",
    "kekeo.exe",
    "SharpDPAPI.exe",
    "SharpDump.exe",
    "procdump.exe",
    "procdump64.exe",
    "wce.exe",
    "fgdump.exe",
    "regsvcs.exe",
    "regasm.exe",
    "msbuild.exe",
    "cmstp.exe",
    "odbcconf.exe",
    "cscript.exe",
    "msiexec.exe",
    "PsExec.exe",
    "PsExec64.exe",
    "psexesvc.exe",
    "wmic.exe",
    "schtasks.exe",
    "at.exe",
    "ncat.exe",
    "nc.exe",
    "nmap.exe",
];

const SKIP_FIELDS: &[&str] = &[
    "ProcessGuid",
    "ParentProcessGuid",
    "SourceProcessGUID",
    "TargetProcessGUID",
    "LogonGuid",
    "ProviderGuid",
    "ProcessId",
    "ParentProcessId",
    "SourceProcessId",
    "TargetProcessId",
    "SourceThreadId",
    "ThreadID",
    "ExecutionProcessID",
    "LogonId",
    "TerminalSessionId",
    "SourcePort",
    "SourcePortName",
    "port",
    "@timestamp",
    "UtcTime",
    "TimeCreated",
    "SystemTime",
    "EventTime",
    "EventReceivedTime",
    "CreationUtcTime",
    "ParentCreationUtcTime",
    "@version",
    "CallTrace",
    "Hashes",
    "Details",
    "SourceModuleName",
    "SourceModuleType",
    "Channel",
    "Computer",
    "Hostname",
    "host",
    "Keywords",
    "Level",
    "Message",
    "Opcode",
    "Path",
    "RecordID",
    "EventRecordID",
    "RecordNumber",
    "SourceName",
    "Task",
    "Version",
    "ProcessID",
    "AccountName",
    "AccountType",
    "EventID",
];

const REGISTRY_HIVES: &[&str] = &[
    "HKLM\\",
    "HKCU\\",
    "HKU\\",
    "HKCR\\",
    "HKCC\\",
    "HKEY_LOCAL_MACHINE\\",
    "HKEY_CURRENT_USER\\",
    "HKEY_USERS\\",
    "HKEY_CLASSES_ROOT\\",
];

const PATH_LOCATIONS: &[(&str, &str)] = &[
    ("\\temp\\", "Temp"),
    ("/tmp/", "Temp"),
    ("\\users\\", "Users"),
    ("\\appdata\\", "AppData"),
    ("\\roaming\\", "Roaming"),
    ("\\local\\", "LocalAppData"),
    ("\\desktop\\", "Desktop"),
    ("\\downloads\\", "Downloads"),
    ("\\startup\\", "Startup"),
    ("\\system32\\", "System32"),
    ("\\syswow64\\", "SysWow64"),
    ("\\programdata\\", "ProgramData"),
    ("\\public\\", "Public"),
];

// ── field mapping and tokenization ────────────────────────────────────────────

fn to_ecs_field(sysmon: &str) -> &str {
    match sysmon {
        "Image" => "process.name",
        "ParentImage" => "process.parent.name",
        "User" => "user.name",
        "IntegrityLevel" => "process.integrity_level",
        "Company" => "process.pe.company",
        "Signed" => "process.code_signature.exists",
        "SignatureStatus" => "process.code_signature.status",
        "OriginalFileName" => "process.pe.original_file_name",
        "Description" => "process.pe.description",
        "Product" => "process.pe.product",
        "FileVersion" => "process.pe.file_version",
        "CurrentDirectory" => "process.working_directory",
        "SourceImage" => "source.process.name",
        "TargetImage" => "target.process.name",
        "GrantedAccess" => "target.process.granted_access",
        "ImageLoaded" => "dll.name",
        "DestinationPort" => "destination.port",
        "DestinationHostname" => "destination.hostname",
        "DestinationPortName" => "destination.service",
        "Protocol" => "network.transport",
        "Initiated" => "network.direction",
        "TargetFilename" => "file.name",
        "TargetObject" => "registry.path",
        "PipeName" => "pipe.name",
        "QueryName" => "dns.question.name",
        "QueryStatus" => "dns.response_code",
        "IsExecutable" => "file.executable",
        "EventType" => "event.action",
        "Device" => "device.id",
        "Archived" => "file.archived",
        other => other,
    }
}

fn file_path_tokens(path: &str) -> Vec<String> {
    let lower = path.to_lowercase();
    PATH_LOCATIONS
        .iter()
        .filter(|(seg, _)| lower.contains(seg))
        .map(|(_, cat)| format!("file.path::{cat}"))
        .collect()
}

fn cmd_tokens(cmdline: &str) -> Vec<String> {
    cmdline
        .split(|c: char| {
            c.is_whitespace()
                || matches!(c, ',' | ';' | '|' | '&' | '(' | ')' | '"' | '\'' | '`')
        })
        .map(|tok| tok.to_lowercase())
        .filter(|tok| tok.len() >= 3 && !tok.chars().all(|c| c.is_ascii_hexdigit()))
        .collect()
}

fn event_to_tokens(v: &serde_json::Value, eid: u32) -> Vec<String> {
    let mut t = vec![format!("event.id::{eid}")];
    let Some(obj) = v.as_object() else { return t };
    for (key, val) in obj {
        if SKIP_FIELDS.contains(&key.as_str()) {
            continue;
        }
        if key == "CommandLine" {
            if let Some(s) = val.as_str() {
                for tok in cmd_tokens(s) {
                    t.push(format!("process.args::{tok}"));
                }
            }
            continue;
        }
        if key == "ParentCommandLine" {
            if let Some(s) = val.as_str() {
                for tok in cmd_tokens(s) {
                    t.push(format!("process.parent.args::{tok}"));
                }
            }
            continue;
        }
        let s = match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => continue,
        };
        if s.is_empty() {
            continue;
        }
        let ecs = to_ecs_field(key);
        if REGISTRY_HIVES.iter().any(|h| s.starts_with(h)) {
            t.push(format!("{}::{}", ecs, hive_of(&s)));
        } else if s.contains('\\') || s.contains('/') {
            t.push(format!("{}::{}", ecs, basename(&s)));
            t.extend(file_path_tokens(&s));
        } else {
            t.push(format!("{}::{}", ecs, s));
        }
    }
    t
}

fn is_attack(v: &serde_json::Value, eid: u32) -> bool {
    let img_field = if eid == 10 { "SourceImage" } else { "Image" };
    let base = basename(v[img_field].as_str().unwrap_or(""));
    ATTACK_PROCS.iter().any(|p| base.eq_ignore_ascii_case(p))
}

// ── file parsing ──────────────────────────────────────────────────────────────

fn parse_file(path: &str) -> std::io::Result<Vec<(Vec<String>, bool)>> {
    let text = fs::read_to_string(path)?;
    let mut events = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if !v["Channel"]
            .as_str()
            .unwrap_or("")
            .starts_with("Microsoft-Windows-Sysmon")
        {
            continue;
        }
        let eid = v["EventID"].as_u64().unwrap_or(0) as u32;
        let attack = is_attack(&v, eid);
        events.push((event_to_tokens(&v, eid), attack));
    }
    Ok(events)
}

// ── download ──────────────────────────────────────────────────────────────────

fn discover_and_download() {
    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
    for tactic in TACTIC_DIRS {
        let api_url = format!(
            "https://api.github.com/repos/OTRF/Security-Datasets/contents/datasets/atomic/windows/{tactic}/host"
        );
        let out = std::process::Command::new("curl")
            .args([
                "-sL",
                "--max-time",
                "30",
                "-H",
                "User-Agent: sysmon-mordor/1.0",
                &api_url,
            ])
            .output()
            .expect("curl not available");
        let Ok(text) = std::str::from_utf8(&out.stdout) else {
            continue;
        };
        let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(text) else {
            eprintln!("  WARN: GitHub API error for {tactic} (rate-limited or no network)");
            continue;
        };
        for entry in &entries {
            let name = entry["name"].as_str().unwrap_or("");
            if !name.ends_with(".zip") {
                continue;
            }
            let stem = name.trim_end_matches(".zip");
            let url = entry["download_url"].as_str().unwrap_or("");
            if url.is_empty() {
                continue;
            }
            let zip_path = format!("{DATA_DIR}/{stem}.zip");
            if Path::new(&zip_path).exists() {
                continue;
            }
            print!("  [{tactic}] {stem}… ");
            let status = std::process::Command::new("curl")
                .args(["-sL", "--max-time", "120", "-o", &zip_path, url])
                .status()
                .expect("curl not available");
            if !status.success() {
                eprintln!("WARN: download failed");
                continue;
            }
            let status = std::process::Command::new("unzip")
                .args(["-o", &zip_path, "-d", DATA_DIR])
                .status()
                .expect("unzip not available");
            if status.success() {
                println!("ok");
            } else {
                eprintln!("WARN: unzip failed");
            }
        }
    }
}

fn collect_json_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("__MACOSX") {
                continue;
            }
            collect_json_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with("._") {
                out.push(path);
            }
        }
    }
}

// ── per-sample reconstruction error ──────────────────────────────────────────

/// Fraction of feature bits incorrectly reconstructed for one sample.
/// Returns a value in [0.0, 1.0]; higher = more anomalous.
///
/// Input bits are determined by checking whether each vocabulary token is
/// present in the input token set — equivalent to what the encoder packs, and
/// avoids accessing the private inner field of EncodedSample.
fn recon_error(ae: &TMAutoEncoder, enc: &Encoder, tokens: &[&str]) -> f64 {
    let sample = enc.encode_one_categorical(tokens);
    let recon = ae.reconstruct(&sample);
    let token_set: HashSet<&str> = tokens.iter().cloned().collect();
    let nf = ae.n_features();
    let errors: usize = (0..nf)
        .filter(|&o| {
            let present = token_set.contains(enc.vocab_token(o)) as u8;
            recon[o] != present
        })
        .count();
    errors as f64 / nf as f64
}

fn recon_error_coalesced(ae: &TMCoalescedAutoEncoder, enc: &Encoder, tokens: &[&str]) -> f64 {
    let sample = enc.encode_one_categorical(tokens);
    let recon = ae.reconstruct(&sample);
    let token_set: HashSet<&str> = tokens.iter().cloned().collect();
    let nf = ae.n_features();
    let errors: usize = (0..nf)
        .filter(|&o| {
            let present = token_set.contains(enc.vocab_token(o)) as u8;
            recon[o] != present
        })
        .count();
    errors as f64 / nf as f64
}

// ── anomaly detection metrics ─────────────────────────────────────────────────

/// Precision / recall / accuracy at a fixed threshold on (score, is_attack) pairs.
fn metrics_at(scores: &[(f64, bool)], threshold: f64) -> (f64, f64, f64) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for &(s, attack) in scores {
        match (s >= threshold, attack) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let accuracy = (tp + tn) as f64 / scores.len() as f64;
    (precision, recall, accuracy)
}

/// Sweep thresholds and return the one maximising balanced accuracy
/// (mean of sensitivity and specificity).
fn best_threshold(scores: &[(f64, bool)]) -> f64 {
    let mut best_t = 0.0f64;
    let mut best_bal = 0.0f64;
    let n_attack = scores.iter().filter(|&&(_, a)| a).count();
    let n_benign = scores.len() - n_attack;
    if n_attack == 0 || n_benign == 0 {
        return 0.0;
    }
    // Try every unique score as a threshold.
    let mut thresholds: Vec<f64> = scores.iter().map(|&(s, _)| s).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();
    for &t in &thresholds {
        let tp = scores
            .iter()
            .filter(|&&(s, a)| a && s >= t)
            .count() as f64;
        let tn = scores
            .iter()
            .filter(|&&(s, a)| !a && s < t)
            .count() as f64;
        let sensitivity = tp / n_attack as f64;
        let specificity = tn / n_benign as f64;
        let bal = (sensitivity + specificity) / 2.0;
        if bal > best_bal {
            best_bal = bal;
            best_t = t;
        }
    }
    best_t
}

// ── per-feature error profile ─────────────────────────────────────────────────

/// For each feature bit, compute the average reconstruction error across samples.
/// Returns (feature_index, avg_error) sorted descending — features most often
/// wrong for the given sample set.
fn per_feature_error(
    ae: &TMAutoEncoder,
    enc: &Encoder,
    all_tokens: &[Vec<String>],
) -> Vec<(usize, f64)> {
    let nf = ae.n_features();
    let mut errors = vec![0u64; nf];
    let n = all_tokens.len();
    for tokens in all_tokens {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let sample = enc.encode_one_categorical(&refs);
        let recon = ae.reconstruct(&sample);
        let token_set: HashSet<&str> = refs.iter().cloned().collect();
        for o in 0..nf {
            let actual = token_set.contains(enc.vocab_token(o)) as u8;
            if recon[o] != actual {
                errors[o] += 1;
            }
        }
    }
    if n == 0 {
        return vec![];
    }
    let mut ranked: Vec<(usize, f64)> = errors
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e as f64 / n as f64))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

// ── run one autoencoder experiment ────────────────────────────────────────────

fn run_vanilla(
    encoder: &Encoder,
    train_benign: &[Vec<String>],
    test_benign: &[Vec<String>],
    test_attack: &[Vec<String>],
) {
    println!("\n━━━ TMAutoEncoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let nf = encoder.n_features();
    let clauses_per_output = 40;
    let threshold = 50;
    let s = 5.0;

    println!(
        "config: n_features={nf}  clauses_per_output={clauses_per_output}  \
         threshold={threshold}  s={s}"
    );
    println!(
        "train on {} benign events (unsupervised, no labels)\n",
        train_benign.len()
    );

    // Encode training batch (benign only).
    let train_refs: Vec<Vec<&str>> = train_benign
        .iter()
        .map(|t| t.iter().map(String::as_str).collect())
        .collect();
    let train_slices: Vec<&[&str]> = train_refs.iter().map(|v| v.as_slice()).collect();
    let batch_train = encoder.encode_batch_categorical(&train_slices);

    let mut ae = TMAutoEncoder::with_config(nf, clauses_per_output, threshold, s, 8, true, 42);

    println!("{:>6}  {:>12}", "Epoch", "Train recon");
    for epoch in 1..=30 {
        ae.fit_epoch(&batch_train);
        if epoch % 5 == 0 || epoch == 1 {
            let tr = ae.reconstruction_accuracy(&batch_train);
            println!("{epoch:>6}  {tr:>12.4}");
        }
    }

    // Compute per-sample anomaly scores on test set.
    let mut scores: Vec<(f64, bool)> = Vec::new();
    for tokens in test_benign {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        scores.push((recon_error(&ae, encoder, &refs), false));
    }
    for tokens in test_attack {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        scores.push((recon_error(&ae, encoder, &refs), true));
    }

    let benign_scores: Vec<f64> = scores
        .iter()
        .filter(|&&(_, a)| !a)
        .map(|&(s, _)| s)
        .collect();
    let attack_scores: Vec<f64> = scores
        .iter()
        .filter(|&&(_, a)| a)
        .map(|&(s, _)| s)
        .collect();

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    println!(
        "\nmean recon error — benign: {:.4}  attack: {:.4}",
        mean(&benign_scores),
        mean(&attack_scores)
    );

    let t = best_threshold(&scores);
    let (prec, rec, acc) = metrics_at(&scores, t);
    println!(
        "best threshold: {t:.4}  →  accuracy={acc:.4}  precision={prec:.4}  recall={rec:.4}"
    );

    // Per-feature error profile: top features most often wrong for attack events.
    println!("\ntop-10 features with highest reconstruction error on attack test events:");
    let profile = per_feature_error(&ae, encoder, test_attack);
    for (feat, err) in profile.iter().take(10) {
        let tok = encoder.vocab_token(*feat);
        if !tok.contains("<UNK>") && !tok.contains("<OOV>") {
            println!("  {:.4}  {tok}", err);
        }
    }

    // Show clause rules for the output bit most often wrong on attack events.
    if let Some(&(top_bit, _)) = profile
        .iter()
        .find(|(f, _)| !encoder.vocab_token(*f).contains("<UNK>"))
    {
        println!(
            "\nclauses for output bit {top_bit} ('{}'):",
            encoder.vocab_token(top_bit)
        );
        for j in 0..ae.clauses_per_output().min(6) {
            let rule = ae.clause_rule(top_bit, j);
            if rule.is_empty() {
                continue;
            }
            let polarity = if ae.clause_is_positive(j) { "+" } else { "-" };
            let w = ae.clause_weight(top_bit, j);
            let lits: Vec<String> = rule
                .iter()
                .filter(|&&(_, neg)| !neg)
                .take(5)
                .map(|&(f, _)| encoder.vocab_token(f).to_string())
                .collect();
            if !lits.is_empty() {
                println!("  [{polarity}] w={w}  {}", lits.join("  "));
            }
        }
    }
}

fn run_coalesced(
    encoder: &Encoder,
    train_benign: &[Vec<String>],
    test_benign: &[Vec<String>],
    test_attack: &[Vec<String>],
) {
    println!(
        "\n━━━ TMCoalescedAutoEncoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    );

    let nf = encoder.n_features();
    let n_clauses = 200; // single shared bank across all output bits
    let threshold = 50;
    let s = 5.0;

    println!(
        "config: n_features={nf}  n_clauses={n_clauses} (shared)  \
         threshold={threshold}  s={s}"
    );
    println!(
        "train on {} benign events (unsupervised, no labels)\n",
        train_benign.len()
    );

    let train_refs: Vec<Vec<&str>> = train_benign
        .iter()
        .map(|t| t.iter().map(String::as_str).collect())
        .collect();
    let train_slices: Vec<&[&str]> = train_refs.iter().map(|v| v.as_slice()).collect();
    let batch_train = encoder.encode_batch_categorical(&train_slices);

    let mut ae =
        TMCoalescedAutoEncoder::with_config(nf, n_clauses, threshold, s, 8, true, 42);

    println!("{:>6}  {:>12}", "Epoch", "Train recon");
    for epoch in 1..=30 {
        ae.fit_epoch(&batch_train);
        if epoch % 5 == 0 || epoch == 1 {
            let tr = ae.reconstruction_accuracy(&batch_train);
            println!("{epoch:>6}  {tr:>12.4}");
        }
    }

    let mut scores: Vec<(f64, bool)> = Vec::new();
    for tokens in test_benign {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        scores.push((recon_error_coalesced(&ae, encoder, &refs), false));
    }
    for tokens in test_attack {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        scores.push((recon_error_coalesced(&ae, encoder, &refs), true));
    }

    let benign_scores: Vec<f64> = scores
        .iter()
        .filter(|&&(_, a)| !a)
        .map(|&(s, _)| s)
        .collect();
    let attack_scores: Vec<f64> = scores
        .iter()
        .filter(|&&(_, a)| a)
        .map(|&(s, _)| s)
        .collect();

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    println!(
        "\nmean recon error — benign: {:.4}  attack: {:.4}",
        mean(&benign_scores),
        mean(&attack_scores)
    );

    let t = best_threshold(&scores);
    let (prec, rec, acc) = metrics_at(&scores, t);
    println!(
        "best threshold: {t:.4}  →  accuracy={acc:.4}  precision={prec:.4}  recall={rec:.4}"
    );

    // Mixed-polarity clause stats (a coalesced-specific property).
    let mixed = (0..n_clauses)
        .filter(|&j| {
            let (has_pos, has_neg) = (0..nf).fold((false, false), |(p, n), o| {
                let w = ae.clause_weight(o, j);
                (p || w > 0, n || w < 0)
            });
            has_pos && has_neg
        })
        .count();
    println!(
        "\n{mixed}/{n_clauses} shared clauses have mixed polarity \
         (vote + for some output bits, - for others)"
    );
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Mordor Autoencoder — unsupervised anomaly detection on Sysmon events\n");
    println!("Downloading / verifying Mordor datasets…");
    discover_and_download();

    let mut json_files = Vec::new();
    collect_json_files(Path::new(DATA_DIR), &mut json_files);
    json_files.sort();

    let mut all_events: Vec<(Vec<String>, bool)> = Vec::new();
    for path in &json_files {
        match parse_file(path.to_str().unwrap()) {
            Ok(evs) => {
                let n = evs.len();
                if n > 0 {
                    println!(
                        "  {} → {n} events",
                        path.file_name().unwrap().to_str().unwrap()
                    );
                }
                all_events.extend(evs);
            }
            Err(e) => eprintln!("  WARN: {e}"),
        }
    }

    let n_attack = all_events.iter().filter(|(_, a)| *a).count();
    let n_benign = all_events.iter().filter(|(_, a)| !a).count();
    println!("\n{n_attack} attack events  {n_benign} benign events\n");

    if n_attack == 0 || n_benign == 0 {
        eprintln!("ERROR: one class is empty — check ATTACK_PROCS or dataset");
        std::process::exit(1);
    }

    // Shuffle deterministically, then split:
    //   80% of benign → train (autoencoder trained on benign only)
    //   20% of benign → test_benign
    //   all attack    → test_attack
    let mut rng = Rng::new(42);
    for i in (1..all_events.len()).rev() {
        let j = rng.below(i + 1);
        all_events.swap(i, j);
    }

    let mut benign: Vec<Vec<String>> = all_events
        .iter()
        .filter(|(_, a)| !a)
        .map(|(t, _)| t.clone())
        .collect();
    let attack: Vec<Vec<String>> = all_events
        .iter()
        .filter(|(_, a)| *a)
        .map(|(t, _)| t.clone())
        .collect();

    let cut = benign.len() * 4 / 5;
    let test_benign = benign.split_off(cut);
    let train_benign = benign;

    println!(
        "split: train_benign={}  test_benign={}  test_attack={}",
        train_benign.len(),
        test_benign.len(),
        attack.len()
    );

    // Build vocabulary from ALL events so attack tokens are representable.
    let all_refs: Vec<Vec<&str>> = all_events
        .iter()
        .map(|(t, _)| t.iter().map(String::as_str).collect())
        .collect();
    let all_slices: Vec<&[&str]> = all_refs.iter().map(|v| v.as_slice()).collect();
    let encoder = Encoder::fit_categorical(&all_slices);
    println!(
        "vocabulary: {} ECS features\n",
        encoder.n_features()
    );

    run_vanilla(&encoder, &train_benign, &test_benign, &attack);
    run_coalesced(&encoder, &train_benign, &test_benign, &attack);
}
