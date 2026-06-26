//! Mordor anomaly detection via TMAutoEncoder and TMCoalescedAutoEncoder.
//!
//! Trains an autoencoder on Sysmon events from the OTRF Mordor Security-Datasets.
//! Every event is labeled by its file's ATT&CK tactic (the Mordor ground truth:
//! tactic directory = label).  Per-tactic reconstruction error shows which
//! tactics produce the most "unusual" behavior relative to the overall distribution.
//!
//! ## What this does
//!
//! 1. Downloads and parses Mordor host datasets (same pipeline as sysmon_mordor).
//! 2. Labels every event with its file's ATT&CK tactic index (from the Mordor
//!    directory structure: execution/, credential_access/, etc.).
//! 3. Fits an encoder vocabulary on ALL data; caps to top-1024 tokens.
//! 4. Trains the autoencoder on 80% of all events (unsupervised — no labels used).
//! 5. Reports per-tactic mean reconstruction error on the 20% test set.
//! 6. Shows which ECS token features are hardest to reconstruct overall.
//! 7. Repeats with TMCoalescedAutoEncoder for comparison.
//!
//! ## Run
//! ```text
//! cargo run --release --example mordor_autoencoder
//! ```

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, hive_of};

use std::{collections::HashMap, collections::HashSet, fs, io::Write, path::Path};
use tmu_rs::{Encoder, Rng, TMAutoEncoder, TMCoalescedAutoEncoder};

// Autoencoder memory = n_features² × clauses_per_output bytes.
// 16K raw features → 21 GB.  Cap to top-N by frequency to keep it < 100 MB.
const MAX_VOCAB: usize = 1024;

// Mini-batch size for training.
const MINI_BATCH_SIZE: usize = 4096;

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

// ── tactic map: ZIP stem → tactic index, persisted between runs ──────────────

const TACTIC_MAP_FILE: &str = "mordor_data/tactic_map.json";

fn load_tactic_map() -> HashMap<String, usize> {
    fs::read_to_string(TACTIC_MAP_FILE)
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, usize>>(&s).ok())
        .unwrap_or_default()
}

fn file_tactic_idx(json_name: &str, map: &HashMap<String, usize>) -> usize {
    map.iter()
        .find(|(stem, _)| json_name.starts_with(stem.as_str()))
        .map(|(_, &idx)| idx)
        .unwrap_or(0)
}

// ── field skip list (GUIDs, PIDs, timestamps, raw hashes, stack traces) ───────

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

// ── file parsing ──────────────────────────────────────────────────────────────

// All events in a file receive the file's tactic index as the label.
fn parse_file(path: &str, tactic_idx: usize) -> std::io::Result<Vec<(Vec<String>, usize)>> {
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
        events.push((event_to_tokens(&v, eid), tactic_idx));
    }
    Ok(events)
}

// ── download ──────────────────────────────────────────────────────────────────

fn discover_and_download() {
    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
    let mut tactic_map = load_tactic_map();
    for (tactic_idx, tactic) in TACTIC_DIRS.iter().enumerate() {
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
            tactic_map.entry(stem.to_string()).or_insert(tactic_idx);
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
    if let Ok(json) = serde_json::to_string_pretty(&tactic_map) {
        let _ = fs::write(TACTIC_MAP_FILE, json);
    }
}

fn collect_json_files(
    dir: &Path,
    tactic_map: &HashMap<String, usize>,
    out: &mut Vec<(std::path::PathBuf, usize)>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("__MACOSX") {
                continue;
            }
            collect_json_files(&path, tactic_map, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("._") || name == "tactic_map.json" {
                continue;
            }
            let tactic_idx = file_tactic_idx(name, tactic_map);
            out.push((path, tactic_idx));
        }
    }
}

// ── per-sample reconstruction error ──────────────────────────────────────────

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

// ── per-feature error profile ─────────────────────────────────────────────────

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
    train: &[Vec<String>],
    test: &[(Vec<String>, usize)],
) {
    println!("\n━━━ TMAutoEncoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let nf = encoder.n_features();
    let clauses_per_output = 40;
    let threshold = 50;
    let s = 5.0;

    println!(
        "config: n_features={nf}  clauses_per_output={clauses_per_output}  \
         threshold={threshold}  s={s}  mini_batch={MINI_BATCH_SIZE}"
    );
    println!("train on {} events (unsupervised)\n", train.len());

    let mut ae = TMAutoEncoder::with_config(nf, clauses_per_output, threshold, s, 8, true, 42);
    let mut shuffle_rng = Rng::new(0xDEAD_BEEF);

    let n_train = train.len();
    let n_batches = n_train.div_ceil(MINI_BATCH_SIZE);
    for epoch in 1..=5 {
        let mut order: Vec<usize> = (0..n_train).collect();
        for i in (1..n_train).rev() {
            let j = shuffle_rng.below(i + 1);
            order.swap(i, j);
        }
        for (b, chunk) in order.chunks(MINI_BATCH_SIZE).enumerate() {
            print!("  epoch {epoch}  batch {}/{n_batches}\r", b + 1);
            let _ = std::io::stdout().flush();
            let refs: Vec<Vec<&str>> = chunk
                .iter()
                .map(|&i| train[i].iter().map(String::as_str).collect())
                .collect();
            let slices: Vec<&[&str]> = refs.iter().map(|v| v.as_slice()).collect();
            let mini = encoder.encode_batch_categorical(&slices);
            ae.fit_epoch(&mini);
        }
        println!("  epoch {epoch} complete                    ");
    }

    // Per-tactic reconstruction error on test set.
    let n_tactics = TACTIC_DIRS.len();
    let mut tactic_errors: Vec<Vec<f64>> = vec![Vec::new(); n_tactics];
    for (tokens, tactic_idx) in test {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let err = recon_error(&ae, encoder, &refs);
        tactic_errors[*tactic_idx].push(err);
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };

    println!("\nper-tactic reconstruction error (higher = more unusual):");
    println!("{:<25}  {:>8}  {:>10}", "tactic", "n_test", "mean_err");
    println!("{}", "─".repeat(50));
    for (idx, tactic) in TACTIC_DIRS.iter().enumerate() {
        let errs = &tactic_errors[idx];
        println!("{:<25}  {:>8}  {:>10.4}", tactic, errs.len(), mean(errs));
    }

    // Top features most often wrong across all test events.
    let all_test_tokens: Vec<Vec<String>> = test.iter().map(|(t, _)| t.clone()).collect();
    println!("\ntop-10 features with highest reconstruction error on test events:");
    let profile = per_feature_error(&ae, encoder, &all_test_tokens);
    for (feat, err) in profile.iter().take(10) {
        let tok = encoder.vocab_token(*feat);
        if !tok.contains("<UNK>") && !tok.contains("<OOV>") {
            println!("  {:.4}  {tok}", err);
        }
    }

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
    train: &[Vec<String>],
    test: &[(Vec<String>, usize)],
) {
    println!(
        "\n━━━ TMCoalescedAutoEncoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    );

    let nf = encoder.n_features();
    let n_clauses = 40;
    let threshold = 50;
    let s = 5.0;

    println!(
        "config: n_features={nf}  n_clauses={n_clauses} (shared)  \
         threshold={threshold}  s={s}  mini_batch={MINI_BATCH_SIZE}"
    );
    println!("train on {} events (unsupervised)\n", train.len());

    let mut ae = TMCoalescedAutoEncoder::with_config(nf, n_clauses, threshold, s, 8, true, 42);
    let mut shuffle_rng = Rng::new(0xDEAD_BEEF);

    let n_train = train.len();
    let n_batches = n_train.div_ceil(MINI_BATCH_SIZE);
    for epoch in 1..=5 {
        let mut order: Vec<usize> = (0..n_train).collect();
        for i in (1..n_train).rev() {
            let j = shuffle_rng.below(i + 1);
            order.swap(i, j);
        }
        for (b, chunk) in order.chunks(MINI_BATCH_SIZE).enumerate() {
            print!("  epoch {epoch}  batch {}/{n_batches}\r", b + 1);
            let _ = std::io::stdout().flush();
            let refs: Vec<Vec<&str>> = chunk
                .iter()
                .map(|&i| train[i].iter().map(String::as_str).collect())
                .collect();
            let slices: Vec<&[&str]> = refs.iter().map(|v| v.as_slice()).collect();
            let mini = encoder.encode_batch_categorical(&slices);
            ae.fit_epoch(&mini);
        }
        println!("  epoch {epoch} complete                    ");
    }

    let n_tactics = TACTIC_DIRS.len();
    let mut tactic_errors: Vec<Vec<f64>> = vec![Vec::new(); n_tactics];
    for (tokens, tactic_idx) in test {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let err = recon_error_coalesced(&ae, encoder, &refs);
        tactic_errors[*tactic_idx].push(err);
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };

    println!("\nper-tactic reconstruction error (higher = more unusual):");
    println!("{:<25}  {:>8}  {:>10}", "tactic", "n_test", "mean_err");
    println!("{}", "─".repeat(50));
    for (idx, tactic) in TACTIC_DIRS.iter().enumerate() {
        let errs = &tactic_errors[idx];
        println!("{:<25}  {:>8}  {:>10.4}", tactic, errs.len(), mean(errs));
    }

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
    println!("Mordor Autoencoder — per-tactic reconstruction error on Sysmon events\n");
    println!("Downloading / verifying Mordor datasets…");
    discover_and_download();

    let tactic_map = load_tactic_map();
    let mut json_files: Vec<(std::path::PathBuf, usize)> = Vec::new();
    collect_json_files(Path::new(DATA_DIR), &tactic_map, &mut json_files);
    json_files.sort_by_key(|(p, _)| p.clone());

    let mut all_events: Vec<(Vec<String>, usize)> = Vec::new();
    let mut tactic_counts = vec![0usize; TACTIC_DIRS.len()];
    for (path, tactic_idx) in &json_files {
        match parse_file(path.to_str().unwrap(), *tactic_idx) {
            Ok(evs) => {
                let n = evs.len();
                if n > 0 {
                    tactic_counts[*tactic_idx] += n;
                    println!(
                        "  [{}] {} → {n} events",
                        TACTIC_DIRS[*tactic_idx],
                        path.file_name().unwrap().to_str().unwrap()
                    );
                }
                all_events.extend(evs);
            }
            Err(e) => eprintln!("  WARN: {e}"),
        }
    }

    println!("\nper-tactic event counts:");
    for (idx, tactic) in TACTIC_DIRS.iter().enumerate() {
        println!("  {idx}  {tactic:<25}  {}", tactic_counts[idx]);
    }
    println!("\ntotal: {} events\n", all_events.len());

    if all_events.is_empty() {
        eprintln!("ERROR: no events loaded — check dataset download");
        std::process::exit(1);
    }

    // Shuffle deterministically, then 80/20 split.
    let mut rng = Rng::new(42);
    for i in (1..all_events.len()).rev() {
        let j = rng.below(i + 1);
        all_events.swap(i, j);
    }

    let cut = all_events.len() * 4 / 5;
    let test_events = all_events.split_off(cut);
    let train_events = all_events;

    println!(
        "split: train={}  test={}",
        train_events.len(),
        test_events.len()
    );

    // Count token frequencies across ALL events to select top-K vocabulary.
    let mut freq: HashMap<String, usize> = HashMap::new();
    for (tokens, _) in train_events.iter().chain(test_events.iter()) {
        for tok in tokens {
            *freq.entry(tok.clone()).or_insert(0) += 1;
        }
    }
    let mut freq_vec: Vec<(String, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_tokens: HashSet<String> = freq_vec
        .iter()
        .take(MAX_VOCAB)
        .map(|(t, _)| t.clone())
        .collect();
    println!("top-{MAX_VOCAB} vocabulary tokens selected (raw vocab: {})", freq_vec.len());

    let filter_tokens = |tokens: &[String]| -> Vec<String> {
        tokens
            .iter()
            .filter(|t| top_tokens.contains(*t))
            .cloned()
            .collect()
    };

    let train_filtered: Vec<Vec<String>> = train_events
        .iter()
        .map(|(t, _)| filter_tokens(t))
        .collect();
    let test_filtered: Vec<(Vec<String>, usize)> = test_events
        .iter()
        .map(|(t, idx)| (filter_tokens(t), *idx))
        .collect();

    // Build encoder from all filtered tokens.
    let all_refs: Vec<Vec<&str>> = train_filtered
        .iter()
        .chain(test_filtered.iter().map(|(t, _)| t))
        .map(|t| t.iter().map(String::as_str).collect())
        .collect();
    let all_slices: Vec<&[&str]> = all_refs.iter().map(|v| v.as_slice()).collect();
    let encoder = Encoder::fit_categorical(&all_slices);
    println!("encoder vocabulary: {} features\n", encoder.n_features());

    run_vanilla(&encoder, &train_filtered, &test_filtered);
    run_coalesced(&encoder, &train_filtered, &test_filtered);
}
