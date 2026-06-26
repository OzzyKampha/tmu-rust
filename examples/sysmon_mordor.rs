//! Mordor full-dataset demo — train the TM on **all Sysmon event types** from real
//! OTRF Security-Datasets traces (one Sysmon event = one TM sample, no windowing).
//!
//! ## What this does
//!
//! 1. Discovers all Mordor host datasets via the GitHub Contents API (8 ATT&CK tactics)
//!    and downloads any not yet in `mordor_data/` (skipped per-file if already present).
//! 2. Parses every Sysmon event into ECS-named `col::val` tokens. Every field in the
//!    JSON is included (except high-cardinality noise like GUIDs/PIDs/timestamps).
//!    Paths → basename + `file.path::<dir>` location tokens.
//!    CommandLine/ParentCommandLine → word-level `process.args::<token>` tokens.
//! 3. Labels every event in a file with the file's ATT&CK tactic (from the Mordor
//!    directory structure: execution/, credential_access/, etc.) — these ARE the
//!    Mordor dataset labels.  No per-event heuristics needed.
//! 4. Trains an 8-class TM (one class per tactic), 80/20 split, per-epoch accuracy.
//! 5. Reports per-tactic accuracy and top discriminating clause rules.
//!
//! ## Run
//! ```text
//! cargo run --release --example sysmon_mordor
//! ```

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, hive_of};

use std::{collections::HashMap, fs, io::Write, path::Path};
use tmu_rs::{CoalescedTsetlinMachine, Encoder, Rng};

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
    // GUIDs
    "ProcessGuid",
    "ParentProcessGuid",
    "SourceProcessGUID",
    "TargetProcessGUID",
    "LogonGuid",
    "ProviderGuid",
    // Process / thread / session IDs
    "ProcessId",
    "ParentProcessId",
    "SourceProcessId",
    "TargetProcessId",
    "SourceThreadId",
    "ThreadID",
    "ExecutionProcessID",
    "LogonId",
    "TerminalSessionId",
    // Ephemeral port numbers (source/high ports change per connection)
    "SourcePort",
    "SourcePortName",
    "port",
    // Timestamps
    "@timestamp",
    "UtcTime",
    "TimeCreated",
    "SystemTime",
    "EventTime",
    "EventReceivedTime",
    "CreationUtcTime",
    "ParentCreationUtcTime",
    // Non-informative metadata
    "@version",
    // High-cardinality raw strings (CommandLine handled separately)
    "CallTrace",
    "Hashes",
    "Details",
    // NXLog / Syslog pipeline metadata (not part of the Sysmon event itself)
    "SourceModuleName",
    "SourceModuleType",
    // System / Winlog / NXLog metadata
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
    // EventID is emitted as the first token — skip in the loop
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

// ── ECS field-name mapping ────────────────────────────────────────────────────

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

// ── path → `file.path::<category>` tokens ────────────────────────────────────

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

fn file_path_tokens(path: &str) -> Vec<String> {
    let lower = path.to_lowercase();
    PATH_LOCATIONS
        .iter()
        .filter(|(seg, _)| lower.contains(seg))
        .map(|(_, cat)| format!("file.path::{cat}"))
        .collect()
}

// ── CommandLine → word-level tokens ──────────────────────────────────────────

fn cmd_tokens(cmdline: &str) -> Vec<String> {
    cmdline
        .split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | ';' | '|' | '&' | '(' | ')' | '"' | '\'' | '`')
        })
        .map(|tok| tok.to_lowercase())
        .filter(|tok| tok.len() >= 3 && !tok.chars().all(|c| c.is_ascii_hexdigit()))
        .collect()
}

// ── per-event tokenizer (all fields, ECS names) ───────────────────────────────

fn event_to_tokens(v: &serde_json::Value, eid: u32) -> Vec<String> {
    let mut t = vec![format!("event.id::{eid}")];
    let Some(obj) = v.as_object() else { return t };

    for (key, val) in obj {
        if SKIP_FIELDS.contains(&key.as_str()) {
            continue;
        }

        // CommandLine fields → word-level tokens with ECS process.args prefix.
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

// ── parse one NDJSON file ──────────────────────────────────────────────────────

// All events in a file receive the file's tactic index as the label.
// This uses the Mordor dataset's own labels (tactic directory structure).
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

// ── dynamic discovery + download ──────────────────────────────────────────────

fn discover_and_download() {
    // Load any existing tactic map from previous runs.
    let mut tactic_map: HashMap<String, usize> = load_tactic_map();

    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
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
            // Record this scenario's tactic even if already downloaded.
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

    // Persist the tactic map so file parsing can look up each scenario's tactic.
    if let Ok(json) = serde_json::to_string_pretty(&tactic_map) {
        let _ = fs::write(TACTIC_MAP_FILE, json);
    }
}

// ── recursive JSON file collection ────────────────────────────────────────────

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

// ── main ───────────────────────────────────────────────────────────────────────

fn main() {
    println!("Mordor Tactic Classifier — 8-class ATT&CK tactic prediction from Sysmon events\n");
    println!("Downloading / verifying Mordor datasets…");
    discover_and_download();

    let tactic_map = load_tactic_map();
    if tactic_map.is_empty() {
        eprintln!("WARN: tactic map is empty — re-run to let the API populate it");
    }

    let mut json_files: Vec<(std::path::PathBuf, usize)> = Vec::new();
    collect_json_files(Path::new(DATA_DIR), &tactic_map, &mut json_files);
    json_files.sort_by_key(|(p, _)| p.clone());

    let mut all_events: Vec<(Vec<String>, usize)> = Vec::new();
    for (path, tactic_idx) in &json_files {
        match parse_file(path.to_str().unwrap(), *tactic_idx) {
            Ok(evs) => {
                let n = evs.len();
                if n > 0 {
                    let tactic = TACTIC_DIRS.get(*tactic_idx).unwrap_or(&"?");
                    println!(
                        "  [{tactic}] {} → {n} events",
                        path.file_name().unwrap().to_str().unwrap()
                    );
                }
                all_events.extend(evs);
            }
            Err(e) => eprintln!("  WARN: {e}"),
        }
    }

    println!();
    let mut tactic_counts = vec![0usize; TACTIC_DIRS.len()];
    for (_, y) in &all_events {
        tactic_counts[*y] += 1;
    }
    for (i, tactic) in TACTIC_DIRS.iter().enumerate() {
        println!("  {i}  {:<25}  {} events", tactic, tactic_counts[i]);
    }
    println!("     {:<25}  {} total\n", "", all_events.len());

    if all_events.is_empty() {
        eprintln!("ERROR: no events loaded — check dataset contents");
        std::process::exit(1);
    }

    // Shuffle, 80/20 split.
    let mut rng = Rng::new(42);
    for i in (1..all_events.len()).rev() {
        let j = rng.below(i + 1);
        all_events.swap(i, j);
    }
    let cut = all_events.len() * 4 / 5;
    let (train_all, test_all) = all_events.split_at(cut);

    let (train_tokens, train_y): (Vec<_>, Vec<_>) = train_all.iter().map(|(t, y)| (t, *y)).unzip();
    let (test_tokens, test_y): (Vec<_>, Vec<_>) = test_all.iter().map(|(t, y)| (t, *y)).unzip();

    let tr_inner: Vec<Vec<&str>> = train_tokens
        .iter()
        .map(|t| t.iter().map(String::as_str).collect())
        .collect();
    let tr_refs: Vec<&[&str]> = tr_inner.iter().map(|v| v.as_slice()).collect();
    let te_inner: Vec<Vec<&str>> = test_tokens
        .iter()
        .map(|t| t.iter().map(String::as_str).collect())
        .collect();
    let te_refs: Vec<&[&str]> = te_inner.iter().map(|v| v.as_slice()).collect();

    let encoder = Encoder::fit_categorical(&tr_refs);
    println!(
        "train={} test={} | vocabulary: {} ECS features\n",
        tr_refs.len(),
        te_refs.len(),
        encoder.n_features(),
    );

    let test_x = encoder.encode_batch_categorical(&te_refs);

    let n_tactics = TACTIC_DIRS.len();
    let n_train = tr_inner.len();

    // Build per-class index lists and print counts.
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_tactics];
    for (i, &y) in train_y.iter().enumerate() {
        class_indices[y].push(i);
    }
    println!("class training counts (balanced sampling per mini-batch):");
    for (i, tactic) in TACTIC_DIRS.iter().enumerate() {
        println!("  {i}  {tactic:<25}  train_count={:>6}", class_indices[i].len());
    }
    println!();

    // CoalescedTsetlinMachine: n_clauses shared across all classes.
    let mut tm = CoalescedTsetlinMachine::with_config(
        n_tactics, encoder.n_features(), 256, 50, 5.0, 8, true, 42,
    );
    let mut shuffle_rng = Rng::new(0xDEAD_BEEF);

    // Each mini-batch draws MINI_BATCH_SIZE/n_tactics samples per class (with
    // replacement for small classes), guaranteeing perfectly balanced batches.
    let per_class = MINI_BATCH_SIZE / n_tactics;
    let n_batches = n_train.div_ceil(MINI_BATCH_SIZE);

    for epoch in 1..=10 {
        let t0 = std::time::Instant::now();
        // Shuffle each class's index list independently.
        for ci in 0..n_tactics {
            let len = class_indices[ci].len();
            for i in (1..len).rev() {
                let j = shuffle_rng.below(i + 1);
                class_indices[ci].swap(i, j);
            }
        }
        for b in 0..n_batches {
            let bt0 = std::time::Instant::now();
            // Sample per_class items from each class (wrap around for small classes).
            let mut batch: Vec<usize> = Vec::with_capacity(MINI_BATCH_SIZE);
            for ci in 0..n_tactics {
                let ci_len = class_indices[ci].len();
                for k in 0..per_class {
                    batch.push(class_indices[ci][(b * per_class + k) % ci_len]);
                }
            }
            // Shuffle within batch to mix classes.
            for i in (1..batch.len()).rev() {
                let j = shuffle_rng.below(i + 1);
                batch.swap(i, j);
            }
            let slices: Vec<&[&str]> = batch.iter().map(|&i| tr_inner[i].as_slice()).collect();
            let mini_x = encoder.encode_batch_categorical(&slices);
            let mini_y: Vec<usize> = batch.iter().map(|&i| train_y[i]).collect();
            tm.fit_epoch(&mini_x, &mini_y);
            let evps = batch.len() as f64 / bt0.elapsed().as_secs_f64();
            if b % 10 == 9 || b + 1 == n_batches {
                println!(
                    "  epoch {epoch:>2}  batch {:>3}/{n_batches}  {evps:>7.0} ev/s",
                    b + 1
                );
                let _ = std::io::stdout().flush();
            }
        }
        let elapsed = t0.elapsed();
        let te_acc = tm.accuracy(&test_x, &test_y) * 100.0;
        println!(
            "epoch {epoch:>2}  test={te_acc:.2}%  ({:.1}s total)\n",
            elapsed.as_secs_f32()
        );
    }

    // Inference speed on pre-encoded test set (no tokenization cost).
    let infer_t0 = std::time::Instant::now();
    let _ = tm.accuracy(&test_x, &test_y);
    let infer_bulk_us = infer_t0.elapsed().as_secs_f64() * 1e6 / test_y.len() as f64;

    // Per-tactic accuracy — encode + predict from raw tokens (full pipeline latency).
    println!("\n--- per-tactic test accuracy ---");
    let fullpipe_t0 = std::time::Instant::now();
    for (class, tactic) in TACTIC_DIRS.iter().enumerate() {
        let indices: Vec<usize> = test_y
            .iter()
            .enumerate()
            .filter(|(_, &y)| y == class)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let correct = indices.iter().filter(|&&i| {
            let s = encoder.encode_one_categorical(&te_inner[i]);
            tm.predict(&s) == class
        }).count();
        println!(
            "  {class}  {:<25}  {correct}/{} ({:.1}%)",
            tactic,
            indices.len(),
            correct as f64 / indices.len() as f64 * 100.0
        );
    }
    let fullpipe_us = fullpipe_t0.elapsed().as_secs_f64() * 1e6 / test_y.len() as f64;

    println!(
        "\ninference speed ({} test events):",
        test_y.len()
    );
    println!(
        "  pre-encoded predict:  {:.2}µs/event  ({:.0} ev/s)",
        infer_bulk_us, 1e6 / infer_bulk_us
    );
    println!(
        "  encode + predict:     {:.2}µs/event  ({:.0} ev/s)",
        fullpipe_us, 1e6 / fullpipe_us
    );

    // Top-3 discriminating rules per tactic.
    // Filter to clauses with few total literals (general, interpretable rules).
    // Slide the cap up from tight→loose until we find at least 3 candidates.
    let meaningful_prefixes = [
        "process.name", "process.args", "process.parent",
        "dll.name", "file.name", "file.path",
        "target.process", "source.process",
        "dns.question", "destination.",
        "network.", "registry.",
    ];

    for (class, tactic) in TACTIC_DIRS.iter().enumerate() {
        let n_class = test_y.iter().filter(|&&y| y == class).count();
        if n_class == 0 { continue; }
        println!("\n--- top rules for {tactic} (class {class}) ---");

        // Try progressively relaxed caps on total literals (pos + neg).
        // This surfaces general rules first; falls back to specific ones if needed.
        let mut ranked: Vec<usize> = Vec::new();
        for &max_lits in &[30usize, 60, 120, 300, usize::MAX] {
            ranked = (0..tm.n_clauses())
                .filter(|&c| tm.clause_weight(class, c) > 0)
                .filter(|&c| {
                    let r = tm.clause_rule(c);
                    r.iter().any(|&(_, neg)| !neg) && r.len() <= max_lits
                })
                .collect();
            if ranked.len() >= 3 { break; }
        }
        // Sort: meaningful-prefix tokens first, then weight desc, then fewer total literals.
        ranked.sort_by(|&a, &b| {
            let wa = tm.clause_weight(class, a);
            let wb = tm.clause_weight(class, b);
            let la = tm.clause_rule(a).len();
            let lb = tm.clause_rule(b).len();
            let ma = tm.clause_rule(a).iter().any(|&(f, neg)| {
                !neg && meaningful_prefixes.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            let mb = tm.clause_rule(b).iter().any(|&(f, neg)| {
                !neg && meaningful_prefixes.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            mb.cmp(&ma).then(wb.cmp(&wa)).then(la.cmp(&lb))
        });

        for (rank, &c) in ranked.iter().take(3).enumerate() {
            let rule = tm.clause_rule(c);
            let w = tm.clause_weight(class, c);
            let pos: Vec<_> = rule.iter().filter(|&&(_, neg)| !neg).collect();
            let neg_count = rule.len() - pos.len();
            let literals: Vec<String> = pos.iter()
                .map(|&&(feat, _)| encoder.vocab_token(feat).to_string())
                .collect();
            let neg_sfx = if neg_count > 0 { format!("  (+ {neg_count} NOT)") } else { String::new() };
            println!("  [{}] w={w}  {}{neg_sfx}", rank + 1, literals.join("  "));
        }
    }
}
