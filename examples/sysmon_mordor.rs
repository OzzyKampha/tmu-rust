//! Mordor full-dataset demo — train the TM on **all Sysmon event types** from real
//! OTRF Security-Datasets traces (one Sysmon event = one TM sample, no windowing).
//!
//! ## What this does
//!
//! 1. Downloads four Mordor execution traces from GitHub into `mordor_data/`
//!    (skipped if already present).
//! 2. Parses every Sysmon event into ECS-named `col::val` tokens. Every field in the
//!    JSON is included (except high-cardinality noise like GUIDs/PIDs/timestamps).
//!    Paths → basename + `file.path::<dir>` location tokens.
//!    CommandLine/ParentCommandLine → word-level `process.args::<token>` tokens.
//! 3. Labels each event using a within-trace heuristic: primary image in ATTACK_PROCS
//!    → label 1; background OS processes → label 0.
//! 4. Shuffles, 80/20 train/test split, trains a TM, reports accuracy every 5 epochs.
//! 5. Prints the vocabulary and the top attack rule.
//!
//! ## Run
//! ```text
//! cargo run --release --example sysmon_mordor
//! ```

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, hive_of};

use std::{collections::HashSet, fs, path::Path};
use tmu_rs::{Encoder, Rng, TsetlinMachine};

// ── dataset URLs ───────────────────────────────────────────────────────────────

const DATASETS: &[(&str, &str)] = &[
    (
        "empire_launcher_vbs",
        "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/empire_launcher_vbs.zip",
    ),
    (
        "cmd_sharpview_pcre_net",
        "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/cmd_sharpview_pcre_net.zip",
    ),
    (
        "psh_powershell_httplistener",
        "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/psh_powershell_httplistener.zip",
    ),
    (
        "psh_python_webserver",
        "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/psh_python_webserver.zip",
    ),
];

const DATA_DIR: &str = "mordor_data";

// ── attack heuristic ──────────────────────────────────────────────────────────

const ATTACK_PROCS: &[&str] = &[
    "SharpView.exe", "powershell.exe", "netsh.exe",
    "wscript.exe",   "python.exe",     "whoami.exe",
    "cmd.exe",       "mshta.exe",      "vbc.exe",
    "vbscript.dll",
];

// ── field skip list (GUIDs, PIDs, timestamps, raw hashes, stack traces) ───────

const SKIP_FIELDS: &[&str] = &[
    // GUIDs
    "ProcessGuid", "ParentProcessGuid", "SourceProcessGUID", "TargetProcessGUID",
    "LogonGuid", "ProviderGuid",
    // Process / thread / session IDs
    "ProcessId", "ParentProcessId", "SourceProcessId", "TargetProcessId",
    "SourceThreadId", "ThreadID", "ExecutionProcessID", "LogonId", "TerminalSessionId",
    // Ephemeral port numbers (source/high ports change per connection)
    "SourcePort", "SourcePortName", "port",
    // Timestamps
    "@timestamp", "UtcTime", "TimeCreated", "SystemTime", "EventTime",
    "EventReceivedTime", "CreationUtcTime", "ParentCreationUtcTime",
    // Non-informative metadata
    "@version",
    // High-cardinality raw strings (CommandLine handled separately)
    "CallTrace", "Hashes", "Details",
    // NXLog / Syslog pipeline metadata (not part of the Sysmon event itself)
    "SourceModuleName", "SourceModuleType",
    // System / Winlog / NXLog metadata
    "Channel", "Computer", "Hostname", "host", "Keywords", "Level", "Message",
    "Opcode", "Path", "RecordID", "EventRecordID", "RecordNumber",
    "SourceName", "Task", "Version", "ProcessID",
    "AccountName", "AccountType",
    // EventID is emitted as the first token — skip in the loop
    "EventID",
];

const REGISTRY_HIVES: &[&str] = &[
    "HKLM\\", "HKCU\\", "HKU\\", "HKCR\\", "HKCC\\",
    "HKEY_LOCAL_MACHINE\\", "HKEY_CURRENT_USER\\",
    "HKEY_USERS\\", "HKEY_CLASSES_ROOT\\",
];

// ── ECS field-name mapping ────────────────────────────────────────────────────

fn to_ecs_field(sysmon: &str) -> &str {
    match sysmon {
        "Image"               => "process.name",
        "ParentImage"         => "process.parent.name",
        "User"                => "user.name",
        "IntegrityLevel"      => "process.integrity_level",
        "Company"             => "process.pe.company",
        "Signed"              => "process.code_signature.exists",
        "SignatureStatus"     => "process.code_signature.status",
        "OriginalFileName"    => "process.pe.original_file_name",
        "Description"         => "process.pe.description",
        "Product"             => "process.pe.product",
        "FileVersion"         => "process.pe.file_version",
        "CurrentDirectory"    => "process.working_directory",
        "SourceImage"         => "source.process.name",
        "TargetImage"         => "target.process.name",
        "GrantedAccess"       => "target.process.granted_access",
        "ImageLoaded"         => "dll.name",
        "DestinationPort"     => "destination.port",
        "DestinationHostname" => "destination.hostname",
        "DestinationPortName" => "destination.service",
        "Protocol"            => "network.transport",
        "Initiated"           => "network.direction",
        "TargetFilename"      => "file.name",
        "TargetObject"        => "registry.path",
        "PipeName"            => "pipe.name",
        "QueryName"           => "dns.question.name",
        "QueryStatus"         => "dns.response_code",
        "IsExecutable"        => "file.executable",
        "EventType"           => "event.action",
        "Device"              => "device.id",
        "Archived"            => "file.archived",
        other                 => other,
    }
}

// ── path → `file.path::<category>` tokens ────────────────────────────────────

const PATH_LOCATIONS: &[(&str, &str)] = &[
    ("\\temp\\",        "Temp"),
    ("/tmp/",           "Temp"),
    ("\\users\\",       "Users"),
    ("\\appdata\\",     "AppData"),
    ("\\roaming\\",     "Roaming"),
    ("\\local\\",       "LocalAppData"),
    ("\\desktop\\",     "Desktop"),
    ("\\downloads\\",   "Downloads"),
    ("\\startup\\",     "Startup"),
    ("\\system32\\",    "System32"),
    ("\\syswow64\\",    "SysWow64"),
    ("\\programdata\\", "ProgramData"),
    ("\\public\\",      "Public"),
];

fn file_path_tokens(path: &str) -> Vec<String> {
    let lower = path.to_lowercase();
    PATH_LOCATIONS.iter()
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
        if SKIP_FIELDS.contains(&key.as_str()) { continue; }

        // CommandLine fields → word-level tokens with ECS process.args prefix.
        if key == "CommandLine" {
            if let Some(s) = val.as_str() {
                for tok in cmd_tokens(s) { t.push(format!("process.args::{tok}")); }
            }
            continue;
        }
        if key == "ParentCommandLine" {
            if let Some(s) = val.as_str() {
                for tok in cmd_tokens(s) { t.push(format!("process.parent.args::{tok}")); }
            }
            continue;
        }

        let s = match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b)   => b.to_string(),
            _                            => continue,
        };
        if s.is_empty() { continue; }

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

// ── labeling heuristic ────────────────────────────────────────────────────────

fn is_attack(v: &serde_json::Value, eid: u32) -> bool {
    let img_field = if eid == 10 { "SourceImage" } else { "Image" };
    let base = basename(v[img_field].as_str().unwrap_or(""));
    ATTACK_PROCS.iter().any(|p| base.eq_ignore_ascii_case(p))
}

// ── parse one NDJSON file ──────────────────────────────────────────────────────

fn parse_file(path: &str) -> std::io::Result<Vec<(Vec<String>, usize)>> {
    let text = fs::read_to_string(path)?;
    let mut events = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if !v["Channel"].as_str().unwrap_or("").starts_with("Microsoft-Windows-Sysmon") {
            continue;
        }
        let eid = v["EventID"].as_u64().unwrap_or(0) as u32;
        let label = usize::from(is_attack(&v, eid));
        events.push((event_to_tokens(&v, eid), label));
    }
    Ok(events)
}

// ── download + unzip ───────────────────────────────────────────────────────────

fn ensure_datasets() {
    if Path::new(DATA_DIR).exists() {
        println!("mordor_data/ already present — skipping download");
        return;
    }
    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
    for (name, url) in DATASETS {
        print!("downloading {name}… ");
        let zip = format!("{DATA_DIR}/{name}.zip");
        let status = std::process::Command::new("curl")
            .args(["-sL", "--max-time", "60", "-o", &zip, url])
            .status()
            .expect("curl not available");
        if !status.success() {
            eprintln!("WARN: download failed for {name}, skipping");
            continue;
        }
        let status = std::process::Command::new("unzip")
            .args(["-o", &zip, "-d", DATA_DIR])
            .status()
            .expect("unzip not available");
        if status.success() { println!("ok"); } else { eprintln!("WARN: unzip failed for {name}"); }
    }
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() {
    ensure_datasets();

    let mut all_events: Vec<(Vec<String>, usize)> = Vec::new();
    for entry in fs::read_dir(DATA_DIR).expect("mordor_data/ not found") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
        match parse_file(path.to_str().unwrap()) {
            Ok(evs) => {
                println!("  {} → {} Sysmon events", path.file_name().unwrap().to_str().unwrap(), evs.len());
                all_events.extend(evs);
            }
            Err(e) => eprintln!("  WARN: {e}"),
        }
    }

    let n_attack = all_events.iter().filter(|(_, y)| *y == 1).count();
    let n_benign = all_events.iter().filter(|(_, y)| *y == 0).count();
    println!("\n{n_attack} attack events (label 1)  ← real attack-tool activity");
    println!("{n_benign} benign events (label 0)  ← real background OS activity\n");

    if n_attack == 0 || n_benign == 0 {
        eprintln!("ERROR: one class is empty — check ATTACK_PROCS or dataset contents");
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
    let (test_tokens,  test_y):  (Vec<_>, Vec<_>) = test_all.iter().map(|(t, y)| (t, *y)).unzip();

    let tr_inner: Vec<Vec<&str>> = train_tokens.iter().map(|t| t.iter().map(String::as_str).collect()).collect();
    let tr_refs: Vec<&[&str]>   = tr_inner.iter().map(|v| v.as_slice()).collect();
    let te_inner: Vec<Vec<&str>> = test_tokens.iter().map(|t| t.iter().map(String::as_str).collect()).collect();
    let te_refs: Vec<&[&str]>   = te_inner.iter().map(|v| v.as_slice()).collect();

    let encoder = Encoder::fit_categorical(&tr_refs);
    println!(
        "train={} test={} | vocabulary: {} ECS features\n",
        tr_refs.len(), te_refs.len(), encoder.n_features(),
    );

    let train_x = encoder.encode_batch_categorical(&tr_refs);
    let test_x  = encoder.encode_batch_categorical(&te_refs);

    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 80, 50, 5.0, 8, true, 42);
    for epoch in 1..=50 {
        tm.fit_epoch(&train_x, &train_y);
        if epoch % 5 == 0 || epoch == 1 {
            let tr_acc = tm.accuracy(&train_x, &train_y) * 100.0;
            let te_acc = tm.accuracy(&test_x, &test_y) * 100.0;
            println!("epoch {epoch:>2}  train={tr_acc:.2}%  test={te_acc:.2}%");
        }
    }

    // Vocabulary printout.
    println!("\n--- vocabulary ({} features) ---", encoder.n_features());
    for bit in 0..encoder.n_features() {
        let tok = encoder.vocab_token(bit);
        if !tok.contains("<UNK>") && !tok.contains("<OOV>") {
            println!("  {bit:>4}  {tok}");
        }
    }
    let n_sentinel = (0..encoder.n_features())
        .filter(|&b| { let t = encoder.vocab_token(b); t.contains("<UNK>") || t.contains("<OOV>") })
        .count();
    println!("  … plus {n_sentinel} <UNK>/<OOV> sentinels");

    // Top-5 rules per class: one compact line each, ranked by weight.
    // Prefer clauses referencing process/args/file tokens (more interpretable).
    let meaningful_prefixes = [
        "process.name", "process.args", "process.parent",
        "dll.name", "file.name", "file.path",
        "target.process", "source.process",
        "dns.question", "destination.",
    ];

    for (class, label) in [(1usize, "attack"), (0, "benign")] {
        println!("\n--- top {} rules (class {}) ---", label, class);
        let mut ranked: Vec<usize> = (0..tm.clauses_per_class())
            .filter(|&c| tm.clause_is_positive(c))
            .filter(|&c| tm.clause_rule(class, c).iter().any(|&(_, neg)| !neg))
            .collect();
        // Sort: most positive literals first, then has-meaningful-token, then weight
        ranked.sort_by(|&a, &b| {
            let pa = tm.clause_rule(class, a).iter().filter(|&&(_, neg)| !neg).count();
            let pb = tm.clause_rule(class, b).iter().filter(|&&(_, neg)| !neg).count();
            let ma = tm.clause_rule(class, a).iter().any(|&(f, neg)| {
                !neg && meaningful_prefixes.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            let mb = tm.clause_rule(class, b).iter().any(|&(f, neg)| {
                !neg && meaningful_prefixes.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            pb.cmp(&pa).then(mb.cmp(&ma)).then(tm.clause_weight(class, b).cmp(&tm.clause_weight(class, a)))
        });
        for (rank, &c) in ranked.iter().take(5).enumerate() {
            let rule = tm.clause_rule(class, c);
            let w = tm.clause_weight(class, c);
            let pos: Vec<_> = rule.iter().filter(|&&(_, neg)| !neg).collect();
            let neg_count = rule.len() - pos.len();
            let literals: Vec<String> = pos.iter()
                .map(|&&(feat, _)| encoder.vocab_token(feat).to_string())
                .collect();
            let neg_suffix = if neg_count > 0 {
                format!("  (+ {neg_count} NOT)")
            } else {
                String::new()
            };
            println!("  [{}] w={}  {}{}", rank + 1, w, literals.join("  "), neg_suffix);
        }
    }

    // Per-sample clause explanation: pick first test sample of each class and show
    // which positive clauses actually fired on it (all their literals were satisfied).
    for (class, label) in [(1usize, "attack"), (0, "benign")] {
        let Some(idx) = test_y.iter().position(|&y| y == class) else { continue };
        let tokens: &[String] = test_tokens[idx];
        let token_set: HashSet<&str> = tokens.iter().map(String::as_str).collect();

        let sample = encoder.encode_one_categorical(&te_inner[idx]);
        let pred = tm.predict(&sample);
        let verdict = if pred == class { "correct" } else { "WRONG" };

        println!("\n--- {} sample explanation (pred={}, {}) ---", label, pred, verdict);
        let mut sorted_toks: Vec<&str> = token_set.iter().cloned().collect();
        sorted_toks.sort();
        println!("  tokens: {}", sorted_toks.join("  "));

        let mut firing: Vec<(i32, Vec<String>)> = vec![];
        for c in 0..tm.clauses_per_class() {
            if !tm.clause_is_positive(c) { continue; }
            let rule = tm.clause_rule(class, c);
            if rule.is_empty() { continue; }
            let fires = rule.iter().all(|&(feat, is_neg)| {
                let present = token_set.contains(encoder.vocab_token(feat));
                if is_neg { !present } else { present }
            });
            if fires {
                let pos_lits: Vec<String> = rule.iter()
                    .filter(|&&(_, neg)| !neg)
                    .map(|&(f, _)| encoder.vocab_token(f).to_string())
                    .collect();
                if pos_lits.is_empty() { continue; } // all-NOT clauses aren't interpretable
                firing.push((tm.clause_weight(class, c), pos_lits));
            }
        }
        firing.sort_by(|a, b| b.0.cmp(&a.0));

        if firing.is_empty() {
            println!("  (no positive clauses fired)");
        } else {
            println!("  top firing clauses ({} total):", firing.len());
            for (w, lits) in firing.iter().take(5) {
                println!("    w={}  {}", w, lits.join("  "));
            }
        }
    }
}
