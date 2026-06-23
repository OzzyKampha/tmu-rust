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
//! 3. Labels each event using a within-trace heuristic: primary image in ATTACK_PROCS
//!    → label 1; background OS processes → label 0.
//! 4. Shuffles, 80/20 train/test split, trains a TM, reports accuracy every 5 epochs.
//! 5. Prints the vocabulary, top attack/benign rules, and per-sample clause explanations.
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

// ── dataset discovery ─────────────────────────────────────────────────────────

const DATA_DIR: &str = "mordor_data";

const TACTIC_DIRS: &[&str] = &[
    "execution", "credential_access", "defense_evasion", "discovery",
    "lateral_movement", "persistence", "privilege_escalation", "collection",
];

// ── attack heuristic ──────────────────────────────────────────────────────────

const ATTACK_PROCS: &[&str] = &[
    // Execution
    "SharpView.exe", "powershell.exe", "netsh.exe", "wscript.exe",
    "python.exe",    "whoami.exe",     "cmd.exe",   "mshta.exe",
    "vbc.exe",       "vbscript.dll",
    // Credential access
    "mimikatz.exe",   "mimilib.dll",    "Rubeus.exe",     "kekeo.exe",
    "SharpDPAPI.exe", "SharpDump.exe",  "procdump.exe",   "procdump64.exe",
    "wce.exe",        "fgdump.exe",
    // Defense evasion / execution via LOLBins rarely seen in benign traces
    "regsvcs.exe",  "regasm.exe",  "msbuild.exe", "cmstp.exe",
    "odbcconf.exe", "cscript.exe", "msiexec.exe",
    // Lateral movement
    "PsExec.exe", "PsExec64.exe", "psexesvc.exe", "wmic.exe",
    // Persistence helpers
    "schtasks.exe", "at.exe",
    // C2 / misc
    "ncat.exe", "nc.exe", "nmap.exe",
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

// ── dynamic discovery + download ──────────────────────────────────────────────

fn discover_and_download() {
    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
    for tactic in TACTIC_DIRS {
        let api_url = format!(
            "https://api.github.com/repos/OTRF/Security-Datasets/contents/datasets/atomic/windows/{tactic}/host"
        );
        let out = std::process::Command::new("curl")
            .args(["-sL", "--max-time", "30",
                   "-H", "User-Agent: sysmon-mordor/1.0",
                   &api_url])
            .output()
            .expect("curl not available");
        let Ok(text) = std::str::from_utf8(&out.stdout) else { continue };
        let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(text) else {
            eprintln!("  WARN: GitHub API error for {tactic} (rate-limited or no network)");
            continue;
        };
        for entry in &entries {
            let name = entry["name"].as_str().unwrap_or("");
            if !name.ends_with(".zip") { continue; }
            let stem = name.trim_end_matches(".zip");
            let url  = entry["download_url"].as_str().unwrap_or("");
            if url.is_empty() { continue; }

            let zip_path = format!("{DATA_DIR}/{stem}.zip");
            if Path::new(&zip_path).exists() { continue; }

            print!("  [{tactic}] {stem}… ");
            let status = std::process::Command::new("curl")
                .args(["-sL", "--max-time", "120", "-o", &zip_path, url])
                .status().expect("curl not available");
            if !status.success() { eprintln!("WARN: download failed"); continue; }
            let status = std::process::Command::new("unzip")
                .args(["-o", &zip_path, "-d", DATA_DIR])
                .status().expect("unzip not available");
            if status.success() { println!("ok"); } else { eprintln!("WARN: unzip failed"); }
        }
    }
}

// ── recursive JSON file collection ────────────────────────────────────────────

fn collect_json_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // skip macOS resource-fork artifact directories
            if path.file_name().and_then(|n| n.to_str()) == Some("__MACOSX") { continue; }
            collect_json_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            // skip macOS resource-fork files (start with "._")
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with("._") { out.push(path); }
        }
    }
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() {
    discover_and_download();

    let mut json_files = Vec::new();
    collect_json_files(Path::new(DATA_DIR), &mut json_files);
    json_files.sort();

    let mut all_events: Vec<(Vec<String>, usize)> = Vec::new();
    for path in &json_files {
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

    // Per-sample clause explanation.
    // For each class: (a) pick the first correct test sample and show its top firing
    // clauses by weight; (b) search all test samples of that class for the one whose
    // best multi-literal (≥2 positive literals) firing clause has the most literals.
    let firing_clauses = |class: usize, toks: &[String]| -> Vec<(i32, Vec<String>)> {
        let token_set: HashSet<&str> = toks.iter().map(String::as_str).collect();
        let mut out: Vec<(i32, Vec<String>)> = vec![];
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
                if pos_lits.is_empty() { continue; }
                out.push((tm.clause_weight(class, c), pos_lits));
            }
        }
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out
    };

    for (class, label) in [(1usize, "attack"), (0, "benign")] {
        // Representative sample: first correct prediction for this class
        let Some(idx) = test_y.iter().enumerate()
            .filter(|&(i, &y)| {
                if y != class { return false; }
                let s = encoder.encode_one_categorical(&te_inner[i]);
                tm.predict(&s) == class
            })
            .map(|(i, _)| i)
            .next()
        else { continue };

        let tokens: &[String] = test_tokens[idx];
        let token_set: HashSet<&str> = tokens.iter().map(String::as_str).collect();
        let sample = encoder.encode_one_categorical(&te_inner[idx]);
        let pred = tm.predict(&sample);
        let verdict = if pred == class { "correct" } else { "WRONG" };

        println!("\n--- {} sample explanation (pred={}, {}) ---", label, pred, verdict);
        let mut sorted_toks: Vec<&str> = token_set.iter().cloned().collect();
        sorted_toks.sort();
        println!("  tokens: {}", sorted_toks.join("  "));

        let firing = firing_clauses(class, tokens);
        if firing.is_empty() {
            println!("  (no positive clauses fired)");
        } else {
            println!("  top firing clauses ({} total):", firing.len());
            for (w, lits) in firing.iter().take(5) {
                println!("    w={}  {}", w, lits.join("  "));
            }
        }

        // Find the test sample (same class) whose best multi-literal firing clause
        // has the most positive literals.
        let best_multi = test_y.iter().enumerate()
            .filter(|&(_, &y)| y == class)
            .filter_map(|(i, _)| {
                firing_clauses(class, test_tokens[i])
                    .into_iter()
                    .filter(|(_, l)| l.len() >= 2)
                    .max_by_key(|(_, l)| l.len())
            })
            .max_by_key(|(_, l)| l.len());

        if let Some((w, lits)) = best_multi {
            println!("  richest AND clause across all {} test samples:", label);
            println!("    w={}  {}", w, lits.join("  "));
        }
    }
}
