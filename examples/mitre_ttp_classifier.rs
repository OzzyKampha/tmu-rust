//! MITRE ATT&CK TTP Classifier — technique-level prediction from Mordor Sysmon events
//!
//! This example goes beyond the tactic-level `sysmon_mordor` demo: it fetches the
//! official ATT&CK technique catalog from the MITRE CTI repository on GitHub, extracts
//! ATT&CK technique IDs embedded in Mordor/OTRF dataset filenames, and trains a Tsetlin
//! Machine to distinguish raw Sysmon events by ATT&CK **technique** (e.g. T1059
//! "Command and Scripting Interpreter", T1003 "OS Credential Dumping") rather than by
//! broad tactic category.
//!
//! ## Pipeline
//!
//! 1. Fetch `enterprise-attack.json` from `github.com/mitre/cti` (the canonical MITRE
//!    ATT&CK STIX bundle); parse technique IDs + names; cache a compact map to
//!    `mordor_data/mitre_techniques.json`.
//! 2. Download all OTRF/Security-Datasets host Sysmon captures for 8 ATT&CK tactic
//!    directories; skips ZIPs already present in `mordor_data/`.
//! 3. Scan every JSON file; extract the ATT&CK base technique ID (e.g. `T1059`) from
//!    the filename — sub-techniques (`T1059.001`) are rolled up to the parent.
//! 4. Keep only techniques whose Sysmon events total at least `MIN_EVENTS`.
//! 5. Train a `CoalescedTsetlinMachine` (shared clause bank, one vote per technique),
//!    80/20 train/test split, balanced mini-batches, 10 epochs.
//! 6. Report per-technique test accuracy with official MITRE technique names and
//!    top-5 annotated clause rules per class.
//!
//! ## Run
//! ```text
//! cargo run --release --example mitre_ttp_classifier
//! ```

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, hive_of};

use std::{collections::HashMap, fs, io::Write, path::Path};
use tmu_rs::{CoalescedTsetlinMachine, Encoder, Rng};

const DATA_DIR: &str = "mordor_data";
const MITRE_CACHE: &str = "mordor_data/mitre_techniques.json";
const MINI_BATCH_SIZE: usize = 4096;
/// Minimum Sysmon events a base technique must have to be included as a classifier class.
const MIN_EVENTS: usize = 200;

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

// ── field skip list (high-cardinality / ephemeral fields) ─────────────────────
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

// ── ECS field-name mapping ─────────────────────────────────────────────────────
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

// ── file path → location category tokens ─────────────────────────────────────
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
    PATH_LOCATIONS
        .iter()
        .filter(|(seg, _)| lower.contains(seg))
        .map(|(_, cat)| format!("file.path::{cat}"))
        .collect()
}

// ── CommandLine → word-level tokens ───────────────────────────────────────────
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

// ── per-event tokenizer ────────────────────────────────────────────────────────
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
            serde_json::Value::Bool(b)   => b.to_string(),
            _                            => continue,
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

// ── MITRE ATT&CK technique catalog fetch ──────────────────────────────────────
//
// Downloads the official enterprise-attack STIX bundle from github.com/mitre/cti,
// parses all non-revoked attack-pattern objects, and returns a map of technique
// ID → name.  A compact version is cached to mordor_data/mitre_techniques.json.

fn fetch_mitre_techniques() -> HashMap<String, String> {
    // Try cache first.
    if let Ok(data) = fs::read_to_string(MITRE_CACHE) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
            if !map.is_empty() {
                println!("  {} techniques loaded from cache ({})", map.len(), MITRE_CACHE);
                return map;
            }
        }
    }

    println!("  Fetching MITRE ATT&CK Enterprise bundle from github.com/mitre/cti …");
    let url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json";
    let output = std::process::Command::new("curl")
        .args([
            "-sL",
            "--max-time",
            "180",
            "-H",
            "User-Agent: tmu-ttp-classifier/1.0",
            url,
        ])
        .output()
        .expect("curl not available");

    if !output.status.success() || output.stdout.is_empty() {
        eprintln!(
            "  WARN: Failed to fetch MITRE ATT&CK data \
             (no network or proxy issue) — classifier runs without technique names"
        );
        return HashMap::new();
    }

    let text = match std::str::from_utf8(&output.stdout) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("  WARN: MITRE ATT&CK response is not valid UTF-8");
            return HashMap::new();
        }
    };

    let bundle: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  WARN: Failed to parse MITRE ATT&CK JSON: {e}");
            return HashMap::new();
        }
    };

    let mut techniques: HashMap<String, String> = HashMap::new();
    if let Some(objects) = bundle["objects"].as_array() {
        for obj in objects {
            if obj["type"].as_str() != Some("attack-pattern") {
                continue;
            }
            if obj["revoked"].as_bool() == Some(true)
                || obj["x_mitre_deprecated"].as_bool() == Some(true)
            {
                continue;
            }
            let name = obj["name"].as_str().unwrap_or("").to_string();
            if let Some(refs) = obj["external_references"].as_array() {
                for r in refs {
                    if r["source_name"].as_str() == Some("mitre-attack") {
                        if let Some(id) = r["external_id"].as_str() {
                            if id.starts_with('T') && id.len() >= 5 {
                                techniques.insert(id.to_string(), name.clone());
                                // Also populate base technique entry for sub-techniques.
                                if let Some(dot) = id.find('.') {
                                    techniques
                                        .entry(id[..dot].to_string())
                                        .or_insert_with(|| name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if techniques.is_empty() {
        eprintln!("  WARN: No techniques extracted — bundle structure may have changed");
        return HashMap::new();
    }

    println!("  → {} ATT&CK technique entries loaded", techniques.len());

    // Cache a compact version so subsequent runs are instant.
    if let Ok(json) = serde_json::to_string_pretty(&techniques) {
        if let Err(e) = fs::write(MITRE_CACHE, &json) {
            eprintln!("  WARN: Could not write technique cache: {e}");
        }
    }

    techniques
}

// ── ATT&CK technique ID extraction from filenames ─────────────────────────────
//
// Scans `s` for `T\d{4}` patterns (optionally followed by `.\d{3}` sub-technique).
// Returns the last match, stripped of any sub-technique suffix, so both
// "psh_invoke_mimikatz_T1003.001.json" and "EMPIRE-T1059.003_eval.json" yield
// a clean base ID ("T1003", "T1059").

fn extract_technique_id(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut found: Option<[usize; 2]> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'T'
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
            && i + 5 <= bytes.len()
            && bytes[i + 1..i + 5].iter().all(|b| b.is_ascii_digit())
        {
            let base_end = i + 5;
            // Optional sub-technique: .NNN (exactly 3 digits, not followed by more digits)
            let full_end = if base_end + 4 <= bytes.len()
                && bytes[base_end] == b'.'
                && bytes[base_end + 1..base_end + 4].iter().all(|b| b.is_ascii_digit())
                && (base_end + 4 >= bytes.len() || !bytes[base_end + 4].is_ascii_digit())
            {
                base_end + 4
            } else {
                base_end
            };
            found = Some([i, full_end]);
            i = full_end;
            continue;
        }
        i += 1;
    }
    found.map(|[start, end]| {
        let id = &s[start..end];
        // Always return base technique (strip .NNN)
        match id.find('.') {
            Some(dot) => id[..dot].to_string(),
            None => id.to_string(),
        }
    })
}

// ── Mordor dataset discovery + download ───────────────────────────────────────
//
// Walks all 8 ATT&CK tactic directories in OTRF/Security-Datasets via the GitHub
// Contents API; downloads ZIP archives not already present; extracts them.

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
                "User-Agent: tmu-ttp-classifier/1.0",
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
            let _ = std::io::stdout().flush();
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

// ── recursive JSON file collection ────────────────────────────────────────────

fn collect_files(dir: &Path, out: &mut Vec<(std::path::PathBuf, String)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("__MACOSX") {
                collect_files(&path, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip metadata files and macOS artifacts.
            if name.starts_with("._")
                || name.ends_with("_techniques.json")
                || name == "tactic_map.json"
            {
                continue;
            }
            if let Some(tid) = extract_technique_id(name) {
                out.push((path, tid));
            }
            // Files with no T-ID in their name have no label → skip.
        }
    }
}

// ── token annotation ──────────────────────────────────────────────────────────

fn explain_token(tok: &str) -> &'static str {
    if tok == "event.id::1"  { return "Sysmon: process creation"; }
    if tok == "event.id::3"  { return "Sysmon: network connection"; }
    if tok == "event.id::7"  { return "Sysmon: image/DLL loaded"; }
    if tok.starts_with("event.id::10") { return "Sysmon: process access (injection / cred dump)"; }
    if tok.starts_with("event.id::11") { return "Sysmon: file created"; }
    if tok.starts_with("event.id::12") || tok.starts_with("event.id::13") {
        return "Sysmon: registry create/set";
    }
    if tok.starts_with("event.id::22") { return "Sysmon: DNS query"; }
    if tok.starts_with("event.id::25") { return "Sysmon: process tampering"; }
    if tok.starts_with("process.name::")        { return "executing process binary"; }
    if tok.starts_with("process.args::")        { return "command-line argument"; }
    if tok.starts_with("process.parent.name::") { return "parent process"; }
    if tok.starts_with("target.process.name::") { return "victim process (accessed/injected)"; }
    if tok.starts_with("source.process.name::") { return "injecting/accessing process"; }
    if tok.starts_with("target.process.granted_access::") { return "access rights (PROCESS_VM_READ etc.)"; }
    if tok.contains("file.path::Temp")     { return "staging in Temp (common dropper location)"; }
    if tok.contains("file.path::System32") { return "system directory (legit or DLL hijack)"; }
    if tok.contains("file.path::AppData")  { return "user profile staging area"; }
    if tok.starts_with("file.path::")      { return "file location category"; }
    if tok.starts_with("file.name::")      { return "file name"; }
    if tok.starts_with("dll.name::")       { return "DLL loaded"; }
    if tok.starts_with("destination.")     { return "outbound C2 / lateral movement target"; }
    if tok.starts_with("dns.question::")   { return "DNS lookup (C2 beacon / lateral)"; }
    if tok.starts_with("registry.")        { return "registry operation"; }
    if tok.contains("::lsass")             { return "LSASS — credential store (high value target)"; }
    if tok.contains("::mimikatz")          { return "Mimikatz credential dumper"; }
    if tok.contains("::powershell")        { return "PowerShell execution"; }
    if tok.contains("::cmd.exe")           { return "Command prompt"; }
    if tok.contains("::rundll32")          { return "LOLBin: arbitrary DLL runner"; }
    if tok.contains("::regsvr32")          { return "LOLBin: COM/SCT payload runner"; }
    if tok.contains("::mshta")             { return "LOLBin: HTML/VBS/JS runner"; }
    if tok.contains("::wmic")             { return "WMI execution / lateral movement"; }
    if tok.contains("::schtasks")          { return "scheduled task (persistence)"; }
    if tok.contains("::net.exe") || tok.contains("::net1.exe") {
        return "domain/user enumeration";
    }
    ""
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("MITRE ATT&CK TTP Classifier — technique-level prediction from Mordor Sysmon events\n");

    fs::create_dir_all(DATA_DIR).ok();

    // ── Step 1: load MITRE ATT&CK technique catalog ───────────────────────────
    println!("Step 1: MITRE ATT&CK Enterprise technique catalog");
    let techniques = fetch_mitre_techniques();
    if techniques.is_empty() {
        eprintln!("  WARN: No technique names available — output will show T-IDs only");
    }

    // ── Step 2: download Mordor datasets ─────────────────────────────────────
    println!("\nStep 2: Mordor dataset download / verification");
    discover_and_download();

    // ── Step 3: collect labeled files ─────────────────────────────────────────
    println!("\nStep 3: Scanning dataset files for ATT&CK technique IDs…");
    let mut raw_files: Vec<(std::path::PathBuf, String)> = Vec::new();
    collect_files(Path::new(DATA_DIR), &mut raw_files);
    raw_files.sort_by_key(|(p, _)| p.clone());

    if raw_files.is_empty() {
        eprintln!("ERROR: No T-ID-labeled JSON files found. Check that data downloaded correctly.");
        std::process::exit(1);
    }

    // ── Step 4: parse events, group by base technique ─────────────────────────
    println!("\nStep 4: Parsing Sysmon events…");
    let mut technique_events: HashMap<String, Vec<Vec<String>>> = HashMap::new();

    for (path, tid) in &raw_files {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let mut count = 0usize;
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
            technique_events
                .entry(tid.clone())
                .or_default()
                .push(event_to_tokens(&v, eid));
            count += 1;
        }
        if count > 0 {
            let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
            println!(
                "  {tid}  {tname:<50}  {:>5} events  — {}",
                count,
                path.file_name().unwrap().to_str().unwrap()
            );
        }
    }

    // ── Step 5: select qualifying techniques ─────────────────────────────────
    let mut all_tids: Vec<(String, usize)> = technique_events
        .iter()
        .map(|(tid, evs)| (tid.clone(), evs.len()))
        .collect();
    all_tids.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    println!("\nAll techniques found (sorted by event count):");
    println!("  {:<8}  {:<50}  {:>7}  {}", "TID", "Name", "events", "status");
    for (tid, count) in &all_tids {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        let status = if *count >= MIN_EVENTS { "included" } else { "too few — skipped" };
        println!("  {tid:<8}  {tname:<50}  {:>7}  {status}", count);
    }

    let mut class_labels: Vec<String> = Vec::new();
    let mut technique_to_class: HashMap<String, usize> = HashMap::new();
    for (tid, count) in &all_tids {
        if *count >= MIN_EVENTS {
            technique_to_class.insert(tid.clone(), class_labels.len());
            class_labels.push(tid.clone());
        }
    }

    if class_labels.is_empty() {
        eprintln!(
            "\nERROR: No techniques with >= {MIN_EVENTS} events. \
             Lower MIN_EVENTS or re-run after downloading more Mordor data."
        );
        std::process::exit(1);
    }

    let n_classes = class_labels.len();
    println!("\n{n_classes} technique classes selected (MIN_EVENTS={MIN_EVENTS}):");
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        println!(
            "  class {:>2}  {tid}  {tname:<50}  {} events",
            cls,
            technique_events[tid].len()
        );
    }

    // ── Step 6: build labeled dataset ─────────────────────────────────────────
    let mut all_events: Vec<(Vec<String>, usize)> = Vec::new();
    for (cls, tid) in class_labels.iter().enumerate() {
        if let Some(evs) = technique_events.get(tid) {
            for tokens in evs {
                all_events.push((tokens.clone(), cls));
            }
        }
    }
    println!("\nTotal labeled events: {}", all_events.len());

    // ── Step 7: shuffle + 80/20 train/test split ──────────────────────────────
    let mut rng = Rng::new(42);
    for i in (1..all_events.len()).rev() {
        let j = rng.below(i + 1);
        all_events.swap(i, j);
    }
    let cut = all_events.len() * 4 / 5;
    let (train_all, test_all) = all_events.split_at(cut);

    let (train_tokens, train_y): (Vec<_>, Vec<_>) =
        train_all.iter().map(|(t, y)| (t, *y)).unzip();
    let (test_tokens, test_y): (Vec<_>, Vec<_>) =
        test_all.iter().map(|(t, y)| (t, *y)).unzip();

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
    let test_x = encoder.encode_batch_categorical(&te_refs);

    println!(
        "train={} test={} | vocabulary: {} ECS tokens\n",
        tr_refs.len(),
        te_refs.len(),
        encoder.n_features()
    );

    // Per-class index lists for balanced mini-batch sampling.
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &y) in train_y.iter().enumerate() {
        class_indices[y].push(i);
    }
    println!("class training distribution:");
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        println!(
            "  class {:>2}  {tid}  {tname:<50}  train={}",
            cls,
            class_indices[cls].len()
        );
    }
    println!();

    // ── Step 8: train CoalescedTsetlinMachine ─────────────────────────────────
    let mut tm = CoalescedTsetlinMachine::with_config(
        n_classes,
        encoder.n_features(),
        256,  // shared clause bank size
        50,   // threshold T
        5.0,  // specificity s
        8,    // TA state bits
        true, // boost true positives
        42,
    );
    let mut shuffle_rng = Rng::new(0xDEAD_BEEF);

    // Each mini-batch draws per_class samples from each class (balanced).
    let per_class = (MINI_BATCH_SIZE / n_classes).max(1);
    let n_batches = tr_inner.len().div_ceil(MINI_BATCH_SIZE);

    for epoch in 1..=10 {
        let t0 = std::time::Instant::now();

        // Shuffle each class list independently.
        for ci in 0..n_classes {
            let len = class_indices[ci].len();
            if len < 2 {
                continue;
            }
            for i in (1..len).rev() {
                let j = shuffle_rng.below(i + 1);
                class_indices[ci].swap(i, j);
            }
        }

        for b in 0..n_batches {
            let bt0 = std::time::Instant::now();
            let mut batch: Vec<usize> = Vec::with_capacity(per_class * n_classes);
            for ci in 0..n_classes {
                let ci_len = class_indices[ci].len();
                if ci_len == 0 {
                    continue;
                }
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
            "epoch {epoch:>2}  test={te_acc:.2}%  ({:.1}s)\n",
            elapsed.as_secs_f32()
        );
    }

    // ── Step 9: per-technique accuracy ────────────────────────────────────────
    println!("\n--- per-technique test accuracy ---");
    let infer_t0 = std::time::Instant::now();
    let _ = tm.accuracy(&test_x, &test_y);
    let infer_us = infer_t0.elapsed().as_secs_f64() * 1e6 / test_y.len() as f64;

    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        let indices: Vec<usize> = test_y
            .iter()
            .enumerate()
            .filter(|(_, &y)| y == cls)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let correct = indices
            .iter()
            .filter(|&&i| {
                let s = encoder.encode_one_categorical(&te_inner[i]);
                tm.predict(&s) == cls
            })
            .count();
        println!(
            "  class {:>2}  {tid}  {tname:<50}  {correct}/{} ({:.1}%)",
            cls,
            indices.len(),
            correct as f64 / indices.len() as f64 * 100.0
        );
    }
    println!(
        "\nbulk inference: {infer_us:.2}µs/event  ({:.0} ev/s)",
        1e6 / infer_us
    );

    // ── Step 10: clause statistics ─────────────────────────────────────────────
    let meaningful = [
        "process.name",
        "process.args",
        "process.parent",
        "dll.name",
        "file.name",
        "file.path",
        "target.process",
        "source.process",
        "dns.question",
        "destination.",
        "network.",
        "registry.",
    ];

    println!("\n━━━ clause statistics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!(
        "{:<8}  {:<50}  {:>10}  {:>10}  {:>9}  {:>6}",
        "TID", "name", "pos_clauses", "neg_clauses", "mean_w_pos", "max_w"
    );
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        let pos: Vec<i32> = (0..tm.n_clauses())
            .filter_map(|c| {
                let w = tm.clause_weight(cls, c);
                if w > 0 { Some(w) } else { None }
            })
            .collect();
        let neg_cnt = (0..tm.n_clauses())
            .filter(|&c| tm.clause_weight(cls, c) < 0)
            .count();
        let mean_w = if pos.is_empty() {
            0.0
        } else {
            pos.iter().sum::<i32>() as f64 / pos.len() as f64
        };
        let max_w = pos.iter().copied().max().unwrap_or(0);
        println!(
            "{tid:<8}  {tname:<50}  {:>10}  {:>10}  {:>9.1}  {:>6}",
            pos.len(),
            neg_cnt,
            mean_w,
            max_w
        );
    }

    // Token frequency across all positively-weighted clauses.
    let mut token_freq: HashMap<usize, usize> = HashMap::new();
    let mut token_class_mask: HashMap<usize, u64> = HashMap::new();
    for cls in 0..n_classes {
        for c in 0..tm.n_clauses() {
            if tm.clause_weight(cls, c) > 0 {
                for &(feat, neg) in tm.clause_rule(c).iter() {
                    if !neg {
                        *token_freq.entry(feat).or_insert(0) += 1;
                        *token_class_mask.entry(feat).or_insert(0) |= 1u64 << cls;
                    }
                }
            }
        }
    }
    let mut freq_vec: Vec<(usize, usize)> = token_freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\ntop-20 most common positive literal tokens (across all classes × clauses):");
    println!("  {:>5}  {:>4}  token", "count", "cls");
    for &(feat, count) in freq_vec.iter().take(20) {
        let n_cls = token_class_mask[&feat].count_ones();
        println!("  {:>5}  {:>4}  {}", count, n_cls, encoder.vocab_token(feat));
    }

    // ── Step 11: top-5 rules per technique ────────────────────────────────────
    println!("\n━━━ top rules per ATT&CK technique ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = techniques.get(tid).map(|n| n.as_str()).unwrap_or("(unknown)");
        let n_test = test_y.iter().filter(|&&y| y == cls).count();
        if n_test == 0 {
            continue;
        }
        println!("── {tid}  {tname}  (class {cls}) ──");

        // Select positive clauses; prefer short, attack-relevant, high-weight.
        let mut ranked: Vec<usize> = Vec::new();
        for &max_lits in &[30usize, 60, 120, 300, usize::MAX] {
            ranked = (0..tm.n_clauses())
                .filter(|&c| tm.clause_weight(cls, c) > 0)
                .filter(|&c| {
                    let r = tm.clause_rule(c);
                    r.iter().any(|&(_, neg)| !neg) && r.len() <= max_lits
                })
                .collect();
            if ranked.len() >= 5 {
                break;
            }
        }
        ranked.sort_by(|&a, &b| {
            let wa = tm.clause_weight(cls, a);
            let wb = tm.clause_weight(cls, b);
            let la = tm.clause_rule(a).len();
            let lb = tm.clause_rule(b).len();
            let ma = tm.clause_rule(a).iter().any(|&(f, neg)| {
                !neg && meaningful.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            let mb = tm.clause_rule(b).iter().any(|&(f, neg)| {
                !neg && meaningful.iter().any(|p| encoder.vocab_token(f).starts_with(p))
            });
            mb.cmp(&ma).then(wb.cmp(&wa)).then(la.cmp(&lb))
        });

        for (rank, &c) in ranked.iter().take(5).enumerate() {
            let rule = tm.clause_rule(c);
            let w = tm.clause_weight(cls, c);
            let pos_lits: Vec<_> = rule.iter().filter(|&&(_, neg)| !neg).collect();
            let neg_count = rule.len() - pos_lits.len();
            let sfx = if neg_count > 0 {
                format!(" (+ {neg_count} NOT-literals)")
            } else {
                String::new()
            };
            println!("  [{}] weight={w}{sfx}", rank + 1);
            for &&(feat, _) in &pos_lits {
                let tok = encoder.vocab_token(feat);
                let note = explain_token(tok);
                if note.is_empty() {
                    println!("       {tok}");
                } else {
                    println!("       {tok:<55}  ← {note}");
                }
            }
        }
        println!();
    }
}
