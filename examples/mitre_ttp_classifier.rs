//! MITRE ATT&CK TTP Classifier — technique-level prediction from Mordor Sysmon events
//!
//! The OTRF/Security-Datasets (Mordor) scenarios use descriptive names
//! (`empire_mimikatz_logonpasswords`, `empire_uac_shellapi_fodhelper`, …) with
//! no T-IDs embedded in filenames or JSON events.  This example:
//!
//! 1. **Fetches technique names + detection strategies** from the official MITRE
//!    ATT&CK STIX bundle at `github.com/mitre/cti` — caches both maps.
//! 2. **Downloads Mordor host datasets** for 8 ATT&CK tactics from
//!    `OTRF/Security-Datasets` (reuses `mordor_data/` from prior runs).
//! 3. **Maps each scenario** to a MITRE ATT&CK technique via keyword rules applied
//!    to the scenario name (e.g. `dcsync → T1003`, `schtask → T1053`).
//! 4. **Augments training with detection-strategy synthetic samples**: each technique's
//!    `x_mitre_detection` text is parsed into ECS feature tokens (Sysmon event IDs,
//!    process names, Windows API calls) and added as synthetic labeled events so the
//!    TM learns MITRE-endorsed detection signals alongside real Mordor observations.
//! 5. **Trains a CoalescedTsetlinMachine** with balanced mini-batches on the combined
//!    dataset; reports per-technique test accuracy with MITRE names and clause rules.
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
const MITRE_DETECT_CACHE: &str = "mordor_data/mitre_detection.json";
const MINI_BATCH_SIZE: usize = 4096;
/// Minimum Sysmon events a technique group must have to become a classifier class.
const MIN_EVENTS: usize = 200;
/// Synthetic samples to add per technique from MITRE detection guidance (training only).
const SYNTHETIC_PER_CLASS: usize = 200;

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

// ── field skip list ───────────────────────────────────────────────────────────
const SKIP_FIELDS: &[&str] = &[
    "ProcessGuid", "ParentProcessGuid", "SourceProcessGUID", "TargetProcessGUID",
    "LogonGuid", "ProviderGuid",
    "ProcessId", "ParentProcessId", "SourceProcessId", "TargetProcessId",
    "SourceThreadId", "ThreadID", "ExecutionProcessID", "LogonId", "TerminalSessionId",
    "SourcePort", "SourcePortName", "port",
    "@timestamp", "UtcTime", "TimeCreated", "SystemTime", "EventTime",
    "EventReceivedTime", "CreationUtcTime", "ParentCreationUtcTime",
    "@version",
    "CallTrace", "Hashes", "Details",
    "SourceModuleName", "SourceModuleType",
    "Channel", "Computer", "Hostname", "host",
    "Keywords", "Level", "Message", "Opcode", "Path",
    "RecordID", "EventRecordID", "RecordNumber",
    "SourceName", "Task", "Version", "ProcessID",
    "AccountName", "AccountType",
    "EventID",
];

const REGISTRY_HIVES: &[&str] = &[
    "HKLM\\", "HKCU\\", "HKU\\", "HKCR\\", "HKCC\\",
    "HKEY_LOCAL_MACHINE\\", "HKEY_CURRENT_USER\\",
    "HKEY_USERS\\", "HKEY_CLASSES_ROOT\\",
];

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
        if SKIP_FIELDS.contains(&key.as_str()) { continue; }
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

// ── MITRE ATT&CK technique catalog + detection guidance ──────────────────────
//
// Downloads the enterprise-attack STIX bundle from github.com/mitre/cti and
// caches both a T-ID → name map and a T-ID → x_mitre_detection map.
// The detection text is later parsed into synthetic training samples.

fn fetch_mitre_attack() -> (HashMap<String, String>, HashMap<String, String>) {
    let names_ok = Path::new(MITRE_CACHE).exists();
    let detects_ok = Path::new(MITRE_DETECT_CACHE).exists();

    if names_ok && detects_ok {
        let names = fs::read_to_string(MITRE_CACHE)
            .ok().and_then(|s| serde_json::from_str::<HashMap<String, String>>(&s).ok())
            .unwrap_or_default();
        let detects = fs::read_to_string(MITRE_DETECT_CACHE)
            .ok().and_then(|s| serde_json::from_str::<HashMap<String, String>>(&s).ok())
            .unwrap_or_default();
        if !names.is_empty() && !detects.is_empty() {
            println!("  {} technique names, {} detection guides loaded from cache",
                     names.len(), detects.len());
            return (names, detects);
        }
    }

    println!("  Fetching MITRE ATT&CK Enterprise bundle from github.com/mitre/cti …");
    let url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json";
    let output = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "180", "-H", "User-Agent: tmu-ttp-classifier/1.0", url])
        .output()
        .expect("curl not available");

    if !output.status.success() || output.stdout.is_empty() {
        eprintln!("  WARN: Failed to fetch MITRE ATT&CK data");
        return (HashMap::new(), HashMap::new());
    }

    let text = match std::str::from_utf8(&output.stdout) {
        Ok(t) => t,
        Err(_) => { eprintln!("  WARN: MITRE response not valid UTF-8"); return (HashMap::new(), HashMap::new()); }
    };

    let bundle: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => { eprintln!("  WARN: Parse error: {e}"); return (HashMap::new(), HashMap::new()); }
    };

    let mut names: HashMap<String, String> = HashMap::new();
    let mut detects: HashMap<String, String> = HashMap::new();

    if let Some(objects) = bundle["objects"].as_array() {
        // Pass 1: collect technique names from non-revoked/deprecated entries only
        for obj in objects {
            if obj["type"].as_str() != Some("attack-pattern") { continue; }
            if obj["revoked"].as_bool() == Some(true)
                || obj["x_mitre_deprecated"].as_bool() == Some(true) { continue; }
            let name = obj["name"].as_str().unwrap_or("").to_string();
            if let Some(refs) = obj["external_references"].as_array() {
                for r in refs {
                    if r["source_name"].as_str() == Some("mitre-attack") {
                        if let Some(id) = r["external_id"].as_str() {
                            if id.starts_with('T') && id.len() >= 5 {
                                names.insert(id.to_string(), name.clone());
                                if let Some(dot) = id.find('.') {
                                    names.entry(id[..dot].to_string()).or_insert_with(|| name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pass 2: aggregate detection guidance text from ALL entries (including revoked).
        // Some technique families (e.g. T1562) are entirely revoked in STIX versioning
        // but still represent valid ATT&CK techniques with useful description text.
        // Fall back to `description` when `x_mitre_detection` is absent (ATT&CK v14+).
        for obj in objects {
            if obj["type"].as_str() != Some("attack-pattern") { continue; }
            let text = obj["x_mitre_detection"].as_str()
                .filter(|s| !s.is_empty())
                .or_else(|| obj["description"].as_str().filter(|s| !s.is_empty()))
                .unwrap_or("")
                .to_string();
            if text.is_empty() { continue; }
            if let Some(refs) = obj["external_references"].as_array() {
                for r in refs {
                    if r["source_name"].as_str() == Some("mitre-attack") {
                        if let Some(id) = r["external_id"].as_str() {
                            if id.starts_with('T') && id.len() >= 5 {
                                // Append — sub-techniques accumulate into the same key
                                let e = detects.entry(id.to_string()).or_insert_with(String::new);
                                if !e.is_empty() { e.push(' '); }
                                e.push_str(&text);
                                // Also contribute to parent
                                if let Some(dot) = id.find('.') {
                                    let parent = id[..dot].to_string();
                                    let pe = detects.entry(parent).or_insert_with(String::new);
                                    if !pe.is_empty() { pe.push(' '); }
                                    pe.push_str(&text);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if names.is_empty() {
        eprintln!("  WARN: No techniques extracted");
        return (HashMap::new(), HashMap::new());
    }

    println!("  → {} names, {} detection guides loaded from ATT&CK", names.len(), detects.len());

    if let Ok(json) = serde_json::to_string_pretty(&names) {
        let _ = fs::write(MITRE_CACHE, &json);
    }
    if let Ok(json) = serde_json::to_string_pretty(&detects) {
        let _ = fs::write(MITRE_DETECT_CACHE, &json);
    }
    (names, detects)
}

/// Fallback technique name lookup — handles parent techniques that may be absent from
/// the STIX cache when sub-techniques are non-revoked but the parent entry is.
fn technique_name<'a>(techniques: &'a HashMap<String, String>, tid: &str) -> &'a str {
    if let Some(n) = techniques.get(tid) { return n.as_str(); }
    match tid {
        "T1003" => "OS Credential Dumping",
        "T1021" => "Remote Services",
        "T1047" => "Windows Management Instrumentation",
        "T1053" => "Scheduled Task/Job",
        "T1055" => "Process Injection",
        "T1059" => "Command and Scripting Interpreter",
        "T1087" => "Account Discovery",
        "T1123" => "Audio Capture",
        "T1136" => "Create Account",
        "T1190" => "Exploit Public-Facing Application",
        "T1197" => "BITS Jobs",
        "T1218" => "System Binary Proxy Execution",
        "T1543" => "Create or Modify System Process",
        "T1546" => "Event Triggered Execution",
        "T1547" => "Boot or Logon Autostart Execution",
        "T1548" => "Abuse Elevation Control Mechanism",
        "T1555" => "Credentials from Password Stores",
        "T1558" => "Steal or Forge Kerberos Tickets",
        "T1562" => "Impair Defenses",
        "T1574" => "Hijack Execution Flow",
        _ => "(unknown)",
    }
}

// ── MITRE detection guidance → ECS feature tokens ────────────────────────────
//
// Parses the free-text `x_mitre_detection` field from each attack-pattern and
// extracts ECS-style feature tokens (Sysmon event IDs, process/tool names,
// Windows API calls, registry indicators).  These tokens form synthetic
// labeled training samples that encode WHAT MITRE SAYS DEFENDERS SHOULD WATCH.

fn detection_to_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();

    // Sysmon / Windows event IDs mentioned in detection guidance
    let event_patterns: &[(u32, &[&str])] = &[
        (1,  &["event id 1", "sysmon event 1", "process creat", "newly executed process", "new process spawned"]),
        (3,  &["event id 3", "network connect", "outbound network", "network traffic"]),
        (7,  &["event id 7", "image load", "dll load", "module load"]),
        (8,  &["event id 8", "createremotethread", "remote thread"]),
        (9,  &["event id 9", "rawaccess", "raw access"]),
        (10, &["event id 10", "process access", "process injection", "lsass"]),
        (11, &["event id 11", "file creat", "file creation", "file written"]),
        (12, &["event id 12", "registry object created"]),
        (13, &["event id 13", "registry value set"]),
        (17, &["event id 17", "named pipe created"]),
        (22, &["event id 22", "dns query", "dns lookup", "dns request"]),
        (25, &["event id 25", "process tamper"]),
    ];
    for &(eid, kws) in event_patterns {
        if kws.iter().any(|k| lower.contains(k)) {
            tokens.push(format!("event.id::{eid}"));
        }
    }

    // Well-known attack tools and system process targets
    let proc_patterns: &[(&str, &str)] = &[
        ("mimikatz",       "process.name::mimikatz.exe"),
        ("rubeus",         "process.name::rubeus.exe"),
        ("procdump",       "process.name::procdump.exe"),
        ("lsass",          "target.process.name::lsass.exe"),
        ("powershell",     "process.name::powershell.exe"),
        ("cmd.exe",        "process.name::cmd.exe"),
        ("wscript",        "process.name::wscript.exe"),
        ("cscript",        "process.name::cscript.exe"),
        ("mshta",          "process.name::mshta.exe"),
        ("wmic",           "process.name::wmic.exe"),
        ("schtasks",       "process.name::schtasks.exe"),
        ("bitsadmin",      "process.name::bitsadmin.exe"),
        ("certutil",       "process.name::certutil.exe"),
        ("regsvr32",       "process.name::regsvr32.exe"),
        ("rundll32",       "process.name::rundll32.exe"),
        ("msiexec",        "process.name::msiexec.exe"),
        ("wuauclt",        "process.name::wuauclt.exe"),
        ("psexec",         "process.name::psexec.exe"),
        ("net.exe",        "process.name::net.exe"),
        ("nltest",         "process.name::nltest.exe"),
        ("at.exe",         "process.name::at.exe"),
        ("winlogon",       "target.process.name::winlogon.exe"),
        ("services.exe",   "target.process.name::services.exe"),
        ("installutil",    "process.name::installutil.exe"),
        ("cmstp",          "process.name::cmstp.exe"),
    ];
    for &(kw, feat) in proc_patterns {
        if lower.contains(kw) {
            tokens.push(feat.to_string());
        }
    }

    // Registry persistence / evasion indicators
    if lower.contains("run key") || lower.contains("userinit") || lower.contains("autostart") {
        tokens.push("registry.path::HKLM".to_string());
    }
    if lower.contains("security account") || (lower.contains("sam") && lower.contains("registry")) {
        tokens.push("registry.path::HKLM".to_string());
    }

    // Windows API calls that MITRE explicitly mentions in detection guidance
    let api_patterns: &[(&str, &str)] = &[
        ("createremotethread",  "api.call::CreateRemoteThread"),
        ("writeprocessmemory",  "api.call::WriteProcessMemory"),
        ("virtualalloc",        "api.call::VirtualAllocEx"),
        ("openprocess",         "api.call::OpenProcess"),
        ("duplicatehandle",     "api.call::DuplicateHandle"),
        ("minidumpwritedump",   "api.call::MiniDumpWriteDump"),
        ("setwindowshookex",    "api.call::SetWindowsHookEx"),
        ("ntcreatesection",     "api.call::NtCreateSection"),
    ];
    for &(kw, feat) in api_patterns {
        if lower.contains(kw) {
            tokens.push(feat.to_string());
        }
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

// ── scenario name → ATT&CK technique ID ──────────────────────────────────────
//
// OTRF/Security-Datasets uses descriptive scenario names (no T-IDs embedded).
// Keyword rules map scenario names to ATT&CK base technique IDs.
// Priority: more-specific keywords are checked before broader ones.

fn scenario_to_technique(s: &str) -> Option<&'static str> {
    // ── T1003: OS Credential Dumping ─────────────────────────────────────────
    if s.contains("dcsync")             { return Some("T1003"); }
    if s.contains("ntds")               { return Some("T1003"); }
    if s.contains("lsass")              { return Some("T1003"); }
    if s.contains("lsadump")            { return Some("T1003"); }
    if s.contains("logonpasswords")     { return Some("T1003"); }
    if s.contains("sam_access")
        || s.contains("sam_copy")
        || s.contains("reg_dump_sam")
        || s.contains("powerdump")      { return Some("T1003"); }
    if s.contains("ninjacopy")          { return Some("T1003"); }
    if s.contains("over_pth")           { return Some("T1003"); }
    if s.contains("extract_keys")       { return Some("T1003"); }
    if s.contains("backupkeys")         { return Some("T1003"); }
    if s.contains("dumpert")            { return Some("T1003"); }
    if s.contains("lsa_secrets_dump")   { return Some("T1003"); }
    if s.contains("mimikatz")           { return Some("T1003"); } // remaining mimikatz

    // ── T1562: Impair Defenses ────────────────────────────────────────────────
    if s.contains("disable_eventlog")
        || s.contains("stop_eventlog")
        || s.contains("stop_event_logging")
        || s.contains("stop_netprofm")
        || s.contains("commandline_logging_disabled")
        || s.contains("modify_security_eventlog")
        || s.contains("wevtutil")
        || s.contains("minint_key")
        || s.contains("netsh_fw")
        || s.contains("auditpol")       { return Some("T1562"); }

    // ── T1055: Process Injection ──────────────────────────────────────────────
    if s.contains("herpaderping")
        || s.contains("dllinjection")
        || s.contains("psinject")
        || s.contains("mavinject")
        || s.contains("pe_injection")
        || s.contains("createremotethread") { return Some("T1055"); }

    // ── T1546: Event Triggered Execution (WMI event subscriptions) ───────────
    if s.contains("wmi_event_subscription")
        || s.contains("wmi_local_event")
        || s.contains("activescripteventconsumer") { return Some("T1546"); }

    // ── T1021: Remote Services (lateral movement) ─────────────────────────────
    if s.contains("psexec")
        || s.contains("smbexec")
        || s.contains("psremoting")
        || s.contains("dcom_shellwindows")
        || s.contains("msbuild_dcerpc_wmi_smb")
        || s.contains("com_wsman_automation")
        || s.contains("dcom_registerxll")
        || s.contains("dcom_executeexcel4macro")
        || s.contains("dcom_iertutil")      { return Some("T1021"); }

    // ── T1047: Windows Management Instrumentation ─────────────────────────────
    if s.contains("sharpwmi")
        || s.contains("wmi_dcerpc_wmi")
        || s.contains("wmi_iwbemservices")
        || s.contains("wmic_remote")        { return Some("T1047"); }

    // ── T1218: System Binary Proxy Execution (LOLBins) ───────────────────────
    if s.contains("mshta")
        || s.contains("regsvr32")
        || s.contains("installutil")
        || s.contains("cmstp")
        || s.contains("hh_local")
        || s.contains("control_panel")
        || s.contains("wuauclt")
        || s.contains("register_cimprovider") { return Some("T1218"); }

    // ── T1053: Scheduled Task / Job ───────────────────────────────────────────
    if s.contains("schtask")
        || s.contains("schtasks")          { return Some("T1053"); }

    // ── T1087: Account Discovery ──────────────────────────────────────────────
    if s.contains("sharpview")
        || s.contains("seatbelt")
        || s.contains("getdomaingroup")
        || s.contains("samr")
        || s.contains("enumdomain")
        || s.contains("net_local")
        || s.contains("powerview")
        || s.contains("localadmin")
        || s.contains("net_users")
        || s.contains("net_localgroup")
        || s.contains("getsession")
        || s.contains("sc_query")
        || s.contains("sharpsc")           { return Some("T1087"); }

    // ── T1547: Boot/Logon Autostart Execution ─────────────────────────────────
    if s.contains("run_keys")
        || s.contains("userinitmprlogonscript") { return Some("T1547"); }

    // ── T1558: Steal or Forge Kerberos Tickets ────────────────────────────────
    if s.contains("rubeus")
        || s.contains("monologue")
        || s.contains("asktgt")            { return Some("T1558"); }

    // ── T1574: Hijack Execution Flow (DLL hijacking) ──────────────────────────
    if s.contains("dll_hijack")
        || s.contains("wbemcomn_dll")      { return Some("T1574"); }

    // ── T1059: Command and Scripting Interpreter ──────────────────────────────
    if s.contains("launcher_vbs")
        || s.contains("python_webserver")
        || s.contains("powershell_httplistener")
        || s.contains("xsl_jscript")
        || s.contains("launcher_sct")      { return Some("T1059"); }

    // ── T1197: BITS Jobs ──────────────────────────────────────────────────────
    if s.contains("bitsadmin")             { return Some("T1197"); }

    // ── T1056: Input Capture ─────────────────────────────────────────────────
    if s.contains("input_capture")
        || s.contains("promptforcreds")    { return Some("T1056"); }

    // ── T1555: Credentials from Password Stores ───────────────────────────────
    if s.contains("vault_web")
        || s.contains("windows_vault")
        || s.contains("export_adfsdatabaseconfig") { return Some("T1555"); }

    // ── T1190: Exploit Public-Facing Application ──────────────────────────────
    if s.contains("proxylogon")            { return Some("T1190"); }

    // ── T1123: Audio Capture ──────────────────────────────────────────────────
    if s.contains("record_mic")            { return Some("T1123"); }

    // ── T1136: Create Account ─────────────────────────────────────────────────
    if s.contains("wmic_add_user")         { return Some("T1136"); }

    // ── T1543: Create or Modify System Process (service modification) ─────────
    if s.contains("service_mod")           { return Some("T1543"); }

    // ── T1548: Abuse Elevation Control Mechanism (UAC bypass) ────────────────
    if s.contains("uac") || s.contains("bypassuac") { return Some("T1548"); }

    None // no mapping → event is unlabeled
}

// ── strip timestamp suffix from extracted JSON filename → scenario name ───────
//
// "empire_mimikatz_logonpasswords_2020-08-07103224.json" → "empire_mimikatz_logonpasswords"
// "cmd_dumping_ntds_dit_file_ntdsutil.json"              → "cmd_dumping_ntds_dit_file_ntdsutil"

fn json_to_scenario(json_name: &str) -> String {
    let s = json_name.trim_end_matches(".json");
    // Strip trailing _<10+ digits> timestamp suffix.
    let bytes = s.as_bytes();
    for i in (1..bytes.len()).rev() {
        if bytes[i - 1] == b'_' {
            let suffix = &s[i..];
            if suffix.len() >= 10 && suffix.chars().all(|c| c.is_ascii_digit()) {
                return s[..i - 1].to_string();
            }
        }
    }
    s.to_string()
}

// ── Mordor dataset discovery + download ───────────────────────────────────────

fn discover_and_download() {
    fs::create_dir_all(DATA_DIR).expect("could not create mordor_data/");
    for tactic in TACTIC_DIRS {
        let api_url = format!(
            "https://api.github.com/repos/OTRF/Security-Datasets/contents/datasets/atomic/windows/{tactic}/host"
        );
        let out = std::process::Command::new("curl")
            .args(["-sL", "--max-time", "30", "-H", "User-Agent: tmu-ttp-classifier/1.0", &api_url])
            .output()
            .expect("curl not available");
        let Ok(text) = std::str::from_utf8(&out.stdout) else { continue; };
        let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(text) else {
            eprintln!("  WARN: GitHub API error for {tactic} (rate-limited or no network)");
            continue;
        };
        for entry in &entries {
            let name = entry["name"].as_str().unwrap_or("");
            if !name.ends_with(".zip") { continue; }
            let stem = name.trim_end_matches(".zip");
            let url = entry["download_url"].as_str().unwrap_or("");
            if url.is_empty() { continue; }
            let zip_path = format!("{DATA_DIR}/{stem}.zip");
            if Path::new(&zip_path).exists() { continue; }
            print!("  [{tactic}] {stem}… ");
            let _ = std::io::stdout().flush();
            let st = std::process::Command::new("curl")
                .args(["-sL", "--max-time", "120", "-o", &zip_path, url])
                .status().expect("curl not available");
            if !st.success() { eprintln!("WARN: download failed"); continue; }
            let st = std::process::Command::new("unzip")
                .args(["-o", &zip_path, "-d", DATA_DIR])
                .status().expect("unzip not available");
            println!("{}", if st.success() { "ok" } else { "WARN: unzip failed" });
        }
    }
}

// ── collect JSON files, label each by ATT&CK technique ───────────────────────

fn collect_files(dir: &Path, out: &mut Vec<(std::path::PathBuf, &'static str)>) {
    let Ok(entries) = fs::read_dir(dir) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("__MACOSX") {
                collect_files(&path, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("._")
                || name.ends_with("_techniques.json")
                || name == "tactic_map.json"
            {
                continue;
            }
            let scenario = json_to_scenario(name);
            if let Some(tid) = scenario_to_technique(&scenario) {
                out.push((path, tid));
            }
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
    if tok.starts_with("target.process.name::") { return "victim process"; }
    if tok.starts_with("source.process.name::") { return "injecting/accessing process"; }
    if tok.starts_with("target.process.granted_access::") { return "access rights"; }
    if tok.contains("file.path::Temp")     { return "staging in Temp (common dropper)"; }
    if tok.contains("file.path::System32") { return "system directory"; }
    if tok.contains("file.path::AppData")  { return "user staging area"; }
    if tok.starts_with("file.path::")      { return "file location category"; }
    if tok.starts_with("file.name::")      { return "file name"; }
    if tok.starts_with("dll.name::")       { return "DLL loaded"; }
    if tok.starts_with("destination.")     { return "outbound C2 / lateral target"; }
    if tok.starts_with("dns.question::")   { return "DNS lookup (C2 beacon / lateral)"; }
    if tok.starts_with("registry.")        { return "registry operation"; }
    if tok.contains("::lsass")             { return "LSASS — credential store"; }
    if tok.contains("::mimikatz")          { return "Mimikatz credential dumper"; }
    if tok.contains("::powershell")        { return "PowerShell execution"; }
    if tok.contains("::cmd.exe")           { return "Command prompt"; }
    if tok.contains("::rundll32")          { return "LOLBin: DLL runner"; }
    if tok.contains("::regsvr32")          { return "LOLBin: COM/SCT payload"; }
    if tok.contains("::mshta")             { return "LOLBin: HTML/VBS/JS runner"; }
    if tok.contains("::wmic")              { return "WMI execution"; }
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

    // ── Step 1: load MITRE ATT&CK technique catalog + detection guidance ─────────
    println!("Step 1: MITRE ATT&CK Enterprise technique catalog + detection strategies");
    let (techniques, detections) = fetch_mitre_attack();

    // ── Step 2: download Mordor datasets ─────────────────────────────────────
    println!("\nStep 2: Mordor dataset download / verification");
    discover_and_download();

    // ── Step 3: collect labeled files ─────────────────────────────────────────
    println!("\nStep 3: Labeling scenarios with ATT&CK technique IDs…");
    let mut raw_files: Vec<(std::path::PathBuf, &'static str)> = Vec::new();
    collect_files(Path::new(DATA_DIR), &mut raw_files);
    raw_files.sort_by_key(|(p, _)| p.clone());

    if raw_files.is_empty() {
        eprintln!("ERROR: No labeled JSON files found in {DATA_DIR}/");
        std::process::exit(1);
    }

    // ── Step 4: parse events, group by technique ──────────────────────────────
    println!("\nStep 4: Parsing Sysmon events…");
    let mut technique_events: HashMap<&'static str, Vec<Vec<String>>> = HashMap::new();

    for (path, tid) in &raw_files {
        let Ok(text) = fs::read_to_string(path) else { continue; };
        let mut count = 0usize;
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue; };
            if !v["Channel"].as_str().unwrap_or("").starts_with("Microsoft-Windows-Sysmon") {
                continue;
            }
            let eid = v["EventID"].as_u64().unwrap_or(0) as u32;
            technique_events.entry(tid).or_default().push(event_to_tokens(&v, eid));
            count += 1;
        }
        if count > 0 {
            let tname = technique_name(&techniques, tid);
            let scenario = json_to_scenario(
                path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            );
            println!("  {tid}  {tname:<52}  {count:>5} ev  {scenario}");
        }
    }

    // ── Step 5: select qualifying technique classes ────────────────────────────
    let mut all_tids: Vec<(&'static str, usize)> = technique_events
        .iter()
        .map(|(tid, evs)| (*tid, evs.len()))
        .collect();
    all_tids.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    println!("\nAll techniques found (sorted by event count):");
    println!("  {:<8}  {:<52}  {:>8}  {}", "TID", "Name", "events", "status");
    for (tid, count) in &all_tids {
        let tname = technique_name(&techniques, tid);
        let status = if *count >= MIN_EVENTS { "✓ included" } else { "✗ skipped (too few events)" };
        println!("  {tid:<8}  {tname:<52}  {:>8}  {status}", count);
    }

    let mut class_labels: Vec<&'static str> = Vec::new();
    let mut technique_to_class: HashMap<&'static str, usize> = HashMap::new();
    for (tid, count) in &all_tids {
        if *count >= MIN_EVENTS {
            technique_to_class.insert(tid, class_labels.len());
            class_labels.push(tid);
        }
    }

    if class_labels.is_empty() {
        eprintln!(
            "\nERROR: No techniques with >= {MIN_EVENTS} events. \
             Lower MIN_EVENTS or download more data."
        );
        std::process::exit(1);
    }

    let n_classes = class_labels.len();
    println!("\n{n_classes} technique classes selected:");
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        println!("  class {:>2}  {tid}  {tname:<52}  {} events", cls, technique_events[tid].len());
    }

    // ── Step 6: build labeled dataset from real Mordor events ─────────────────
    let mut all_real: Vec<(Vec<String>, usize)> = Vec::new();
    for (cls, tid) in class_labels.iter().enumerate() {
        if let Some(evs) = technique_events.get(tid) {
            for tokens in evs {
                all_real.push((tokens.clone(), cls));
            }
        }
    }
    println!("\nTotal real labeled events: {}", all_real.len());

    // ── Step 7: shuffle + 80/20 split on REAL events only ────────────────────
    let mut rng = Rng::new(42);
    for i in (1..all_real.len()).rev() {
        let j = rng.below(i + 1);
        all_real.swap(i, j);
    }
    let cut = all_real.len() * 4 / 5;
    let (train_real, test_slice) = all_real.split_at(cut);

    // ── Step 7b: add MITRE detection-strategy synthetic samples (train only) ──
    // Each technique's x_mitre_detection text is parsed into ECS feature tokens
    // and added as labeled synthetic events so the TM learns MITRE-endorsed
    // detection signals in addition to raw Mordor observations.
    println!("\nStep 7b: MITRE detection strategies → synthetic training samples");
    let mut all_train: Vec<(Vec<String>, usize)> = train_real.to_vec();
    let mut n_syn_total = 0usize;
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        if let Some(text) = detections.get(*tid) {
            let tokens = detection_to_tokens(text);
            if tokens.is_empty() {
                println!("  {tid}  {tname:<52}  (no detection tokens extracted)");
                continue;
            }
            for _ in 0..SYNTHETIC_PER_CLASS {
                all_train.push((tokens.clone(), cls));
            }
            n_syn_total += SYNTHETIC_PER_CLASS;
            println!("  {tid}  {tname:<52}  {} tokens → {} synthetic samples",
                     tokens.len(), SYNTHETIC_PER_CLASS);
        } else {
            println!("  {tid}  {tname:<52}  (no detection guidance in ATT&CK bundle)");
        }
    }
    println!("  total synthetic: {n_syn_total}  |  combined training: {}", all_train.len());

    // Shuffle training so synthetic and real samples are interleaved
    for i in (1..all_train.len()).rev() {
        let j = rng.below(i + 1);
        all_train.swap(i, j);
    }

    let (train_tokens, train_y): (Vec<_>, Vec<_>) =
        all_train.iter().map(|(t, y)| (t, *y)).unzip();
    let (test_tokens, test_y): (Vec<_>, Vec<_>) =
        test_slice.iter().map(|(t, y)| (t, *y)).unzip();

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

    // Encoder is fit on training (including synthetic) so detection tokens enter vocabulary
    let encoder = Encoder::fit_categorical(&tr_refs);
    let test_x = encoder.encode_batch_categorical(&te_refs);

    println!(
        "\ntrain={} ({}+{} synthetic) test={} | vocabulary: {} ECS tokens\n",
        tr_refs.len(), train_real.len(), n_syn_total, te_refs.len(), encoder.n_features()
    );

    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &y) in train_y.iter().enumerate() {
        class_indices[y].push(i);
    }
    println!("class training distribution (real + synthetic):");
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        println!("  class {:>2}  {tid}  {tname:<52}  train={}", cls, class_indices[cls].len());
    }
    println!();

    // ── Step 8: train CoalescedTsetlinMachine ─────────────────────────────────
    let mut tm = CoalescedTsetlinMachine::with_config(
        n_classes,
        encoder.n_features(),
        256,   // shared clause bank
        50,    // threshold T
        5.0,   // specificity s
        8,     // TA state bits
        true,  // boost true positives
        42,
    );
    let mut shuffle_rng = Rng::new(0xDEAD_BEEF);
    let per_class = (MINI_BATCH_SIZE / n_classes).max(1);
    let n_batches = tr_inner.len().div_ceil(MINI_BATCH_SIZE);

    for epoch in 1..=10 {
        let t0 = std::time::Instant::now();
        for ci in 0..n_classes {
            let len = class_indices[ci].len();
            if len < 2 { continue; }
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
                if ci_len == 0 { continue; }
                for k in 0..per_class {
                    batch.push(class_indices[ci][(b * per_class + k) % ci_len]);
                }
            }
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
                println!("  epoch {epoch:>2}  batch {:>3}/{n_batches}  {evps:>7.0} ev/s", b + 1);
                let _ = std::io::stdout().flush();
            }
        }
        let te_acc = tm.accuracy(&test_x, &test_y) * 100.0;
        println!("epoch {epoch:>2}  test={te_acc:.2}%  ({:.1}s)\n", t0.elapsed().as_secs_f32());
    }

    // ── Step 9: per-technique accuracy ────────────────────────────────────────
    println!("\n--- per-technique test accuracy ---");
    let infer_t0 = std::time::Instant::now();
    let _ = tm.accuracy(&test_x, &test_y);
    let infer_us = infer_t0.elapsed().as_secs_f64() * 1e6 / test_y.len() as f64;

    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        let indices: Vec<usize> = test_y.iter().enumerate()
            .filter(|(_, &y)| y == cls).map(|(i, _)| i).collect();
        if indices.is_empty() { continue; }
        let correct = indices.iter().filter(|&&i| {
            let s = encoder.encode_one_categorical(&te_inner[i]);
            tm.predict(&s) == cls
        }).count();
        println!(
            "  class {:>2}  {tid}  {tname:<52}  {correct}/{} ({:.1}%)",
            cls, indices.len(), correct as f64 / indices.len() as f64 * 100.0
        );
    }
    println!("\nbulk inference: {infer_us:.2}µs/event  ({:.0} ev/s)", 1e6 / infer_us);

    // ── Step 10: clause statistics ─────────────────────────────────────────────
    let meaningful = [
        "process.name", "process.args", "process.parent",
        "dll.name", "file.name", "file.path",
        "target.process", "source.process",
        "dns.question", "destination.", "network.", "registry.",
    ];

    println!("\n━━━ clause statistics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("{:<8}  {:<52}  {:>10}  {:>10}  {:>9}  {:>6}", "TID", "name", "pos_clauses", "neg_clauses", "mean_w", "max_w");
    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        let pos: Vec<i32> = (0..tm.n_clauses())
            .filter_map(|c| { let w = tm.clause_weight(cls, c); if w > 0 { Some(w) } else { None } })
            .collect();
        let neg_cnt = (0..tm.n_clauses()).filter(|&c| tm.clause_weight(cls, c) < 0).count();
        let mean_w = if pos.is_empty() { 0.0 } else { pos.iter().sum::<i32>() as f64 / pos.len() as f64 };
        let max_w = pos.iter().copied().max().unwrap_or(0);
        println!("{tid:<8}  {tname:<52}  {:>10}  {:>10}  {:>9.1}  {:>6}", pos.len(), neg_cnt, mean_w, max_w);
    }

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

    println!("\ntop-20 positive literal tokens across all classes:");
    for &(feat, count) in freq_vec.iter().take(20) {
        let n_cls = token_class_mask[&feat].count_ones();
        println!("  {:>5} cls={:>2}  {}", count, n_cls, encoder.vocab_token(feat));
    }

    // ── Step 11: top-5 rules per technique ────────────────────────────────────
    println!("\n━━━ top rules per ATT&CK technique ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (cls, tid) in class_labels.iter().enumerate() {
        let tname = technique_name(&techniques, tid);
        let n_test = test_y.iter().filter(|&&y| y == cls).count();
        if n_test == 0 { continue; }
        println!("── {tid}  {tname}  (class {cls}) ──");

        let mut ranked: Vec<usize> = Vec::new();
        for &max_lits in &[30usize, 60, 120, 300, usize::MAX] {
            ranked = (0..tm.n_clauses())
                .filter(|&c| tm.clause_weight(cls, c) > 0)
                .filter(|&c| {
                    let r = tm.clause_rule(c);
                    r.iter().any(|&(_, neg)| !neg) && r.len() <= max_lits
                })
                .collect();
            if ranked.len() >= 5 { break; }
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
            let sfx = if neg_count > 0 { format!(" (+ {neg_count} NOT)") } else { String::new() };
            println!("  [{}] w={w}{sfx}", rank + 1);
            for &&(feat, _) in &pos_lits {
                let tok = encoder.vocab_token(feat);
                let note = explain_token(tok);
                if note.is_empty() { println!("       {tok}"); }
                else { println!("       {tok:<55}  ← {note}"); }
            }
        }
        println!();
    }
}
