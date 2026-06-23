//! Mordor full-dataset demo — train the TM on **all Sysmon event types** from real
//! OTRF Security-Datasets traces (one Sysmon event = one TM sample, no windowing).
//!
//! ## What this does
//!
//! 1. Downloads three Mordor execution traces from GitHub into `mordor_data/`
//!    (skipped if already present).
//! 2. Parses every Sysmon event (EventIDs 1,3,5,7,9,10,11,12,13,17,18,22,23)
//!    into `col::val` tokens. Each event type emits its own fields as columns,
//!    e.g. `EventID::10`, `SourceImage::powershell.exe`, `TargetImage::lsass.exe`,
//!    `GrantedAccess::0x1010`.
//! 3. Labels all real-trace events as **attack (1)** (dataset-level ground truth).
//! 4. Generates an equal number of synthetic **benign (0)** events across the same
//!    event types so classes are balanced.
//! 5. Shuffles, 80/20 train/test split, trains a TM, reports accuracy every 5 epochs.
//! 6. Prints the full vocabulary so you can see which `col::val` features the TM
//!    learned.
//!
//! ## Run
//! ```text
//! cargo run --release --example sysmon_mordor
//! ```
//!
//! ## Note on labeling
//!
//! Mordor datasets are labeled at the **file** level (the whole trace is one attack
//! scenario); individual events are not labeled. We therefore label all events from
//! the attack traces as class 1 and synthetic benign events as class 0. This means
//! background OS events in the attack traces are also labeled 1 — the TM has to learn
//! what correlates with attack traces, not a hand-curated per-event ground truth.

#[path = "sysmon_shared.rs"]
mod shared;
use shared::{basename, benign_tokens, hive_of, parent_dir};

use std::{fs, path::Path};
use tmu_rs::{Encoder, Rng, TsetlinMachine};

// ── dataset URLs (OTRF Security-Datasets, execution/host) ──────────────────────

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
        if status.success() {
            println!("ok");
        } else {
            eprintln!("WARN: unzip failed for {name}");
        }
    }
}

// ── per-event tokenizer ────────────────────────────────────────────────────────

/// Convert a single Sysmon event to a list of `col::val` tokens.
///
/// Every event gets `EventID::N`. Additional fields are event-type-specific.
/// High-cardinality raw values (full paths, GUIDs, PIDs) are normalized:
/// paths → basename, registry paths → hive only, temp file paths → parent dir.
fn event_to_tokens(v: &serde_json::Value, eid: u32) -> Vec<String> {
    let mut t = vec![format!("EventID::{eid}")];
    let s = |k: &str| v[k].as_str().unwrap_or("").to_string();

    // Push `Col::val` only when val is non-empty. $val must be an owned String.
    macro_rules! push {
        ($col:expr, $val:expr) => {{
            let v: String = $val;
            if !v.is_empty() { t.push(format!("{}::{}", $col, v)); }
        }};
    }

    match eid {
        1 => {
            // ProcessCreate
            push!("Image",          basename(&s("Image")));
            push!("ParentImage",    basename(&s("ParentImage")));
            push!("User",           s("User"));
            push!("IntegrityLevel", s("IntegrityLevel"));
            push!("Company",        s("Company"));
            push!("Signed",         s("Signed"));
        }
        3 => {
            // NetworkConnect
            push!("Image",           basename(&s("Image")));
            push!("DestinationPort", s("DestinationPort"));
            push!("Protocol",        s("Protocol"));
            push!("Initiated",       s("Initiated"));
        }
        5 => {
            // ProcessTerminate
            push!("Image", basename(&s("Image")));
        }
        7 => {
            // ImageLoad
            push!("Image",       basename(&s("Image")));
            push!("ImageLoaded", basename(&s("ImageLoaded")));
            push!("Signed",      s("Signed"));
            push!("Company",     s("Company"));
        }
        9 => {
            // RawAccessRead
            push!("Image",  basename(&s("Image")));
            push!("Device", s("Device"));
        }
        10 => {
            // ProcessAccess
            push!("SourceImage",   basename(&s("SourceImage")));
            push!("TargetImage",   basename(&s("TargetImage")));
            push!("GrantedAccess", s("GrantedAccess"));
        }
        11 => {
            // FileCreate
            push!("Image",          basename(&s("Image")));
            push!("TargetFilename", parent_dir(&s("TargetFilename")));
        }
        12 | 13 => {
            // RegistryEvent (object create/delete / value set)
            push!("Image",        basename(&s("Image")));
            push!("TargetObject", hive_of(&s("TargetObject")));
        }
        17 | 18 => {
            // PipeEvent (created / connected)
            push!("Image",    basename(&s("Image")));
            push!("PipeName", s("PipeName"));
        }
        22 => {
            // DnsQuery
            push!("Image",     basename(&s("Image")));
            push!("QueryName", s("QueryName"));
        }
        23 => {
            // FileDelete
            push!("Image",          basename(&s("Image")));
            push!("TargetFilename", parent_dir(&s("TargetFilename")));
        }
        _ => {} // unknown type — EventID token is enough
    }
    t
}

// ── parse one NDJSON file into token lists ─────────────────────────────────────

fn parse_file(path: &str) -> std::io::Result<Vec<Vec<String>>> {
    let text = fs::read_to_string(path)?;
    let mut events = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if !v["Channel"].as_str().unwrap_or("").starts_with("Microsoft-Windows-Sysmon") {
            continue;
        }
        let eid = v["EventID"].as_u64().unwrap_or(0) as u32;
        events.push(event_to_tokens(&v, eid));
    }
    Ok(events)
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() {
    ensure_datasets();

    // Collect all Sysmon events from downloaded JSON files.
    let mut attack: Vec<Vec<String>> = Vec::new();
    for entry in fs::read_dir(DATA_DIR).expect("mordor_data/ not found") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match parse_file(path.to_str().unwrap()) {
            Ok(mut evs) => {
                println!("  {} → {} Sysmon events", path.file_name().unwrap().to_str().unwrap(), evs.len());
                attack.append(&mut evs);
            }
            Err(e) => eprintln!("  WARN: {e}"),
        }
    }
    println!("\n{} real attack events (label 1)", attack.len());

    // Generate equal-count synthetic benign events (label 0).
    let mut rng = Rng::new(42);
    let benign: Vec<Vec<String>> = (0..attack.len()).map(|_| benign_tokens(&mut rng)).collect();
    println!("{} synthetic benign events (label 0)\n", benign.len());

    // Combine, shuffle, 80/20 split.
    let mut all: Vec<(Vec<String>, usize)> = attack
        .into_iter().map(|t| (t, 1))
        .chain(benign.into_iter().map(|t| (t, 0)))
        .collect();
    for i in (1..all.len()).rev() {
        let j = rng.below(i + 1);
        all.swap(i, j);
    }
    let cut = all.len() * 4 / 5;
    let (train_all, test_all) = all.split_at(cut);

    let (train_tokens, train_y): (Vec<_>, Vec<_>) = train_all.iter().map(|(t, y)| (t, *y)).unzip();
    let (test_tokens, test_y): (Vec<_>, Vec<_>) = test_all.iter().map(|(t, y)| (t, *y)).unzip();

    // Build &[&[&str]] references required by the categorical encoder.
    let tr_inner: Vec<Vec<&str>> = train_tokens.iter().map(|t| t.iter().map(String::as_str).collect()).collect();
    let tr_refs: Vec<&[&str]> = tr_inner.iter().map(|v| v.as_slice()).collect();

    let te_inner: Vec<Vec<&str>> = test_tokens.iter().map(|t| t.iter().map(String::as_str).collect()).collect();
    let te_refs: Vec<&[&str]> = te_inner.iter().map(|v| v.as_slice()).collect();

    // Fit categorical encoder on training set.
    let encoder = Encoder::fit_categorical(&tr_refs);
    println!(
        "train={} test={} | vocabulary: {} features ({} col::val tokens + {} UNK/OOV)\n",
        tr_refs.len(),
        te_refs.len(),
        encoder.n_features(),
        encoder.n_features().saturating_sub(25), // rough: subtract sentinel count
        25,
    );

    let train_x = encoder.encode_batch_categorical(&tr_refs);
    let test_x  = encoder.encode_batch_categorical(&te_refs);

    // Train.
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
    println!("  … plus {} <UNK>/<OOV> sentinels", {
        (0..encoder.n_features())
            .filter(|&b| {
                let t = encoder.vocab_token(b);
                t.contains("<UNK>") || t.contains("<OOV>")
            })
            .count()
    });
}
