//! Sysmon demo — turn a Sysmon **Event ID 1 (ProcessCreate)** log into
//! `column::value` vocabulary tokens and train the TM to flag suspicious events.
//!
//! This reuses the crate's existing **categorical** encoder
//! ([`Encoder::fit_categorical`]): the "sysmon encoder" is just a small parser
//! that turns a raw key=value log record into `"Field::value"` tokens.
//!
//! Pipeline:  raw key=value text  ->  `col::val` tokens  ->  categorical encoder  ->  TM
//!
//! `cargo run --release --example sysmon`
//!
//! [`Encoder::fit_categorical`]: tmu_rs::Encoder::fit_categorical

use tmu_rs::{Encoder, Rng, TsetlinMachine};

// ── 1. parse a raw Event ID 1 record into `col::val` tokens ────────────────────

/// Fields we tokenize. Low-cardinality, categorical fields make good features;
/// high-cardinality fields like `CommandLine` and `Hashes` are skipped.
const FIELDS: &[&str] = &["Image", "ParentImage", "User", "IntegrityLevel", "Company", "Signed"];

/// Take the basename of a Windows path: `C:\...\powershell.exe` -> `powershell.exe`.
fn basename(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// Parse a raw key=value Event ID 1 record into a list of `"Field::value"` tokens.
///
/// Only the fields in [`FIELDS`] are emitted; `Image`/`ParentImage` are reduced to
/// their basename so the vocabulary stays small and meaningful.
fn parse_record(raw: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        let Some((key, value)) = line.split_once('=') else { continue };
        if !FIELDS.contains(&key) {
            continue;
        }
        let value = match key {
            "Image" | "ParentImage" => basename(value),
            _ => value,
        };
        tokens.push(format!("{key}::{value}"));
    }
    tokens
}

// ── 2. synthetic Event ID 1 log generator (so the demo is self-contained) ──────

/// Generate `n` raw key=value Event ID 1 records plus their labels.
///
/// Label = suspicious (class 1) when either:
///   - an Office app (WINWORD/EXCEL) spawns a shell (powershell/cmd), or
///   - the process runs at `System` integrity while unsigned.
/// Otherwise benign (class 0).
fn make(n: usize, seed: u64) -> (Vec<String>, Vec<usize>) {
    let images = [
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\Windows\System32\cmd.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Windows\System32\svchost.exe",
    ];
    let parents = [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        r"C:\Windows\explorer.exe",
        r"C:\Windows\System32\services.exe",
    ];
    let users = ["CORP\\alice", "CORP\\bob", "NT AUTHORITY\\SYSTEM"];
    let integrities = ["Medium", "High", "System"];
    let companies = ["Microsoft Corporation", "Google LLC", "<unknown>"];

    let mut rng = Rng::new(seed);
    let mut records = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for _ in 0..n {
        let image = images[rng.below(images.len())];
        let parent = parents[rng.below(parents.len())];
        let user = users[rng.below(users.len())];
        let integrity = integrities[rng.below(integrities.len())];
        let company = companies[rng.below(companies.len())];
        let signed = rng.next_u64() & 1 == 0;

        let image_b = basename(image);
        let parent_b = basename(parent);

        let office_parent = parent_b.eq_ignore_ascii_case("WINWORD.EXE")
            || parent_b.eq_ignore_ascii_case("EXCEL.EXE");
        let shell_child = image_b.eq_ignore_ascii_case("powershell.exe")
            || image_b.eq_ignore_ascii_case("cmd.exe");
        let suspicious = (office_parent && shell_child) || (integrity == "System" && !signed);

        let raw = format!(
            "EventID=1\n\
             Image={image}\n\
             ParentImage={parent}\n\
             User={user}\n\
             IntegrityLevel={integrity}\n\
             Company={company}\n\
             Signed={signed}\n"
        );
        records.push(raw);
        labels.push(suspicious as usize);
    }
    (records, labels)
}

// ── 3. encode with the existing categorical encoder + train ────────────────────

fn main() {
    let (train_raw, train_y) = make(2000, 1);
    let (test_raw, test_y) = make(2000, 2);

    // Parse raw logs -> `col::val` tokens.
    let train_tok: Vec<Vec<String>> = train_raw.iter().map(|r| parse_record(r)).collect();
    let test_tok: Vec<Vec<String>> = test_raw.iter().map(|r| parse_record(r)).collect();

    // Borrow as &[&[&str]] for the categorical encoder API.
    let train_refs_v: Vec<Vec<&str>> =
        train_tok.iter().map(|s| s.iter().map(String::as_str).collect()).collect();
    let test_refs_v: Vec<Vec<&str>> =
        test_tok.iter().map(|s| s.iter().map(String::as_str).collect()).collect();
    let train_refs: Vec<&[&str]> = train_refs_v.iter().map(|v| v.as_slice()).collect();
    let test_refs: Vec<&[&str]> = test_refs_v.iter().map(|v| v.as_slice()).collect();

    // Show what one record becomes.
    println!("example record tokens:");
    for t in &train_tok[0] {
        println!("  {t}");
    }
    println!();

    let encoder = Encoder::fit_categorical(&train_refs);
    println!("vocabulary: {} `col::val` features\n", encoder.n_features());

    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 20, 15, 5.0, 8, true, 42);

    let train_x = encoder.encode_batch_categorical(&train_refs);
    let test_x = encoder.encode_batch_categorical(&test_refs);

    for epoch in 1..=30 {
        tm.fit_epoch(&train_x, &train_y);
        println!("epoch {epoch:>2}  accuracy={:.2}%", tm.accuracy(&test_x, &test_y) * 100.0);
    }

    // Interpretability: list the vocabulary tokens the model could key on.
    println!("\nlearned vocabulary tokens:");
    for bit in 0..encoder.n_features() {
        println!("  bit {bit:>2} -> {}", encoder.vocab_token(bit));
    }
}
