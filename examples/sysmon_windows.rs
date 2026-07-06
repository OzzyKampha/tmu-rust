//! Sysmon **time-window** demo — aggregate 10-minute spans of Event ID 1
//! (ProcessCreate) logs into a single TM sample.
//!
//! The modeling unit here is a *window*, not a single event. Each window's
//! events are aggregated into two feature blocks, concatenated into one bit
//! vector, and packed with [`EncodedBatch::from_bit_rows`]:
//!
//! - **presence + count** of each `col::val` token, thermometer-encoded
//!   (`count >= {1, 3, 10}`) — `count >= 1` is plain presence;
//! - **window aggregates** (total event count, distinct image count),
//!   thermometer-encoded.
//!
//! Every feature stays interpretable, so a learned clause reads like
//! *"WINWORD→powershell edge present AND powershell count ≥ 3 → suspicious"*.
//!
//! ## Run (self-contained synthetic windows — trains + reports accuracy)
//! ```text
//! cargo run --release --example sysmon_windows
//! ```
//!
//! ## Run on a real Mordor / Security-Datasets NDJSON file (feature-extraction demo)
//! ```text
//! # one-time download (see https://github.com/OTRF/Security-Datasets):
//! curl -L -o ds.zip https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/empire_launcher_vbs.zip
//! unzip ds.zip
//! cargo run --release --example sysmon_windows -- empire_launcher_vbs_2020-09-04160940.json
//! ```
//! Mordor data is unlabeled, so the real-file mode parses Event ID 1 records,
//! groups them into 10-min windows, and prints each window's feature vector —
//! demonstrating the pipeline on genuine telemetry.
//!
//! [`EncodedBatch::from_bit_rows`]: tmu_rs::EncodedBatch::from_bit_rows

use std::collections::{BTreeMap, BTreeSet};

use tmu_rs::{EncodedBatch, Rng, TsetlinMachine};

const WINDOW_SECS: u64 = 600; // 10 minutes
const TOKEN_THRESHOLDS: &[u32] = &[1, 3, 10]; // per-token count thermometer (>=1 is presence)
const TOTAL_THRESHOLDS: &[u32] = &[5, 20, 50, 100]; // window event-count thermometer
const DISTINCT_THRESHOLDS: &[u32] = &[2, 5, 10]; // distinct-image thermometer

// ── one process-create event (the fields we keep) ──────────────────────────────

#[derive(Clone)]
struct Event {
    time_sec: u64,
    image: String,  // basename, e.g. "powershell.exe"
    parent: String, // basename, e.g. "WINWORD.EXE"
    user: String,
    integrity: String,
    company: String,
    signed: bool,
}

/// `col::val` tokens describing one event (including the parent→child edge).
fn event_tokens(e: &Event) -> Vec<String> {
    vec![
        format!("Image::{}", e.image),
        format!("ParentImage::{}", e.parent),
        format!("User::{}", e.user),
        format!("IntegrityLevel::{}", e.integrity),
        format!("Company::{}", e.company),
        format!("Signed::{}", e.signed),
        format!("edge::{}->{}", e.parent, e.image),
    ]
}

/// A 10-minute window: the events that fell within it.
type Window = Vec<Event>;

/// Group a time-sorted event stream into fixed 10-minute windows.
fn into_windows(mut events: Vec<Event>) -> Vec<Window> {
    events.sort_by_key(|e| e.time_sec);
    let mut windows: BTreeMap<u64, Window> = BTreeMap::new();
    for e in events {
        windows.entry(e.time_sec / WINDOW_SECS).or_default().push(e);
    }
    windows.into_values().collect()
}

// ── window encoder: tokens + counts -> bit vector ──────────────────────────────

struct WindowEncoder {
    tokens: Vec<String>,        // sorted vocabulary of col::val tokens
    feature_names: Vec<String>, // human-readable name per output bit
}

impl WindowEncoder {
    /// Build the token vocabulary from the training windows.
    fn fit(windows: &[Window]) -> Self {
        let mut vocab: BTreeSet<String> = BTreeSet::new();
        for w in windows {
            for e in w {
                vocab.extend(event_tokens(e));
            }
        }
        let tokens: Vec<String> = vocab.into_iter().collect();

        // Build the parallel list of feature names (same order as encode()).
        let mut feature_names = Vec::new();
        for t in &tokens {
            for thr in TOKEN_THRESHOLDS {
                feature_names.push(format!("count({t}) >= {thr}"));
            }
        }
        for thr in TOTAL_THRESHOLDS {
            feature_names.push(format!("window_events >= {thr}"));
        }
        for thr in DISTINCT_THRESHOLDS {
            feature_names.push(format!("distinct_images >= {thr}"));
        }
        Self {
            tokens,
            feature_names,
        }
    }

    fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    fn feature_name(&self, bit: usize) -> &str {
        &self.feature_names[bit]
    }

    /// Encode one window into a 0/1 bit vector of length `n_features()`.
    fn encode(&self, w: &Window) -> Vec<u8> {
        // Count token occurrences across all events in the window.
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut images: BTreeSet<&str> = BTreeSet::new();
        for e in w {
            images.insert(e.image.as_str());
            for t in event_tokens(e) {
                *counts.entry(t).or_insert(0) += 1;
            }
        }

        let mut bits = vec![0u8; self.n_features()];
        let mut cursor = 0;

        // per-token count thermometer
        for tok in &self.tokens {
            let c = counts.get(tok).copied().unwrap_or(0);
            for (k, thr) in TOKEN_THRESHOLDS.iter().enumerate() {
                bits[cursor + k] = (c >= *thr) as u8;
            }
            cursor += TOKEN_THRESHOLDS.len();
        }

        // window total-events thermometer
        let total = w.len() as u32;
        for (k, thr) in TOTAL_THRESHOLDS.iter().enumerate() {
            bits[cursor + k] = (total >= *thr) as u8;
        }
        cursor += TOTAL_THRESHOLDS.len();

        // distinct-image thermometer
        let distinct = images.len() as u32;
        for (k, thr) in DISTINCT_THRESHOLDS.iter().enumerate() {
            bits[cursor + k] = (distinct >= *thr) as u8;
        }

        bits
    }

    /// Encode many windows into a single batch (showcases `from_bit_rows`).
    fn encode_batch(&self, windows: &[Window]) -> EncodedBatch {
        let rows: Vec<Vec<u8>> = windows.iter().map(|w| self.encode(w)).collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        EncodedBatch::from_bit_rows(&refs, self.n_features())
    }
}

// ── synthetic 10-min-window generator (labeled, learnable) ──────────────────────

/// Draw a "normal" background event.
fn benign_event(rng: &mut Rng, t: u64) -> Event {
    let imgs = ["chrome.exe", "svchost.exe", "notepad.exe", "OUTLOOK.EXE"];
    let pars = ["explorer.exe", "services.exe"];
    let img = imgs[rng.below(imgs.len())];
    Event {
        time_sec: t,
        image: img.to_string(),
        parent: pars[rng.below(pars.len())].to_string(),
        user: "CORP\\alice".to_string(),
        integrity: ["Medium", "High"][rng.below(2)].to_string(),
        company: ["Microsoft Corporation", "Google LLC"][rng.below(2)].to_string(),
        signed: true,
    }
}

/// Generate `n` labeled 10-min windows of synthetic process-create activity.
///
/// Suspicious (class 1) windows embed a small attack burst — an Office app
/// spawning a script host that in turn spawns several powershell processes —
/// mirroring the real `empire_launcher_vbs` Mordor trace.
fn make(n: usize, seed: u64) -> (Vec<Window>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut windows = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for w in 0..n {
        let base = (w as u64) * WINDOW_SECS;
        let mut events = Vec::new();

        // benign background: 8..40 events
        let bg = 8 + rng.below(32);
        for _ in 0..bg {
            let t = base + rng.below(WINDOW_SECS as usize) as u64;
            events.push(benign_event(&mut rng, t));
        }

        let suspicious = rng.next_u64() & 1 == 0;
        if suspicious {
            let t = base + rng.below(WINDOW_SECS as usize) as u64;
            // Office -> script host
            events.push(Event {
                time_sec: t,
                image: "wscript.exe".to_string(),
                parent: "WINWORD.EXE".to_string(),
                user: "CORP\\alice".to_string(),
                integrity: "Medium".to_string(),
                company: "Microsoft Corporation".to_string(),
                signed: true,
            });
            // script host -> several powershell spawns (high count signal)
            for _ in 0..(3 + rng.below(6)) {
                events.push(Event {
                    time_sec: t + rng.below(60) as u64,
                    image: "powershell.exe".to_string(),
                    parent: "wscript.exe".to_string(),
                    user: "CORP\\alice".to_string(),
                    integrity: "Medium".to_string(),
                    company: "<unknown>".to_string(),
                    signed: false,
                });
            }
        }

        windows.push(events);
        labels.push(suspicious as usize);
    }
    (windows, labels)
}

// ── real Mordor NDJSON reader (Event ID 1 only) ─────────────────────────────────

fn basename(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// Parse `"YYYY-MM-DD HH:MM:SS"` into a monotonic second count.
///
/// Day-of-month * 86400 + time-of-day is enough to window a single trace; it is
/// not a real epoch (good enough for grouping, documented simplification).
fn parse_time(s: &str) -> Option<u64> {
    let (date, time) = s.split_once(' ')?;
    let mut d = date.split('-');
    let (_y, _m, dd) = (d.next()?, d.next()?, d.next()?);
    let mut t = time.split(':');
    let (hh, mm, ss) = (t.next()?, t.next()?, t.next()?);
    let dd: u64 = dd.parse().ok()?;
    let hh: u64 = hh.parse().ok()?;
    let mm: u64 = mm.parse().ok()?;
    let ss: u64 = ss.parse().ok()?;
    Some(dd * 86_400 + hh * 3_600 + mm * 60 + ss)
}

/// Read Sysmon Event ID 1 records from a Mordor/Security-Datasets NDJSON file.
fn read_mordor(path: &str) -> std::io::Result<Vec<Event>> {
    let text = std::fs::read_to_string(path)?;
    let mut events = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let channel = v.get("Channel").and_then(|c| c.as_str()).unwrap_or("");
        let eid = v.get("EventID").and_then(|e| e.as_u64()).unwrap_or(0);
        if !channel.starts_with("Microsoft-Windows-Sysmon") || eid != 1 {
            continue;
        }
        let get = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
        let Some(time_sec) = parse_time(&get("EventTime")) else {
            continue;
        };
        let company = get("Company");
        events.push(Event {
            time_sec,
            image: basename(&get("Image")).to_string(),
            parent: basename(&get("ParentImage")).to_string(),
            user: get("User"),
            integrity: get("IntegrityLevel"),
            signed: company.contains("Microsoft") || company.contains("Google"),
            company: if company.is_empty() {
                "<unknown>".to_string()
            } else {
                company
            },
        });
    }
    Ok(events)
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        real_file_mode(&path);
    } else {
        synthetic_mode();
    }
}

/// Default: train + evaluate on labeled synthetic 10-min windows.
fn synthetic_mode() {
    let (train_w, train_y) = make(1500, 1);
    let (test_w, test_y) = make(1500, 2);

    let encoder = WindowEncoder::fit(&train_w);
    println!(
        "10-min windows: {} train / {} test, {} interpretable features\n",
        train_w.len(),
        test_w.len(),
        encoder.n_features()
    );

    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 40, 20, 5.0, 8, true, 42);
    let train_x = encoder.encode_batch(&train_w);
    let test_x = encoder.encode_batch(&test_w);

    for epoch in 1..=40 {
        tm.fit_epoch(&train_x, &train_y);
        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "epoch {epoch:>2}  accuracy={:.2}%",
                tm.accuracy(&test_x, &test_y) * 100.0
            );
        }
    }

    // Interpretability: show a handful of the window features available to the TM.
    println!(
        "\nsample window features (first 12 of {}):",
        encoder.n_features()
    );
    for bit in 0..encoder.n_features().min(12) {
        println!("  bit {bit:>3} -> {}", encoder.feature_name(bit));
    }
}

/// Real-file: parse Mordor NDJSON, window it, and print each window's features.
fn real_file_mode(path: &str) {
    let events = match read_mordor(path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("could not read {path}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "parsed {} Sysmon Event ID 1 records from {path}",
        events.len()
    );
    if events.is_empty() {
        eprintln!("no Event ID 1 records found — is this a Mordor/Security-Datasets NDJSON file?");
        return;
    }

    let windows = into_windows(events);
    let encoder = WindowEncoder::fit(&windows);
    println!(
        "grouped into {} 10-min window(s), {} features\n",
        windows.len(),
        encoder.n_features()
    );

    for (i, w) in windows.iter().enumerate() {
        let bits = encoder.encode(w);
        let active: Vec<&str> = bits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == 1)
            .map(|(bit, _)| encoder.feature_name(bit))
            .collect();
        println!(
            "window {i}: {} events, {} active features",
            w.len(),
            active.len()
        );
        for name in active.iter().take(20) {
            println!("    {name}");
        }
        if active.len() > 20 {
            println!("    … {} more", active.len() - 20);
        }
    }
}
