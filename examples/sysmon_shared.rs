//! Shared utilities for the sysmon_* examples.
//! Included via `#[path]` — not a standalone crate module.

#![allow(dead_code)]

use tmu_rs::Rng;

// ── path helpers ───────────────────────────────────────────────────────────────

/// Last path component: `C:\...\powershell.exe` → `"powershell.exe"`.
pub fn basename(path: &str) -> String {
    path.rsplit(['\\', '/']).next().unwrap_or(path).to_string()
}

/// First segment of a registry path (the hive): `HKLM\SOFTWARE\...` → `"HKLM"`.
pub fn hive_of(path: &str) -> String {
    path.split('\\').next().unwrap_or(path).to_string()
}

/// Parent directory basename: `C:\Windows\Temp\foo.tmp` → `"Temp"`.
pub fn parent_dir(path: &str) -> String {
    let without_file = path.rsplitn(2, ['\\', '/']).nth(1).unwrap_or("");
    without_file.rsplit(['\\', '/']).next().unwrap_or(".").to_string()
}

// ── timestamp ──────────────────────────────────────────────────────────────────

/// Parse `"YYYY-MM-DD HH:MM:SS"` into a monotonic second count.
/// Uses day * 86400 + time-of-day — enough for intra-trace windowing.
pub fn parse_time(s: &str) -> Option<u64> {
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

// ── synthetic benign event generator ──────────────────────────────────────────

/// Generate a synthetic benign event as a list of `col::val` tokens.
///
/// Produces a realistic mix of Sysmon event types (mirroring the distribution
/// in the Mordor attack datasets) but with benign-looking field values.
pub fn benign_tokens(rng: &mut Rng) -> Vec<String> {
    // Weighted event-type distribution matching the real attack dataset:
    // EID 10 ~40%, EID 7 ~30%, EID 12/13 ~27%, EID 1 ~3%.
    let roll = rng.below(100);
    match roll {
        0..=39 => benign_eid10(rng),
        40..=69 => benign_eid7(rng),
        70..=86 => benign_eid12(rng),
        87..=93 => benign_eid13(rng),
        94..=97 => benign_eid1(rng),
        _ => benign_eid11(rng),
    }
}

// ── per-type benign generators ─────────────────────────────────────────────────

fn benign_eid1(rng: &mut Rng) -> Vec<String> {
    let imgs = ["chrome.exe", "notepad.exe", "OUTLOOK.EXE", "mspaint.exe", "calc.exe"];
    let pars = ["explorer.exe", "taskmgr.exe", "ctfmon.exe"];
    tok(1, &[
        ("Image",          imgs[rng.below(imgs.len())]),
        ("ParentImage",    pars[rng.below(pars.len())]),
        ("IntegrityLevel", "Medium"),
        ("Company",        "Microsoft Corporation"),
        ("Signed",         "true"),
    ])
}

fn benign_eid7(rng: &mut Rng) -> Vec<String> {
    let imgs = ["svchost.exe", "lsass.exe", "explorer.exe", "chrome.exe", "OUTLOOK.EXE"];
    let libs = ["ntdll.dll", "kernel32.dll", "user32.dll", "advapi32.dll", "msvcrt.dll"];
    tok(7, &[
        ("Image",       imgs[rng.below(imgs.len())]),
        ("ImageLoaded", libs[rng.below(libs.len())]),
        ("Signed",      "true"),
        ("Company",     "Microsoft Corporation"),
    ])
}

fn benign_eid10(rng: &mut Rng) -> Vec<String> {
    // Normal process access: browsers, system services accessing each other.
    let srcs = ["svchost.exe", "explorer.exe", "chrome.exe", "services.exe", "csrss.exe"];
    let tgts = ["svchost.exe", "explorer.exe", "wininit.exe", "smss.exe", "winlogon.exe"];
    let acc  = ["0x1000", "0x1400", "0x100000", "0x1fffff"];
    tok(10, &[
        ("SourceImage",    srcs[rng.below(srcs.len())]),
        ("TargetImage",    tgts[rng.below(tgts.len())]),
        ("GrantedAccess",  acc[rng.below(acc.len())]),
    ])
}

fn benign_eid11(rng: &mut Rng) -> Vec<String> {
    let imgs = ["svchost.exe", "explorer.exe", "OUTLOOK.EXE"];
    let dirs = ["Temp", "Local", "AppData"];
    tok(11, &[
        ("Image",          imgs[rng.below(imgs.len())]),
        ("TargetFilename", dirs[rng.below(dirs.len())]),
    ])
}

fn benign_eid12(rng: &mut Rng) -> Vec<String> {
    let imgs = ["svchost.exe", "explorer.exe", "regedit.exe"];
    let hives = ["HKLM", "HKCU", "HKU"];
    tok(12, &[
        ("Image",        imgs[rng.below(imgs.len())]),
        ("TargetObject", hives[rng.below(hives.len())]),
    ])
}

fn benign_eid13(rng: &mut Rng) -> Vec<String> {
    let imgs  = ["svchost.exe", "explorer.exe"];
    let hives = ["HKLM", "HKCU"];
    tok(13, &[
        ("Image",        imgs[rng.below(imgs.len())]),
        ("TargetObject", hives[rng.below(hives.len())]),
    ])
}

fn tok(eid: u32, fields: &[(&str, &str)]) -> Vec<String> {
    let mut out = vec![format!("EventID::{eid}")];
    for (col, val) in fields {
        if !val.is_empty() {
            out.push(format!("{col}::{val}"));
        }
    }
    out
}
