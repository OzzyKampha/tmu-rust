//! Shared utilities for the sysmon_* examples.
//! Included via `#[path]` — not a standalone crate module.

// ── path helpers ───────────────────────────────────────────────────────────────

/// Last path component: `C:\...\powershell.exe` → `"powershell.exe"`.
pub fn basename(path: &str) -> String {
    path.rsplit(['\\', '/']).next().unwrap_or(path).to_string()
}

/// First segment of a registry path (the hive): `HKLM\SOFTWARE\...` → `"HKLM"`.
pub fn hive_of(path: &str) -> String {
    path.split('\\').next().unwrap_or(path).to_string()
}
