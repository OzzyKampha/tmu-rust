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

// ── behavioral attack labeling ─────────────────────────────────────────────────
//
// Labels events by WHAT THEY DO (Sysmon event type + field values), not by
// process name.  This catches LOLBin attacks and removes the circular dependency
// on ATTACK_PROCS that caused the TM to learn "powershell.exe = attack".
//
// Mordor datasets are labeled at the scenario/file level by the tactic directory
// structure (execution/, credential_access/, etc.).  Within each file, background
// Windows noise also appears.  These rules separate the actual attack activity
// from that noise.

pub fn is_attack_behavior(v: &serde_json::Value, eid: u32) -> bool {
    match eid {
        // EID 8 — CreateRemoteThread: classic process injection technique.
        8 => true,

        // EID 9 — RawAccessRead: direct disk read bypassing the filesystem,
        // used for credential theft (NTDS.dit, SAM) and volume shadow copy abuse.
        9 => true,

        // EID 10 — ProcessAccess: check both target process and access rights.
        // Memory read/write/dup-handle on credential-store processes = credential dump.
        10 => {
            let target = basename(v["TargetImage"].as_str().unwrap_or("")).to_lowercase();
            let access_str = v["GrantedAccess"].as_str().unwrap_or("0x0");
            let access = u64::from_str_radix(access_str.trim_start_matches("0x"), 16)
                .unwrap_or(0);
            let sensitive_target = matches!(
                target.as_str(),
                "lsass.exe" | "winlogon.exe" | "csrss.exe" | "wininit.exe" | "services.exe"
            );
            // VM_READ(0x10) | VM_WRITE(0x20) | PROCESS_DUP_HANDLE(0x40)
            // or any full-access mask
            let suspicious_access = access & 0x70 != 0 || access >= 0x1f0000;
            sensitive_target && suspicious_access
        }

        // EID 11 — FileCreate: executable dropped in a staging location by a
        // scripting host (T1059 Command and Scripting Interpreter).
        11 => {
            let fname = v["TargetFilename"].as_str().unwrap_or("").to_lowercase();
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();
            let scripting = matches!(
                image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                    | "mshta.exe" | "python.exe" | "wmic.exe"
            );
            let staging = fname.contains("\\temp\\")
                || fname.contains("\\appdata\\local\\temp")
                || fname.contains("\\public\\")
                || fname.contains("\\programdata\\");
            let executable = fname.ends_with(".exe")
                || fname.ends_with(".dll")
                || fname.ends_with(".ps1")
                || fname.ends_with(".bat")
                || fname.ends_with(".vbs")
                || fname.ends_with(".js");
            scripting && (staging || executable)
        }

        // EID 12/13 — RegistryEvent: writes to persistence or defense-evasion keys.
        12 | 13 => {
            let obj = v["TargetObject"].as_str().unwrap_or("").to_lowercase();
            obj.contains("\\run\\")
                || obj.contains("\\runonce\\")
                || obj.contains("userinitmprlogonscript")
                || obj.contains("\\currentversion\\winlogon")
                || obj.contains("\\currentversion\\windows\\load")
                || obj.contains("\\policies\\explorer\\run")
                || obj.contains("\\securityproviders\\")
                || obj.contains("minint")   // MinInt key disables event logging
                || (obj.contains("\\services\\") && obj.contains("\\start"))
                || (obj.contains("\\eventlog\\") && obj.contains("\\start"))
        }

        // EID 3 — NetworkConnect: outbound from LOLBins used for C2 staging.
        3 => {
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();
            let initiated = v["Initiated"].as_str().unwrap_or("") == "true";
            let lolbin = matches!(
                image.as_str(),
                "mshta.exe" | "regsvr32.exe" | "wscript.exe" | "cscript.exe"
                    | "certutil.exe" | "bitsadmin.exe" | "wmic.exe" | "rundll32.exe"
                    | "odbcconf.exe" | "cmstp.exe"
            );
            lolbin && initiated
        }

        // EID 1 — ProcessCreate: suspicious parent→child chains.
        1 => {
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();
            let parent = basename(v["ParentImage"].as_str().unwrap_or("")).to_lowercase();

            // Dedicated offensive tools — never legitimately present.
            let attack_tool = matches!(
                image.as_str(),
                "mimikatz.exe" | "rubeus.exe" | "sharpdpapi.exe" | "sharpdump.exe"
                    | "procdump.exe" | "procdump64.exe" | "wce.exe" | "fgdump.exe"
                    | "psexec.exe" | "psexec64.exe" | "psexesvc.exe" | "paexec.exe"
                    | "sharpview.exe" | "seatbelt.exe" | "bloodhound.exe" | "sharphound.exe"
                    | "ncat.exe" | "covenantgruntstager.exe"
            );

            // Office macro → scripting host (T1566 Phishing / T1059).
            let office_to_script = matches!(
                parent.as_str(),
                "winword.exe" | "excel.exe" | "powerpnt.exe" | "outlook.exe" | "onenote.exe"
            ) && matches!(
                image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                    | "mshta.exe" | "regsvr32.exe" | "rundll32.exe"
            );

            // Scripting host → recon tools (T1087/T1069/T1016 Discovery).
            let script_to_recon = matches!(
                parent.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
            ) && matches!(
                image.as_str(),
                "whoami.exe" | "net.exe" | "net1.exe" | "nltest.exe" | "ipconfig.exe"
                    | "systeminfo.exe" | "tasklist.exe" | "arp.exe" | "nslookup.exe"
                    | "ping.exe" | "netstat.exe" | "query.exe" | "dsquery.exe"
            );

            attack_tool || office_to_script || script_to_recon
        }

        // EID 19/20/21 — WMI event subscription: persistence / lateral movement.
        19 | 20 | 21 => true,

        _ => false,
    }
}
