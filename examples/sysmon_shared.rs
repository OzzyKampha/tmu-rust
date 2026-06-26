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

// ── rule display helpers ───────────────────────────────────────────────────────

/// Token prefixes that indicate attack-relevant fields (process, file, network, registry).
/// Used to rank clause rules by security significance.
pub const MEANINGFUL_PREFIXES: &[&str] = &[
    "process.name", "process.args", "process.parent",
    "dll.name", "file.name", "file.path",
    "target.process", "source.process",
    "dns.question", "destination.",
    "network.", "registry.",
];

/// Maps an ECS token string to a short security annotation for rule display.
/// Returns an empty string if the token has no specific security note.
pub fn explain_token(tok: &str) -> &'static str {
    if tok.starts_with("event.id::1 ")  || tok == "event.id::1"  { return "Sysmon: process creation"; }
    if tok.starts_with("event.id::3 ")  || tok == "event.id::3"  { return "Sysmon: network connection"; }
    if tok.starts_with("event.id::7 ")  || tok == "event.id::7"  { return "Sysmon: image/DLL loaded"; }
    if tok.starts_with("event.id::10")                           { return "Sysmon: process access (injection / cred dump)"; }
    if tok.starts_with("event.id::11")                           { return "Sysmon: file created"; }
    if tok.starts_with("event.id::12") || tok.starts_with("event.id::13") || tok.starts_with("event.id::14") {
        return "Sysmon: registry create/set/delete";
    }
    if tok.starts_with("event.id::22")                           { return "Sysmon: DNS query"; }
    if tok.starts_with("event.id::25")                           { return "Sysmon: process tampering"; }
    if tok.starts_with("process.name::")                         { return "executing process binary"; }
    if tok.starts_with("process.args::")                         { return "command-line argument token"; }
    if tok.starts_with("process.parent.name::") || tok.starts_with("process.parent::") { return "parent process"; }
    if tok.starts_with("target.process.name::")                  { return "victim process (accessed/injected)"; }
    if tok.starts_with("source.process.name::")                  { return "accessor/injecting process"; }
    if tok.starts_with("target.process.granted_access::")        { return "access rights requested (PROCESS_VM_READ etc.)"; }
    if tok.starts_with("file.path::Temp") || tok.contains("::Temp") { return "staging in temp directory (common dropper)"; }
    if tok.starts_with("file.path::System32")                    { return "system directory (legit or DLL hijack)"; }
    if tok.starts_with("file.path::AppData")                     { return "user profile staging area"; }
    if tok.starts_with("file.path::")                            { return "file location"; }
    if tok.starts_with("file.name::")                            { return "file name"; }
    if tok.starts_with("dll.name::")                             { return "DLL loaded"; }
    if tok.starts_with("network.destination") || tok.starts_with("destination.") { return "outbound C2 / lateral target"; }
    if tok.starts_with("dns.question::")                         { return "DNS lookup (C2 beacon / lateral)"; }
    if tok.starts_with("registry.")                              { return "registry operation"; }
    if tok.contains("::lsass")                                   { return "LSASS — credential store (high value target)"; }
    if tok.contains("::mimikatz") || tok.contains("::mimi")      { return "Mimikatz credential dumper"; }
    if tok.contains("::powershell") || tok.contains("::psh")     { return "PowerShell execution"; }
    if tok.contains("::cmd.exe")                                 { return "Command prompt execution"; }
    if tok.contains("::rundll32")                                { return "LOLBin: runs arbitrary DLLs"; }
    if tok.contains("::regsvr32")                                { return "LOLBin: runs COM/SCT payloads"; }
    if tok.contains("::mshta")                                   { return "LOLBin: runs HTML/VBS/JS"; }
    if tok.contains("::wmic")                                    { return "WMI execution / lateral movement"; }
    if tok.contains("::schtasks") || tok.contains("::at.exe")    { return "scheduled task (persistence)"; }
    if tok.contains("::net.exe") || tok.contains("::net1.exe")   { return "domain/user enumeration"; }
    if tok.contains("Signature::Microsoft Windows")              { return "Microsoft-signed binary (very common — not specific)"; }
    if tok.contains("code_signature.status::Valid")              { return "valid code signature (most legit processes)"; }
    if tok.contains("UserID::S-1-5-18")                         { return "SYSTEM account"; }
    if tok.contains("UserID::S-1-5-19") || tok.contains("UserID::S-1-5-20") { return "LOCAL/NETWORK SERVICE account"; }
    ""
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
        // ── EID 1 — ProcessCreate ────────────────────────────────────────────────
        1 => {
            let image   = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();
            let parent  = basename(v["ParentImage"].as_str().unwrap_or("")).to_lowercase();
            let cmdline = v["CommandLine"].as_str().unwrap_or("").to_lowercase();
            let img_path = v["Image"].as_str().unwrap_or("").to_lowercase();

            // Known offensive tools — never legitimately present.
            let attack_tool = matches!(image.as_str(),
                "mimikatz.exe" | "rubeus.exe" | "sharpdpapi.exe" | "sharpdump.exe"
                | "procdump.exe" | "procdump64.exe" | "wce.exe" | "fgdump.exe"
                | "psexec.exe" | "psexec64.exe" | "psexesvc.exe" | "paexec.exe"
                | "sharpview.exe" | "seatbelt.exe" | "bloodhound.exe" | "sharphound.exe"
                | "ncat.exe" | "nc.exe" | "covenantgruntstager.exe"
                | "lazagne.exe" | "pwdump.exe" | "pwdump6.exe" | "pwdump7.exe"
                | "gsecdump.exe" | "cachedump.exe" | "winhex.exe"
                | "adfind.exe" | "adrecon.exe" | "ldapdomaindump.exe"
                | "kerbrute.exe" | "kekeo.exe" | "safetykatz.exe" | "sharpkatz.exe"
                | "nanodump.exe" | "dumpert.exe" | "handlekatz.exe"
                | "crackmapexec.exe" | "impacket-secretsdump.exe"
                | "invoke-mimikatz.exe" | "invoke-ninjacopy.exe"
                | "sharpchisel.exe" | "chisel.exe" | "ligolo.exe" | "rpivot.exe"
                | "cobaltstrike.exe" | "beacon.exe" | "brute_ratel.exe" | "nighthawk.exe"
                | "accesschk.exe" | "accesschk64.exe"
            );

            // Office macro → scripting host (T1566 phishing / T1059).
            let office_to_script = matches!(parent.as_str(),
                "winword.exe" | "excel.exe" | "powerpnt.exe" | "outlook.exe"
                | "onenote.exe" | "msaccess.exe" | "mspub.exe" | "visio.exe"
                | "eqnedt32.exe"
            ) && matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "regsvr32.exe" | "rundll32.exe" | "certutil.exe"
                | "bitsadmin.exe" | "curl.exe" | "wget.exe" | "msiexec.exe"
                | "installutil.exe" | "msbuild.exe"
            );

            // LOLBin parent → scripting child (living-off-the-land chains).
            let lolbin_to_script = matches!(parent.as_str(),
                "msbuild.exe" | "installutil.exe" | "regasm.exe" | "regsvcs.exe"
                | "ieexec.exe" | "cmstp.exe" | "odbcconf.exe" | "xwizard.exe"
                | "dnscmd.exe" | "esentutl.exe" | "certutil.exe" | "bitsadmin.exe"
                | "msiexec.exe" | "regsvr32.exe" | "mshta.exe" | "wmic.exe"
                | "hh.exe" | "pcalua.exe" | "syncappvpublishingserver.exe"
                | "appsyncpublishingserver.exe" | "presentationhost.exe"
                | "diskshadow.exe" | "msconfig.exe" | "wab.exe" | "extrac32.exe"
            ) && matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "regsvr32.exe" | "rundll32.exe"
            );

            // Scripting host → recon/discovery (T1087/T1069/T1016/T1057/T1083).
            let script_to_recon = matches!(parent.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe" | "mshta.exe"
            ) && matches!(image.as_str(),
                "whoami.exe" | "net.exe" | "net1.exe" | "nltest.exe" | "ipconfig.exe"
                | "systeminfo.exe" | "tasklist.exe" | "arp.exe" | "nslookup.exe"
                | "ping.exe" | "netstat.exe" | "query.exe" | "dsquery.exe" | "dsget.exe"
                | "gpresult.exe" | "reg.exe" | "sc.exe" | "wmic.exe" | "schtasks.exe"
                | "at.exe" | "attrib.exe" | "dir.exe" | "tree.exe"
                | "nbtscan.exe" | "nmap.exe" | "masscan.exe" | "fsutil.exe"
                | "auditpol.exe" | "bcdedit.exe" | "dnscmd.exe"
                | "wevtutil.exe" | "vssadmin.exe" | "diskshadow.exe"
            );

            // System process → shell (injection artifact / process hollowing).
            let system_spawns_shell = matches!(parent.as_str(),
                "lsass.exe" | "spoolsv.exe" | "winlogon.exe" | "dllhost.exe"
                | "taskhost.exe" | "taskhostw.exe" | "userinit.exe"
            ) && matches!(image.as_str(),
                "cmd.exe" | "powershell.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "rundll32.exe" | "regsvr32.exe"
            );

            // Encoded / obfuscated PowerShell (T1059.001 / T1027).
            let encoded_ps = image == "powershell.exe"
                && (cmdline.contains("-encodedcommand")
                    || cmdline.contains(" -enc ")
                    || cmdline.contains(" -ec ")
                    || cmdline.contains("-nop")
                    || cmdline.contains("-bypass")
                    || cmdline.contains("hidden")
                    || cmdline.contains("iex(")
                    || cmdline.contains("iex (")
                    || cmdline.contains("invoke-expression")
                    || cmdline.contains("downloadstring")
                    || cmdline.contains("net.webclient")
                    || cmdline.contains("bitstransfer")
                    || cmdline.contains("start-bitstransfer")
                    || cmdline.contains("webclient")
                    || cmdline.contains("frombase64string"));

            // Certutil used as downloader/decoder (T1140 / T1105).
            let certutil_abuse = image == "certutil.exe"
                && (cmdline.contains("-decode")
                    || cmdline.contains("-decodehex")
                    || cmdline.contains("-urlcache")
                    || cmdline.contains("-verifyctl")
                    || cmdline.contains("http"));

            // Wscript/cscript running a file from a staging path.
            let script_from_staging = matches!(image.as_str(), "wscript.exe" | "cscript.exe")
                && (cmdline.contains("\\temp\\")
                    || cmdline.contains("\\appdata\\")
                    || cmdline.contains("\\public\\")
                    || cmdline.contains("\\programdata\\")
                    || cmdline.contains("\\downloads\\"));

            // Executable launched directly from a staging/writable path.
            let staged_exec = img_path.ends_with(".exe")
                && (img_path.contains("\\temp\\")
                    || img_path.contains("\\appdata\\local\\temp")
                    || img_path.contains("\\users\\public\\")
                    || img_path.contains("\\programdata\\")
                    || img_path.contains("\\downloads\\"));

            // T1548.002 UAC bypass binaries — auto-elevate without prompt.
            // These are abused to spawn a high-integrity shell without a UAC dialog.
            let uac_bypass = matches!(image.as_str(),
                "fodhelper.exe" | "eventvwr.exe" | "sdclt.exe" | "wsreset.exe"
                | "computerdefaults.exe" | "slui.exe" | "cmstp.exe" | "dccw.exe"
                | "mavinject.exe"  // T1218.013 — injects DLL into running process
            );

            // T1562.001 Impair Defenses — disable AV/EDR via service or cmdline.
            let impair_defenses =
                // sc.exe stop/delete on known security service names
                (image == "sc.exe" && (
                    cmdline.contains("windefend") || cmdline.contains("wdnissvc")
                    || cmdline.contains("sense")   || cmdline.contains("msseces")
                    || cmdline.contains("sfc")      || cmdline.contains("wscsvc")
                    || cmdline.contains("securityhealthservice")
                    || (cmdline.contains("stop") || cmdline.contains("delete"))))
                // taskkill targeting AV/EDR processes
                || (image == "taskkill.exe" && (
                    cmdline.contains("msmpeng") || cmdline.contains("msseces")
                    || cmdline.contains("mbam")   || cmdline.contains("avp.exe")
                    || cmdline.contains("avgnt")  || cmdline.contains("eav_trial")))
                // PowerShell disabling Windows Defender real-time protection
                || (image == "powershell.exe" && (
                    cmdline.contains("set-mppreference")
                    || cmdline.contains("disablerealtimemonitoring")
                    || cmdline.contains("disablebehaviormonitoring")
                    || cmdline.contains("disableioavprotection")
                    || cmdline.contains("add-mppreference -exclusion")))
                // auditpol clearing audit policy (T1562.002)
                || (image == "auditpol.exe" && (
                    cmdline.contains("/set") && (
                        cmdline.contains("success:disable")
                        || cmdline.contains("failure:disable")
                        || cmdline.contains("/clear"))));

            // T1070.001 Clear Windows Event Logs — remove forensic evidence.
            let clear_logs =
                (image == "wevtutil.exe" && (
                    cmdline.contains(" cl ") || cmdline.contains(" clear-log ")
                    || cmdline.contains(" sl ") || cmdline.contains("/ms:0")))
                || (image == "powershell.exe" && cmdline.contains("clear-eventlog"))
                || (image == "powershell.exe" && cmdline.contains("remove-eventlog"))
                || (image == "cmd.exe" && cmdline.contains("wevtutil") && cmdline.contains("cl "));

            // T1490 Inhibit System Recovery — delete backups/shadow copies pre-ransomware.
            let inhibit_recovery =
                (image == "vssadmin.exe" && (
                    cmdline.contains("delete") || cmdline.contains("shadows")))
                || (image == "bcdedit.exe" && (
                    cmdline.contains("recoveryenabled") || cmdline.contains("no")
                    || cmdline.contains("bootstatuspolicy")
                    || cmdline.contains("ignoreallfailures")))
                || (image == "wbadmin.exe" && (
                    cmdline.contains("delete") || cmdline.contains("catalog")
                    || cmdline.contains("systemstatebackup")))
                || (image == "powershell.exe" && (
                    cmdline.contains("get-wmiobject win32_shadowcopy")
                    || cmdline.contains(".delete()")));

            // T1136.001 Create Local Account — add a backdoor user.
            let create_account = (image == "net.exe" || image == "net1.exe")
                && cmdline.contains("user")
                && cmdline.contains("/add");

            // T1543.003 Create/Modify Windows Service with a suspicious binary path.
            let malicious_service = image == "sc.exe"
                && cmdline.contains("create")
                && (cmdline.contains("\\temp\\") || cmdline.contains("\\appdata\\")
                    || cmdline.contains("\\public\\") || cmdline.contains("\\programdata\\")
                    || cmdline.contains("\\users\\") || cmdline.contains("cmd.exe")
                    || cmdline.contains("powershell") || cmdline.contains("rundll32"));

            // T1053.005 Scheduled Task with attack payload in /tr argument.
            let schtask_attack = image == "schtasks.exe"
                && cmdline.contains("/create")
                && (cmdline.contains("powershell") || cmdline.contains("cmd.exe")
                    || cmdline.contains("wscript")  || cmdline.contains("cscript")
                    || cmdline.contains("mshta")    || cmdline.contains("regsvr32")
                    || cmdline.contains("rundll32") || cmdline.contains("\\temp\\")
                    || cmdline.contains("\\appdata\\") || cmdline.contains("encodedcommand")
                    || cmdline.contains("http"));

            // T1021.006 WinRM lateral movement — wsmprovhost spawns arbitrary commands.
            let winrm_exec = parent == "wsmprovhost.exe"
                && matches!(image.as_str(),
                    "cmd.exe" | "powershell.exe" | "net.exe" | "whoami.exe"
                    | "ipconfig.exe" | "systeminfo.exe" | "tasklist.exe");

            // T1059.004 Unix shell via WSL / bash (T1202 Indirect Command Execution).
            let wsl_exec = matches!(image.as_str(), "wsl.exe" | "bash.exe")
                && !cmdline.is_empty()
                && (cmdline.contains("-c ") || cmdline.contains("curl")
                    || cmdline.contains("wget") || cmdline.contains("python")
                    || cmdline.contains("nc ") || cmdline.contains("ncat"));

            // T1003.002/T1003.003 reg.exe dumping credential hive files.
            let reg_dump = image == "reg.exe"
                && cmdline.contains("save")
                && (cmdline.contains("sam") || cmdline.contains("system")
                    || cmdline.contains("security") || cmdline.contains("ntds"));

            // T1047 WMI execution — wmic process call create or remote /node:, or wmiprvse spawning children.
            // wmiprvse.exe = WMI Provider Host; anything it spawns is a WMI-executed payload.
            // CAR-2016-03-002: log_source=process/create, mutable=command_line, wmic /node: <host>
            let wmi_exec =
                (image == "wmic.exe"
                    && (cmdline.contains("/node:")                        // remote WMI (T1021/T1047 lateral)
                        || (cmdline.contains("process") && cmdline.contains("call"))))
                || (parent == "wmiprvse.exe"
                    && !matches!(image.as_str(), "conhost.exe" | "wbemcons.exe" | "wbemcore.exe"));

            // T1021.003 DCOM lateral movement — mmc.exe or dllhost.exe spawning shells.
            // These COM server processes should never spawn interactive child processes.
            let dcom_exec = matches!(parent.as_str(), "mmc.exe" | "dllhost.exe")
                && matches!(image.as_str(),
                    "cmd.exe" | "powershell.exe" | "wscript.exe" | "cscript.exe"
                    | "mshta.exe" | "rundll32.exe" | "regsvr32.exe");

            // T1218.011 Signed binary proxy: rundll32 with unusual / non-standard targets.
            let rundll32_abuse = image == "rundll32.exe"
                && (cmdline.contains("javascript:")
                    || cmdline.contains("mshtml")
                    || cmdline.contains("pcwutl.dll")
                    || cmdline.contains("ieadvpack")
                    || cmdline.contains("setupapi")
                    || cmdline.contains("shdocvw")
                    || cmdline.contains("control_rundll")
                    || cmdline.contains(",shellexec_rundll")
                    || (cmdline.contains(".dll,") && !cmdline.contains("\\system32\\")));

            // T1569.002 Service execution — services.exe spawning non-service processes.
            let service_spawn = parent == "services.exe"
                && !matches!(image.as_str(),
                    "svchost.exe" | "spoolsv.exe" | "lsass.exe" | "lsm.exe" | "csrss.exe"
                    | "wininit.exe" | "taskhost.exe" | "taskhostw.exe" | "msdtc.exe"
                    | "vds.exe" | "wbengine.exe" | "dfsr.exe" | "dfsrs.exe" | "dns.exe"
                    | "ismserv.exe" | "ntfrs.exe" | "netlogon.exe");

            // T1543.003 Modify existing service binary path (sc config binPath=).
            let modify_service = image == "sc.exe"
                && cmdline.contains("config")
                && (cmdline.contains("powershell") || cmdline.contains("cmd.exe")
                    || cmdline.contains("\\temp\\") || cmdline.contains("\\appdata\\")
                    || cmdline.contains("rundll32") || cmdline.contains("mshta"));

            // T1021.002 Net use for lateral movement / SMB share access.
            let net_lateral = (image == "net.exe" || image == "net1.exe")
                && (cmdline.contains(" use ") || cmdline.contains(" session "));

            // T1134 Token impersonation — runas abuse or PowerShell token manipulation.
            let token_abuse = image == "runas.exe"
                || (image == "powershell.exe" && (
                    cmdline.contains("seimpersonateprivilege")
                    || cmdline.contains("[system.security.principal]")
                    || cmdline.contains("windowsidentity::impersonate")
                    || cmdline.contains("impersonateloggedonuser")
                    || cmdline.contains("duplicatetoken")));

            // T1550 Pass-the-Hash / DCSync / Overpass-the-Hash via Mimikatz syntax.
            let pth_attack = cmdline.contains("sekurlsa::pth")
                || cmdline.contains("sekurlsa::wdigest")
                || cmdline.contains("lsadump::dcsync")
                || cmdline.contains("lsadump::sam")
                || cmdline.contains("lsadump::lsa")
                || cmdline.contains("kerberos::ptt")
                || cmdline.contains("token::elevate")
                || cmdline.contains("privilege::debug");

            // T1059.005/006 Visual Basic / Macro execution chains via mshta or wscript.
            let vbs_exec = matches!(parent.as_str(), "mshta.exe" | "wscript.exe" | "cscript.exe")
                && matches!(image.as_str(),
                    "cmd.exe" | "powershell.exe" | "net.exe" | "whoami.exe"
                    | "ipconfig.exe" | "systeminfo.exe" | "reg.exe" | "wmic.exe");

            // T1218.005 Mshta with URL or inline script (not just a local file).
            let mshta_abuse = image == "mshta.exe"
                && (cmdline.contains("http") || cmdline.contains("javascript:")
                    || cmdline.contains("vbscript:") || cmdline.contains("about:"));

            // CAR-2020-09-003: fltmc.exe unloading a minifilter driver disables AV/EDR (T1562.006).
            // log_source=process/create, mutable=command_line (exe="fltmc.exe", cmd contains "unload")
            let fltmc_unload = image == "fltmc.exe" && cmdline.contains("unload");

            // CAR-2021-02-002: GetSystem via named pipe echo (Meterpreter/CS privilege escalation T1134).
            // log_source=process/create, mutable=command_line (echo + \pipe\)
            let getsystem_pipe = cmdline.contains("echo")
                && (cmdline.contains("\\pipe\\") || cmdline.contains("\\\\.\\pipe"));

            // CAR-2020-11-004: Process parent-chain mismatch — indicates hollowing / injection (T1055.012).
            // log_source=process/create, mutable=parent_exe (deviation from expected Windows lineage)
            // Guard: skip events with missing/empty parent (early-boot processes have no recorded parent).
            let hollow_parent = !parent.is_empty() && (
                (image == "smss.exe"     && !matches!(parent.as_str(), "smss.exe" | "system"))
                || (image == "csrss.exe" && !matches!(parent.as_str(), "smss.exe" | "svchost.exe"))
                || (image == "wininit.exe" && parent != "smss.exe")
                || (image == "winlogon.exe" && parent != "smss.exe")
                || (image == "lsass.exe" && !matches!(parent.as_str(), "wininit.exe" | "winlogon.exe"))
                || (image == "services.exe" && parent != "wininit.exe")
                || (image == "spoolsv.exe" && parent != "services.exe")
                || (image == "userinit.exe"
                    && !matches!(parent.as_str(), "dwm.exe" | "winlogon.exe" | "explorer.exe")));

            // CAR-2021-04-001: Common Windows process running from non-standard path (T1036.005).
            // log_source=process/create, mutable=image_path (masquerading via wrong directory)
            // Guard: only check when we have a full path (contains '\'); bare basenames match nothing.
            let masquerade = img_path.contains('\\')
                && matches!(image.as_str(),
                    "svchost.exe" | "smss.exe" | "csrss.exe" | "wininit.exe" | "winlogon.exe"
                    | "lsass.exe" | "services.exe" | "spoolsv.exe" | "taskhost.exe" | "explorer.exe")
                && !img_path.contains("\\windows\\system32\\")
                && !img_path.contains("\\windows\\syswow64\\");

            // CAR-2019-07-002: procdump targeting lsass (T1003.001).
            // log_source=process/create, mutable=command_line + exe
            let procdump_lsass = image.starts_with("procdump")
                && (cmdline.contains("lsass") || cmdline.contains("-ma"));

            // CAR-2021-05-008: certutil -exportPFX to steal certificates (T1606.002).
            // log_source=process/create, mutable=command_line
            let certutil_cert = image == "certutil.exe" && cmdline.contains("-exportpfx");

            // T1218.007 Msiexec loading remote MSI (LOLBin staging / supply-chain execution).
            // log_source=process/create, mutable=command_line (http or UNC remote path — not /quiet alone)
            let msiexec_remote = image == "msiexec.exe"
                && (cmdline.contains("http://") || cmdline.contains("https://")
                    || (cmdline.contains("/i") && cmdline.contains("\\\\")));

            // T1134 /savecred stores credentials for later reuse (runas /savecred).
            // log_source=process/create, mutable=command_line
            let savecred = cmdline.contains("/savecred");

            attack_tool
                || office_to_script
                || lolbin_to_script
                || script_to_recon
                || system_spawns_shell
                || encoded_ps
                || certutil_abuse
                || script_from_staging
                || staged_exec
                || uac_bypass
                || impair_defenses
                || clear_logs
                || inhibit_recovery
                || create_account
                || malicious_service
                || modify_service
                || schtask_attack
                || winrm_exec
                || wsl_exec
                || reg_dump
                || wmi_exec
                || dcom_exec
                || rundll32_abuse
                || service_spawn
                || net_lateral
                || token_abuse
                || pth_attack
                || vbs_exec
                || mshta_abuse
                || fltmc_unload
                || getsystem_pipe
                || hollow_parent
                || masquerade
                || procdump_lsass
                || certutil_cert
                || msiexec_remote
                || savecred
        }

        // ── EID 2 — FileCreationTimeChanged: timestomping (T1070.006) ───────────
        // Any timestamp manipulation is an anti-forensic indicator.
        2 => true,

        // ── EID 3 — NetworkConnect ───────────────────────────────────────────────
        3 => {
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();
            let initiated = v["Initiated"].as_str().unwrap_or("") == "true";
            let dest_port: u16 = v["DestinationPort"].as_str().unwrap_or("0")
                .parse().unwrap_or(0);
            let dest_ip = v["DestinationIp"].as_str().unwrap_or("");

            // LOLBins making outbound connections — staging / C2 (T1105 / T1071).
            let lolbin = matches!(image.as_str(),
                "mshta.exe" | "regsvr32.exe" | "wscript.exe" | "cscript.exe"
                | "certutil.exe" | "bitsadmin.exe" | "wmic.exe" | "rundll32.exe"
                | "odbcconf.exe" | "cmstp.exe" | "msbuild.exe" | "installutil.exe"
                | "regasm.exe" | "regsvcs.exe" | "ieexec.exe" | "xwizard.exe"
                | "dnscmd.exe" | "msiexec.exe" | "esentutl.exe" | "expand.exe"
                | "extrac32.exe" | "hh.exe" | "makecab.exe" | "msdt.exe"
                | "pcalua.exe" | "rpcping.exe" | "verclsid.exe" | "wab.exe"
                | "wusa.exe" | "presentationhost.exe" | "diskshadow.exe"
                | "bash.exe" | "forfiles.exe" | "appvlp.exe"
            );

            // PowerShell / cmd making non-standard-port outbound (C2 beaconing).
            let ps_nonstandard = initiated
                && matches!(image.as_str(), "powershell.exe" | "cmd.exe")
                && !matches!(dest_port, 80 | 443 | 8080 | 8443 | 53 | 25 | 587 | 465);

            // Any scripting host connecting to internal IPs (lateral movement staging).
            let internal_lateral = initiated
                && matches!(image.as_str(),
                    "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                    | "mshta.exe" | "python.exe")
                && (dest_ip.starts_with("10.")
                    || dest_ip.starts_with("172.16.") || dest_ip.starts_with("172.17.")
                    || dest_ip.starts_with("172.18.") || dest_ip.starts_with("172.19.")
                    || dest_ip.starts_with("172.2")   || dest_ip.starts_with("172.3")
                    || dest_ip.starts_with("192.168."));

            // T1021.001 RDP — unexpected process connecting on 3389 (T1021.001).
            let rdp_nonstandard = initiated && dest_port == 3389
                && !matches!(image.as_str(), "mstsc.exe" | "msrdcw.exe");

            // T1021.002 SMB lateral movement — scripting host on port 445 (T1021.002).
            let smb_lateral = initiated && dest_port == 445
                && matches!(image.as_str(),
                    "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                    | "mshta.exe" | "python.exe" | "wmic.exe");

            // T1021.006 WinRM — unexpected process on WinRM ports 5985/5986.
            let winrm_lateral = initiated
                && matches!(dest_port, 5985 | 5986)
                && !matches!(image.as_str(), "wsmprovhost.exe" | "winrm.cmd");

            // T1071.001 LOLBins or scripting hosts on HTTP/HTTPS (staging/C2).
            // These are attack even on standard ports (certutil, bitsadmin, etc.).
            let http_lolbin = initiated
                && matches!(dest_port, 80 | 443 | 8080 | 8443)
                && matches!(image.as_str(),
                    "wscript.exe" | "cscript.exe" | "mshta.exe" | "rundll32.exe"
                    | "regsvr32.exe" | "msbuild.exe" | "installutil.exe" | "regasm.exe"
                    | "certutil.exe" | "bitsadmin.exe" | "wmic.exe" | "cmstp.exe");

            // T1021.001 Unexpected RDP client process (not just non-mstsc, but scripting hosts).
            // Also covers: psexec making RDP connections.
            let psexec_smb = initiated && dest_port == 445
                && (image.contains("psexec") || image.contains("paexec") || image == "sc.exe");

            (lolbin && initiated) || ps_nonstandard || internal_lateral
                || rdp_nonstandard || smb_lateral || winrm_lateral
                || http_lolbin || psexec_smb
        }

        // ── EID 6 — DriverLoad: BYOVD / unsigned driver (T1068 / T1014) ─────────
        6 => {
            let img = v["ImageLoaded"].as_str().unwrap_or("").to_lowercase();
            let sig_status = v["SignatureStatus"].as_str().unwrap_or("");
            let signed = v["Signed"].as_str().unwrap_or("false");

            // Known BYOVD (Bring Your Own Vulnerable Driver) names.
            let byovd = img.contains("dbutil_2_3") || img.contains("mhyprot2")
                || img.contains("gdrv.sys") || img.contains("asmmap")
                || img.contains("rtcore64") || img.contains("zamguard")
                || img.contains("ene_io")   || img.contains("physmem")
                || img.contains("wfp_test") || img.contains("procexp");

            // Unsigned driver or loaded from writable path (staging rootkit).
            let unsigned = signed == "false" || sig_status != "Valid";
            let suspicious_path = img.contains("\\temp\\")
                || img.contains("\\appdata\\") || img.contains("\\users\\public\\")
                || img.contains("\\programdata\\") || img.contains("\\downloads\\");

            byovd || (unsigned && suspicious_path)
        }

        // ── EID 7 — ImageLoad: DLL side-loading / hijacking (T1574) ─────────────
        7 => {
            let img = v["ImageLoaded"].as_str().unwrap_or("").to_lowercase();
            let sig_status = v["SignatureStatus"].as_str().unwrap_or("");
            let signed = v["Signed"].as_str().unwrap_or("false");
            let process = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();

            // Known attack / reflective-injection DLL names.
            let known_bad = img.contains("mimilib") || img.contains("kiwi.dll")
                || img.contains("sekurlsa")  || img.contains("wceaux.dll")
                || img.contains("reflective") || img.contains("payload.dll")
                || img.contains("beacon.dll") || img.contains("inject.dll")
                || img.contains("cobaltstrike");

            // DLL loaded from a staging / writable path (side-load / dropper).
            let staged_dll = (img.ends_with(".dll") || img.ends_with(".ocx"))
                && (img.contains("\\temp\\")
                    || img.contains("\\appdata\\local\\temp")
                    || img.contains("\\users\\public\\")
                    || img.contains("\\programdata\\")
                    || img.contains("\\downloads\\"));

            // Unsigned DLL loaded into a high-value process.
            let unsigned_in_sensitive = (signed == "false" || sig_status != "Valid")
                && matches!(process.as_str(),
                    "lsass.exe" | "winlogon.exe" | "csrss.exe" | "wininit.exe"
                    | "services.exe" | "svchost.exe");

            // T1574.001 DLL hijack via DCOM process (mmc.exe, dllhost.exe, excel.exe).
            // DCOM lateral movement loads hijacked DLLs into DCOM server processes.
            let dcom_dll_hijack = matches!(process.as_str(),
                "dllhost.exe" | "mmc.exe" | "excel.exe" | "outlook.exe" | "svchost.exe")
                && (signed == "false" || sig_status != "Valid")
                && !img.contains("\\windows\\system32\\")
                && !img.contains("\\windows\\syswow64\\")
                && !img.contains("\\program files\\");

            // T1218.010 Regsvr32 loading remotely-fetched DLLs.
            let regsvr_dll = process == "regsvr32.exe"
                && (staged_dll || signed == "false");

            known_bad || staged_dll || unsigned_in_sensitive || dcom_dll_hijack || regsvr_dll
        }

        // ── EID 8 — CreateRemoteThread: process injection (T1055) ────────────────
        8 => true,

        // ── EID 9 — RawAccessRead: credential/volume-shadow theft (T1003) ────────
        9 => true,

        // ── EID 10 — ProcessAccess: credential dump / token theft ────────────────
        10 => {
            let target = basename(v["TargetImage"].as_str().unwrap_or("")).to_lowercase();
            let access_str = v["GrantedAccess"].as_str().unwrap_or("0x0");
            let access = u64::from_str_radix(access_str.trim_start_matches("0x"), 16)
                .unwrap_or(0);

            // Any access to lsass is suspicious (T1003.001).
            if target == "lsass.exe" { return true; }

            // Memory-read/write/dup-handle on other credential-store processes.
            let sensitive_target = matches!(target.as_str(),
                "winlogon.exe" | "csrss.exe" | "wininit.exe" | "services.exe"
                | "explorer.exe" | "svchost.exe" | "spoolsv.exe"
                | "taskhost.exe" | "taskhostw.exe" | "vaultcvc.exe"
                | "msmpeng.exe" | "msseces.exe"
            );
            // VM_READ(0x10)|VM_WRITE(0x20)|DUP_HANDLE(0x40) or broad PROCESS_ALL_ACCESS.
            // Also: 0x1F0FFF = PROCESS_ALL_ACCESS (DET0363), 0x1410 = token+query ops (T1134).
            let suspicious_access = access & 0x70 != 0
                || access >= 0x1f0000
                || access == 0x1f0fff
                || access == 0x1410
                || access == 0x1fffff;

            // T1134 Token impersonation via handle duplication on winlogon/services.exe.
            let token_steal = access & 0x40 != 0   // PROCESS_DUP_HANDLE
                && matches!(target.as_str(), "winlogon.exe" | "services.exe");

            (sensitive_target && suspicious_access) || token_steal
        }

        // ── EID 11 — FileCreate: dropper / payload staging ───────────────────────
        11 => {
            let fname = v["TargetFilename"].as_str().unwrap_or("").to_lowercase();
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();

            // Broader set of scripting / LOLBin writers.
            let scripting = matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "python.exe" | "wmic.exe" | "regsvr32.exe"
                | "rundll32.exe" | "msbuild.exe" | "cmstp.exe" | "certutil.exe"
                | "bitsadmin.exe" | "excel.exe" | "winword.exe" | "outlook.exe"
                | "onenote.exe" | "msaccess.exe" | "curl.exe" | "expand.exe"
            );

            let staging = fname.contains("\\temp\\")
                || fname.contains("\\appdata\\local\\temp")
                || fname.contains("\\appdata\\roaming\\")
                || fname.contains("\\public\\")
                || fname.contains("\\programdata\\")
                || fname.contains("\\downloads\\")
                || fname.contains("\\users\\default\\");

            // Executable / script extensions (T1027, T1105).
            let executable = fname.ends_with(".exe") || fname.ends_with(".dll")
                || fname.ends_with(".ps1")  || fname.ends_with(".bat")
                || fname.ends_with(".vbs")  || fname.ends_with(".js")
                || fname.ends_with(".hta")  || fname.ends_with(".jse")
                || fname.ends_with(".wsf")  || fname.ends_with(".vbe")
                || fname.ends_with(".scr")  || fname.ends_with(".pif")
                || fname.ends_with(".cpl")  || fname.ends_with(".inf")
                || fname.ends_with(".sys")  || fname.ends_with(".ocx")
                || fname.ends_with(".xll"); // Excel add-in

            // Memory-dump artefacts (credential theft).
            let dump = (fname.ends_with(".dmp") || fname.ends_with(".mdmp"))
                || (fname.ends_with(".bin")
                    && (fname.contains("lsass") || fname.contains("ntds") || fname.contains("sam")));

            // CAR-2020-09-001: Task file created in Tasks directory by a non-scheduler process (T1053.005).
            // log_source=file/create, mutable=file_path (path in \Tasks\) + image_path (not svchost)
            let task_drop = (fname.contains("\\windows\\system32\\tasks\\")
                || fname.contains("\\windows\\tasks\\"))
                && !matches!(image.as_str(), "svchost.exe" | "taskeng.exe" | "taskhostw.exe");

            // CAR-2021-05-002: Batch/cmd script dropped into System32 (T1204.002 / T1059.003).
            // log_source=file/create, mutable=file_path (in system32) + extension (.bat/.cmd)
            let system32_script = fname.contains("\\windows\\system32\\")
                && (fname.ends_with(".bat") || fname.ends_with(".cmd") || fname.ends_with(".ps1"));

            scripting && (staging || executable) || dump || task_drop || system32_script
        }

        // ── EID 12/13 — RegistryEvent: persistence / defence-evasion keys ────────
        12 | 13 => {
            let obj = v["TargetObject"].as_str().unwrap_or("").to_lowercase();
            obj.contains("\\run\\")
                || obj.contains("\\runonce\\")
                || obj.contains("\\runonceex\\")
                || obj.contains("userinitmprlogonscript")
                || obj.contains("\\currentversion\\winlogon")
                || obj.contains("\\currentversion\\windows\\load")
                || obj.contains("\\policies\\explorer\\run")
                || obj.contains("\\securityproviders\\")
                || obj.contains("minint")
                || (obj.contains("\\services\\") && obj.contains("\\start"))
                || (obj.contains("\\eventlog\\") && obj.contains("\\start"))
                || obj.contains("\\image file execution options\\") // debugger hijack T1546.012
                || obj.contains("\\appcompatflags\\")               // shim T1546.011
                || obj.contains("\\lsa\\")                          // LSA provider T1547.002
                || obj.contains("\\safeboot\\")                     // safeboot bypass T1562
                || obj.contains("\\sessionmanager\\")               // boot-execute T1547.006
                || obj.contains("\\bootexecute")
                || obj.contains("\\knowndlls\\")                    // DLL hijack T1574.001
                || obj.contains("\\command processor\\autorun")     // cmd.exe autorun
                || obj.contains("\\active setup\\")                 // active setup T1547.014
                || obj.contains("\\browser helper objects\\")       // BHO T1176
                || obj.contains("userinit")
                || obj.contains("\\netsh\\")                        // netsh helper T1546.007
                || obj.contains("\\print\\monitors\\")              // print monitor T1547.010
                || obj.contains("\\print\\providers\\")
                || obj.contains("\\terminal server\\")              // RDP backdoor
                || obj.contains("\\wbem\\")                         // WMI persistence T1546.003
                || obj.contains("\\environment\\comspec")           // ComSpec hijack
                || obj.contains("\\policies\\system\\enablelua")    // UAC bypass
                || obj.contains("\\disableantispy")
                || obj.contains("\\audit\\")                        // audit policy tampering
                // T1548.002 UAC bypass via registry hijack
                || obj.contains("\\software\\classes\\ms-settings\\") // fodhelper UAC bypass
                || obj.contains("\\software\\classes\\mscfile\\")      // eventvwr UAC bypass
                || obj.contains("\\software\\classes\\exefile\\")      // file association hijack
                || obj.contains("\\software\\classes\\clsid\\")        // COM hijack T1546.015
                // T1562.001 Disable Windows Defender via registry
                || obj.contains("\\windows defender\\")
                || obj.contains("\\disablerealtimemonitoring")
                || obj.contains("\\disablebehaviormonitoring")
                || obj.contains("\\disableioavprotection")
                || obj.contains("\\disableantispyware")
                // T1543.003 Service binary path → staging location
                || (obj.contains("\\services\\") && obj.contains("imagepath")
                    && {
                        let detail = v["Details"].as_str().unwrap_or("").to_lowercase();
                        detail.contains("\\temp\\") || detail.contains("\\appdata\\")
                            || detail.contains("\\public\\") || detail.contains("\\programdata\\")
                            || detail.contains("powershell") || detail.contains("cmd.exe")
                            || detail.contains("rundll32") || detail.contains("mshta")
                    })
                // T1112 Disable Windows Error Reporting / crash dumps (anti-forensics)
                || obj.contains("\\windows error reporting\\disabled")
                || obj.contains("\\crashcontrol\\crashdumpenabled")
                // CAR-2021-11-001: SafeDllSearchMode=0 enables DLL hijacking via search-order (T1574.001).
                // log_source=registry/value_edit, mutable=value (SafeDllSearchMode=0 in Session Manager)
                || obj.contains("safedllsearchmode")
                // CAR-2022-03-001: EventLog service key tampering to disable logging (T1562.002).
                // log_source=registry/value_edit, mutable=value (Start/File/MaxSize keys under EventLog)
                || (obj.contains("\\eventlog\\")
                    && (obj.ends_with("\\start") || obj.ends_with("\\file")
                        || obj.ends_with("\\maxsize") || obj.ends_with("\\autobackuplogfiles")))
                // T1562.002 MiniNt key disables Security event log on next boot.
                || obj.contains("\\minint")
                // CAR-2021-12-002: Common Startup folder redirection (T1547.001).
                // log_source=registry/add, mutable=key (Common Startup path)
                || obj.contains("common startup")
        }

        // ── EID 14 — RegistryKeyValueRename: hiding persistence keys ─────────────
        14 => true,

        // ── EID 15 — FileCreateStreamHash: ADS (alternate data stream) ───────────
        // Zone.Identifier is benign (MOTW); anything else = hidden payload T1564.004.
        15 => {
            let fname = v["TargetFilename"].as_str().unwrap_or("").to_lowercase();
            !fname.ends_with(":zone.identifier")
        }

        // ── EID 16 — Sysmon config change: defence evasion (T1562.001) ───────────
        16 => true,

        // ── EID 17/18 — PipeEvent: C2 named-pipe comms (T1071 / T1559) ───────────
        17 | 18 => {
            let pipe = v["PipeName"].as_str().unwrap_or("").to_lowercase();
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();

            // Known Cobalt Strike / Metasploit / Covenant pipe patterns.
            let known_c2_pipe = pipe.contains("postex_")  || pipe.contains("msse-")
                || pipe.contains("status_")   || pipe.contains("msagent_")
                || pipe.contains("ntsvcs-")   || pipe.contains("46a9e56")
                || pipe.contains("583da945")  || pipe.contains("wkssvc-")
                || pipe.contains("isapi_")    || pipe.contains("dce_pipe")
                || pipe.contains("gruntstager") || pipe.contains("interprocess_")
                || pipe.contains("atctl")     || pipe.contains("gecko_channel");

            // Pipe created by a scripting / LOLBin process.
            let suspicious_creator = matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "rundll32.exe" | "regsvr32.exe" | "msbuild.exe");

            known_c2_pipe || suspicious_creator
        }

        // ── EID 19/20/21 — WMI event subscriptions: persistence / LM ─────────────
        19 | 20 | 21 => true,

        // ── EID 22 — DNSEvent: C2 beaconing / DNS tunnelling (T1071.004) ─────────
        22 => {
            let query = v["QueryName"].as_str().unwrap_or("").to_lowercase();
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();

            // LOLBins or scripting hosts making DNS queries.
            let suspicious_process = matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe"
                | "mshta.exe" | "rundll32.exe" | "regsvr32.exe" | "certutil.exe"
                | "bitsadmin.exe" | "wmic.exe" | "msbuild.exe" | "installutil.exe"
            );

            // Very long subdomain label or overall query → DNS tunnelling.
            let dns_tunnel = query.len() > 60
                || query.split('.').any(|part| part.len() > 40);

            suspicious_process || dns_tunnel
        }

        // ── EID 23/26 — FileDelete: evidence destruction (T1070.004) ─────────────
        23 | 26 => {
            let fname = v["TargetFilename"].as_str().unwrap_or("").to_lowercase();
            let image = basename(v["Image"].as_str().unwrap_or("")).to_lowercase();

            // Event-log deletion (T1070.001).
            let log_delete = fname.contains("\\winevt\\")
                || fname.contains("\\windows\\logs\\")
                || fname.ends_with(".evtx");

            // Prefetch deletion (anti-forensics).
            let prefetch = fname.contains("\\prefetch\\") && fname.ends_with(".pf");

            // Scripting host deleting executables or dumps (cleanup).
            let scripted = matches!(image.as_str(),
                "powershell.exe" | "cmd.exe" | "wscript.exe" | "cscript.exe")
                && (fname.ends_with(".exe") || fname.ends_with(".dll")
                    || fname.ends_with(".dmp") || fname.ends_with(".evtx")
                    || fname.ends_with(".ps1") || fname.ends_with(".bat"));

            log_delete || prefetch || scripted
        }

        // ── EID 25 — ProcessTampering: hollowing / herpaderping (T1055) ──────────
        25 => true,

        _ => false,
    }
}
