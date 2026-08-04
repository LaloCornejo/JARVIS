use std::collections::HashSet;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use regex::Regex;

lazy_static::lazy_static! {
    static ref ALLOWED_COMMANDS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        set.insert("ls");
        set.insert("git");
        set.insert("cat");
        set.insert("grep");
        set.insert("find");
        set.insert("ps");
        set.insert("top");
        set.insert("df");
        set.insert("du");
        set.insert("free");
        set.insert("uptime");
        set.insert("whoami");
        set.insert("pwd");
        set.insert("echo");
        set.insert("date");
        set.insert("cal");
        set.insert("wc");
        set.insert("head");
        set.insert("tail");
        set.insert("sort");
        set.insert("uniq");
        set.insert("cut");
        set.insert("awk");
        set.insert("sed");
        set.insert("chmod");
        set.insert("chown");
        set.insert("mkdir");
        set.insert("rmdir");
        set.insert("touch");
        set.insert("cp");
        set.insert("mv");
        set.insert("ln");
        set.insert("tar");
        set.insert("gzip");
        set.insert("gunzip");
        set.insert("zip");
        set.insert("unzip");
        set.insert("curl");
        set.insert("wget");
        set.insert("ping");
        set.insert("traceroute");
        set.insert("nslookup");
        set.insert("dig");
        set.insert("ssh");
        set.insert("scp");
        set.insert("rsync");
        set.insert("docker");
        set.insert("kubectl");
        set.insert("npm");
        set.insert("yarn");
        set.insert("pip");
        set.insert("python");
        set.insert("node");
        set.insert("cargo");
        set.insert("rustc");
        set
    };

    static ref BLOCKED_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"rm\s+-rf").unwrap(),
        Regex::new(r"sudo").unwrap(),
        Regex::new(r"su").unwrap(),
        Regex::new(r"passwd").unwrap(),
        Regex::new(r"chmod\s+777").unwrap(),
        Regex::new(r"dd\s+if=").unwrap(),
        Regex::new(r"mkfs").unwrap(),
        Regex::new(r"fdisk").unwrap(),
        Regex::new(r"parted").unwrap(),
        Regex::new(r"mount").unwrap(),
        Regex::new(r"umount").unwrap(),
        Regex::new(r"systemctl").unwrap(),
        Regex::new(r"service").unwrap(),
        Regex::new(r"killall").unwrap(),
        Regex::new(r"pkill").unwrap(),
        Regex::new(r"reboot").unwrap(),
        Regex::new(r"shutdown").unwrap(),
        Regex::new(r"halt").unwrap(),
        Regex::new(r"poweroff").unwrap(),
        Regex::new(r"eval").unwrap(),
        Regex::new(r"exec").unwrap(),
        Regex::new(r"source").unwrap(),
        Regex::new(r"\.\s+/").unwrap(),
        Regex::new(r">\s*/").unwrap(),
        Regex::new(r">>\s*/").unwrap(),
    ];
}

#[derive(Debug, serde::Serialize)]
pub struct ToolResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub async fn execute(command: &str, timeout_secs: u64) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    // Split command to get the base command
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty command".into());
    }
    let base_cmd = parts[0];

    // Check if command is allowed
    if !ALLOWED_COMMANDS.contains(base_cmd) {
        return Err(format!("Command '{}' is not allowed", base_cmd).into());
    }

    // Check for blocked patterns
    for pattern in BLOCKED_PATTERNS.iter() {
        if pattern.is_match(command) {
            return Err(format!("Command contains blocked pattern: {}", pattern.as_str()).into());
        }
    }

    // Execute with timeout
    let result = timeout(Duration::from_secs(timeout_secs), async {
        let output = Command::new(base_cmd)
            .args(&parts[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(output)
    }).await??;

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();

    Ok(ToolResult {
        exit_code: result.status.code(),
        stdout,
        stderr,
    })
}