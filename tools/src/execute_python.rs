use std::fs;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use regex::Regex;
use rand::Rng;

lazy_static::lazy_static! {
    static ref BLOCKED_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"os\.system").unwrap(),
        Regex::new(r"subprocess\.").unwrap(),
        Regex::new(r"eval\(").unwrap(),
        Regex::new(r"exec\(").unwrap(),
        Regex::new(r"__import__\(").unwrap(),
        Regex::new(r"import\s+os").unwrap(),
        Regex::new(r"import\s+subprocess").unwrap(),
        Regex::new(r"import\s+sys").unwrap(),
        Regex::new(r"sys\.exit").unwrap(),
        Regex::new(r"open\([^)]*\bw\b").unwrap(),  // writing files
        Regex::new(r"rm\s+-rf").unwrap(),
        Regex::new(r"sudo").unwrap(),
        Regex::new(r"su").unwrap(),
    ];
}

#[derive(Debug, serde::Serialize)]
pub struct ToolResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub async fn execute(code: &str, timeout_secs: u64) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    // Validate code
    for pattern in BLOCKED_PATTERNS.iter() {
        if pattern.is_match(code) {
            return Err(format!("Code contains blocked pattern: {}", pattern.as_str()).into());
        }
    }

    // Create temp file
    let temp_dir = std::env::temp_dir();
    let random_suffix: u32 = rand::thread_rng().gen();
    let temp_file = temp_dir.join(format!("jarvis_exec_{}.py", random_suffix));

    // Write code to temp file
    fs::write(&temp_file, code)?;

    // Execute with timeout
    let python_cmd = if cfg!(target_os = "windows") { "py" } else { "python" };
    let result = timeout(Duration::from_secs(timeout_secs), async {
        let output = Command::new(python_cmd)
            .arg(&temp_file)
            .output()
            .await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(output)
    }).await??;

    // Clean up temp file
    let _ = fs::remove_file(&temp_file);

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();

    Ok(ToolResult {
        exit_code: result.status.code(),
        stdout,
        stderr,
    })
}