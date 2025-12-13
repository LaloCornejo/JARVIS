use std::path::Path;
use tokio::fs;

#[derive(Debug, serde::Serialize)]
pub struct ToolResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub async fn read_file(path: &str, max_lines: usize) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(path);
    if !path.exists() {
        return Ok(ToolResult {
            exit_code: Some(1),
            stdout: "".to_string(),
            stderr: "File does not exist".to_string(),
        });
    }
    if !path.is_file() {
        return Ok(ToolResult {
            exit_code: Some(1),
            stdout: "".to_string(),
            stderr: "Path is not a file".to_string(),
        });
    }
    let content = fs::read_to_string(path).await?;
    let lines: Vec<&str> = content.lines().collect();
    let limited_lines = &lines[..lines.len().min(max_lines)];
    let stdout = limited_lines.join("\n");
    Ok(ToolResult {
        exit_code: Some(0),
        stdout,
        stderr: "".to_string(),
    })
}

pub async fn write_file(path: &str, content: &str, append: bool) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    if append {
        let mut file = fs::OpenOptions::new().append(true).create(true).open(path).await?;
        use tokio::io::AsyncWriteExt;
        file.write_all(content.as_bytes()).await?;
    } else {
        fs::write(path, content).await?;
    }
    Ok(ToolResult {
        exit_code: Some(0),
        stdout: "".to_string(),
        stderr: "".to_string(),
    })
}

pub async fn list_directory(path: &str, pattern: Option<&str>) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(path);
    if !path.exists() || !path.is_dir() {
        return Ok(ToolResult {
            exit_code: Some(1),
            stdout: "".to_string(),
            stderr: "Directory does not exist".to_string(),
        });
    }
    let mut entries = Vec::new();
    if let Some(pat) = pattern {
        let pattern_path = path.join(pat);
        for entry in glob::glob(&pattern_path.to_string_lossy())? {
            let entry = entry?;
            if let Some(name) = entry.file_name() {
                entries.push(name.to_string_lossy().to_string());
            }
        }
    } else {
        let mut dir = fs::read_dir(path).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            entries.push(name);
        }
    }
    entries.sort();
    let limited = entries.into_iter().take(100).collect::<Vec<_>>();
    let stdout = limited.join("\n");
    Ok(ToolResult {
        exit_code: Some(0),
        stdout,
        stderr: "".to_string(),
    })
}

pub async fn search_files(path: &str, pattern: &str) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let pattern_path = Path::new(path).join("**").join(pattern);
    let mut results = Vec::new();
    for entry in glob::glob(&pattern_path.to_string_lossy())? {
        let entry = entry?;
        if let Ok(rel) = entry.strip_prefix(path) {
            results.push(rel.to_string_lossy().to_string());
        }
    }
    results.sort();
    let limited = results.into_iter().take(100).collect::<Vec<_>>();
    let stdout = limited.join("\n");
    Ok(ToolResult {
        exit_code: Some(0),
        stdout,
        stderr: "".to_string(),
    })
}

pub async fn file_info(path: &str) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(path);
    let metadata = fs::metadata(path).await?;
    let file_type = if metadata.is_file() { "file" } else if metadata.is_dir() { "directory" } else { "other" };
    let size = metadata.len();
    let modified = metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs();
    let stdout = format!("type: {}\nsize: {}\nmodified: {}", file_type, size, modified);
    Ok(ToolResult {
        exit_code: Some(0),
        stdout,
        stderr: "".to_string(),
    })
}