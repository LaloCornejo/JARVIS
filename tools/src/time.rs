use chrono;
use chrono_tz;
use serde_json::json;

#[derive(Debug, serde::Serialize)]
pub struct ToolResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub fn execute(timezone: &str) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let tz: chrono_tz::Tz = timezone.parse().map_err(|_| format!("Invalid timezone: {}", timezone))?;
    let now = chrono::Utc::now().with_timezone(&tz);
    
    // Create a structured JSON response
    let time_data = json!({
        "datetime": now.to_rfc3339(),
        "timestamp": now.timestamp(),
        "timezone": timezone,
        "formatted": now.format("%Y-%m-%d %H:%M:%S %Z").to_string(),
        "date": now.format("%Y-%m-%d").to_string(),
        "time": now.format("%H:%M:%S").to_string(),
    });
    
    Ok(ToolResult {
        exit_code: Some(0),
        stdout: time_data.to_string(),
        stderr: "".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_time() {
        let result = execute("UTC");
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(!res.stdout.is_empty());
    }
    
    #[test]
    fn test_get_time_mexico_city() {
        let result = execute("America/Mexico_City");
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(!res.stdout.is_empty());
    }
}