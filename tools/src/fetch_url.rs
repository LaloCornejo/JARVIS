use reqwest;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolResult {
    pub url: String,
    pub status: u16,
    pub content: String,
}

pub async fn execute(url: &str) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("JARVIS/1.0")
        .build()?;

    let response = client.get(url).send().await?;
    let status = response.status().as_u16();
    response.error_for_status_ref()?;

    let content = response.text().await?;
    // Limit to 10KB
    let limited_content = if content.len() > 10240 {
        content[..10240].to_string()
    } else {
        content
    };

    Ok(ToolResult {
        url: url.to_string(),
        status,
        content: limited_content,
    })
}