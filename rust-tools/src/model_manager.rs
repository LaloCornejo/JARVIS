use clap::{Parser, Subcommand};
use reqwest::Client;
use serde::Deserialize;
use std::error::Error;

/// JARVIS Model Manager - Manage LLM models
#[derive(Parser)]
#[clap(name = "jarvis-model-manager", version = "0.1.0")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available models
    List,
    
    /// Show model details
    Info {
        /// Model name
        model: String,
    },
    
    /// Pull a model
    Pull {
        /// Model name
        model: String,
    },
}

#[derive(Deserialize, Debug)]
struct ModelInfo {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: ModelDetails,
}

#[derive(Deserialize, Debug)]
struct ModelDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Option<Vec<String>>,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Deserialize, Debug)]
struct ListResponse {
    models: Vec<ModelInfo>,
}

async fn list_models(client: &Client, url: &str) -> Result<(), Box<dyn Error>> {
    let full_url = format!("{}/api/tags", url);
    let response = client.get(&full_url).send().await?;
    
    if response.status().is_success() {
        let list_response: ListResponse = response.json().await?;
        
        println!("Available Models:");
        println!("=================");
        for model in list_response.models {
            println!("{}", model.name);
            println!("  Family: {}", model.details.family);
            println!("  Parameters: {}", model.details.parameter_size);
            println!("  Quantization: {}", model.details.quantization_level);
            println!("  Size: {:.2} GB", model.size as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("  Modified: {}", model.modified_at);
            println!();
        }
    } else {
        let status = response.status();
        let error_text = response.text().await?;
        eprintln!("Error {}: {}", status, error_text);
    }
    
    Ok(())
}

async fn model_info(client: &Client, url: &str, model_name: &str) -> Result<(), Box<dyn Error>> {
    let full_url = format!("{}/api/tags", url);
    let response = client.get(&full_url).send().await?;
    
    if response.status().is_success() {
        let list_response: ListResponse = response.json().await?;
        
        for model in list_response.models {
            if model.name == model_name {
                println!("Model Information:");
                println!("==================");
                println!("Name: {}", model.name);
                println!("Family: {}", model.details.family);
                println!("Parameters: {}", model.details.parameter_size);
                println!("Quantization: {}", model.details.quantization_level);
                println!("Format: {}", model.details.format);
                println!("Size: {:.2} GB", model.size as f64 / (1024.0 * 1024.0 * 1024.0));
                println!("Modified: {}", model.modified_at);
                if let Some(families) = model.details.families {
                    println!("Families: {}", families.join(", "));
                }
                return Ok(());
            }
        }
        
        eprintln!("Model '{}' not found", model_name);
    } else {
        let status = response.status();
        let error_text = response.text().await?;
        eprintln!("Error {}: {}", status, error_text);
    }
    
    Ok(())
}

async fn pull_model(client: &Client, url: &str, model_name: &str) -> Result<(), Box<dyn Error>> {
    println!("Pulling model '{}'...", model_name);
    
    let full_url = format!("{}/api/pull", url);
    let response = client
        .post(&full_url)
        .json(&serde_json::json!({ "name": model_name }))
        .send()
        .await?;
    
    if response.status().is_success() {
        println!("Model '{}' pulled successfully", model_name);
    } else {
        let status = response.status();
        let error_text = response.text().await?;
        eprintln!("Error {}: {}", status, error_text);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let client = Client::new();
    let url = "http://localhost:11434"; // Hardcoded for now
    
    match &cli.command {
        Commands::List => {
            list_models(&client, url).await?;
        }
        Commands::Info { model } => {
            model_info(&client, url, model).await?;
        }
        Commands::Pull { model } => {
            pull_model(&client, url, model).await?;
        }
    }
    
    Ok(())
}