use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

/// JARVIS High-Performance Data Processor
#[derive(Parser)]
#[clap(name = "jarvis-data-processor", version = "0.1.0")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process large JSON datasets in parallel
    ProcessJson {
        /// Input JSON file (newline-delimited)
        input: String,
        
        /// Output file for processed results
        output: String,
        
        /// Operation to perform (count, filter, aggregate)
        operation: String,
        
        /// Field to operate on
        field: String,
    },
    
    /// Parallel file analysis
    AnalyzeFiles {
        /// Directory to analyze
        path: String,
        
        /// File pattern to match
        pattern: Option<String>,
    },
}

#[derive(Serialize, Debug)]
struct JsonResult {
    success: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
    duration_ms: u128,
    records_processed: usize,
}

#[derive(Serialize, Debug, Clone)]
struct FileStats {
    path: String,
    size: u64,
    lines: usize,
    words: usize,
    extension: String,
}

#[derive(Serialize, Debug)]
struct FileAnalysisResult {
    success: bool,
    data: Option<Vec<FileStats>>,
    error: Option<String>,
    duration_ms: u128,
    files_analyzed: usize,
}

/// Process large JSON datasets with streaming to reduce memory usage
fn process_json_dataset(
    input: &str,
    output: &str,
    operation: &str,
    field: &str,
) -> Result<JsonResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    
    // Stream processing - process one line at a time instead of loading all into memory
    let mut processed_count = 0;
    let mut total_count = 0;
    
    // Open output file early
    let mut output_file = File::create(output)?;
    
    // Process lines one by one to reduce memory footprint
    for line_result in reader.lines() {
        let line = line_result?;
        total_count += 1;
        
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&line) {
            let should_include = match operation {
                "filter" => {
                    if let Some(field_value) = json_value.get(field) {
                        !field_value.is_null()
                    } else {
                        false
                    }
                }
                "count" => true,
                _ => true,
            };
            
            if should_include {
                writeln!(output_file, "{}", serde_json::to_string(&json_value)?)?;
                processed_count += 1;
            }
        }
        
        // Periodic progress reporting for very large files
        if total_count % 10000 == 0 {
            eprintln!("Processed {} lines...", total_count);
        }
    }
    
    let duration = start_time.elapsed();
    
    Ok(JsonResult {
        success: true,
        data: Some(serde_json::Value::String(format!(
            "Processed {} records, {} filtered",
            total_count,
            processed_count
        ))),
        error: None,
        duration_ms: duration.as_millis(),
        records_processed: processed_count,
    })
}

fn analyze_files_parallel(path: &str, pattern: Option<&str>) -> Result<serde_json::Value, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let search_path = Path::new(path);
    if !search_path.exists() || !search_path.is_dir() {
        return Ok(serde_json::json!({
            "success": false,
            "error": "Path does not exist or is not a directory"
        }));
    }
    
    // Collect all files that match the pattern
    let files: Vec<_> = walkdir::WalkDir::new(search_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            if let Some(ref pat) = pattern {
                e.file_name().to_string_lossy().contains(pat)
            } else {
                true
            }
        })
        .collect();
    
    // Process files in parallel
    let file_stats: Vec<FileStats> = files
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            if let Ok(metadata) = std::fs::metadata(path) {
                let size = metadata.len();
                let extension = path.extension()
                    .map(|ext| ext.to_string_lossy().to_string())
                    .unwrap_or_default();
                
                // Count lines and words
                if let Ok(mut file) = File::open(path) {
                    let mut content = String::new();
                    if file.read_to_string(&mut content).is_ok() {
                        let lines = content.lines().count();
                        let words = content.split_whitespace().count();
                        
                        return Some(FileStats {
                            path: path.to_string_lossy().to_string(),
                            size,
                            lines,
                            words,
                            extension,
                        });
                    }
                }
            }
            None
        })
        .collect();
    
    let duration = start_time.elapsed();
    
    Ok(serde_json::json!({
        "success": true,
        "data": file_stats,
        "duration_ms": duration.as_millis(),
        "files_analyzed": file_stats.len()
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    
    let result = match &cli.command {
        Commands::ProcessJson {
            input,
            output,
            operation,
            field,
        } => {
            let result = process_json_dataset(input, output, operation, field)?;
            serde_json::to_string(&result)?
        }
        Commands::AnalyzeFiles { path, pattern } => {
            let result = analyze_files_parallel(path, pattern.as_deref())?;
            serde_json::to_string(&result)?
        }
    };
    
    println!("{}", result);
    Ok(())
}