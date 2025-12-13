use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
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

fn process_json_dataset(
    input: &str,
    output: &str,
    operation: &str,
    field: &str,
) -> Result<JsonResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    
    let processed_records: Vec<serde_json::Value> = lines
        .par_iter()  // Parallel processing with Rayon
        .filter_map(|line| {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(line) {
                match operation {
                    "filter" => {
                        if let Some(field_value) = json_value.get(field) {
                            if !field_value.is_null() {
                                Some(json_value)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    "count" => Some(json_value),
                    _ => Some(json_value),
                }
            } else {
                None
            }
        })
        .collect();
    
    // Write results to output file
    let mut output_file = File::create(output)?;
    for record in &processed_records {
        writeln!(output_file, "{}", serde_json::to_string(record)?)?;
    }
    
    let duration = start_time.elapsed();
    
    Ok(JsonResult {
        success: true,
        data: Some(serde_json::Value::String(format!(
            "Processed {} records, {} filtered",
            lines.len(),
            processed_records.len()
        ))),
        error: None,
        duration_ms: duration.as_millis(),
        records_processed: processed_records.len(),
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
            if let Some(pat) = pattern {
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
    
    /// Batch data transformation
    TransformBatch {
        /// Input directory
        input_dir: String,
        
        /// Output directory
        output_dir: String,
        
        /// Transformation type
        transform: String,
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

fn process_json_dataset(
    input: &str,
    output: &str,
    operation: &str,
    field: &str,
) -> Result<JsonResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    
    let processed_records: Vec<serde_json::Value> = lines
        .par_iter()  // Parallel processing with Rayon
        .filter_map(|line| {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(line) {
                match operation {
                    "filter" => {
                        if let Some(field_value) = json_value.get(field) {
                            if !field_value.is_null() {
                                Some(json_value)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    "count" => Some(json_value),
                    _ => Some(json_value),
                }
            } else {
                None
            }
        })
        .collect();
    
    // Write results to output file
    let mut output_file = File::create(output)?;
    for record in &processed_records {
        writeln!(output_file, "{}", serde_json::to_string(record)?)?;
    }
    
    let duration = start_time.elapsed();
    
    Ok(JsonResult {
        success: true,
        data: Some(serde_json::Value::String(format!(
            "Processed {} records, {} filtered",
            lines.len(),
            processed_records.len()
        ))),
        error: None,
        duration_ms: duration.as_millis(),
        records_processed: processed_records.len(),
    })
}

fn analyze_files_parallel(path: &str, pattern: Option<&str>) -> Result<FileAnalysisResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let search_path = Path::new(path);
    if !search_path.exists() || !search_path.is_dir() {
        return Ok(FileAnalysisResult {
            success: false,
            data: None,
            error: Some("Path does not exist or is not a directory".to_string()),
            duration_ms: 0,
            files_analyzed: 0,
        });
    }
    
    // Collect all files that match the pattern
    let files: Vec<_> = walkdir::WalkDir::new(search_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            if let Some(pat) = pattern {
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
    
    Ok(FileAnalysisResult {
        success: true,
        data: Some(file_stats.clone()),
        error: None,
        duration_ms: duration.as_millis(),
        files_analyzed: file_stats.len(),
    })
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
        Commands::TransformBatch {
            input_dir: _,
            output_dir: _,
            transform: _,
        } => {
            serde_json::to_string(&JsonResult {
                success: true,
                data: Some(serde_json::Value::String("Batch transformation not yet implemented".to_string())),
                error: None,
                duration_ms: 0,
                records_processed: 0,
            })?
        }
    };
    
    println!("{}", result);
    Ok(())
}