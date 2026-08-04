use clap::{Parser, Subcommand};
use regex::Regex;
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// JARVIS High-Performance File Processor
#[derive(Parser)]
#[clap(name = "jarvis-file-processor", version = "0.1.0")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for patterns in files
    Search {
        /// Pattern to search for (regex)
        pattern: String,
        
        /// Path to search in
        path: String,
        
        /// Maximum number of results to return
        #[clap(short, long, default_value = "100")]
        limit: usize,
        
        /// Case insensitive search
        #[clap(short, long)]
        ignore_case: bool,
    },
    
    /// Count lines in files
    LineCount {
        /// Path to count lines in
        path: String,
    },
    
    /// Extract structured data from files
    Extract {
        /// Path to extract data from
        path: String,
        
        /// Extraction pattern (regex with capture groups)
        pattern: String,
    },
}

#[derive(Serialize, Debug)]
struct SearchResult {
    file: String,
    line_number: usize,
    content: String,
    matches: Vec<(usize, usize)>, // start, end positions of matches
}

#[derive(Serialize, Debug)]
struct FileProcessorResult {
    success: bool,
    data: Vec<SearchResult>,
    error: Option<String>,
    duration_ms: u128,
    files_processed: usize,
    total_matches: usize,
}

#[derive(Serialize, Debug)]
struct LineCountResult {
    success: bool,
    line_count: usize,
    file_size: u64,
    error: Option<String>,
    duration_ms: u128,
}

#[derive(Serialize, Debug)]
struct ExtractResult {
    success: bool,
    data: Vec<Vec<String>>, // captured groups for each match
    error: Option<String>,
    duration_ms: u128,
    matches_found: usize,
}

fn search_files(
    pattern: &str,
    path: &str,
    limit: usize,
    ignore_case: bool,
) -> Result<FileProcessorResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    let regex_pattern = if ignore_case {
        format!("(?i){}", pattern)
    } else {
        pattern.to_string()
    };
    
    let regex = Regex::new(&regex_pattern)?;
    let search_path = Path::new(path);
    
    let mut results = Vec::new();
    let mut files_processed = 0;
    let mut total_matches = 0;
    
    if search_path.is_file() {
        files_processed += 1;
        let file_results = search_in_file(search_path, &regex, limit.saturating_sub(results.len()))?;
        total_matches += file_results.len();
        results.extend(file_results);
    } else if search_path.is_dir() {
        for entry in walkdir::WalkDir::new(search_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if results.len() >= limit {
                break;
            }
            
            files_processed += 1;
            let file_results = search_in_file(entry.path(), &regex, limit.saturating_sub(results.len()))?;
            total_matches += file_results.len();
            results.extend(file_results);
        }
    }
    
    let duration = start_time.elapsed();
    
    Ok(FileProcessorResult {
        success: true,
        data: results,
        error: None,
        duration_ms: duration.as_millis(),
        files_processed,
        total_matches,
    })
}

fn search_in_file(
    file_path: &Path,
    regex: &Regex,
    limit: usize,
) -> Result<Vec<SearchResult>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut results = Vec::new();
    let mut line_number = 0;
    
    for line in reader.lines() {
        line_number += 1;
        let line_content = line?;
        
        let matches: Vec<(usize, usize)> = regex
            .find_iter(&line_content)
            .map(|m| (m.start(), m.end()))
            .collect();
        
        if !matches.is_empty() {
            results.push(SearchResult {
                file: file_path.to_string_lossy().to_string(),
                line_number,
                content: line_content,
                matches,
            });
            
            if results.len() >= limit {
                break;
            }
        }
    }
    
    Ok(results)
}

fn count_lines(path: &str) -> Result<LineCountResult, Box<dyn Error>> {
    let start_time = Instant::now();
    let file_path = Path::new(path);
    
    if !file_path.exists() {
        return Ok(LineCountResult {
            success: false,
            line_count: 0,
            file_size: 0,
            error: Some("File not found".to_string()),
            duration_ms: start_time.elapsed().as_millis(),
        });
    }
    
    let metadata = fs::metadata(file_path)?;
    let file_size = metadata.len();
    
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let line_count = reader.lines().count();
    
    let duration = start_time.elapsed();
    
    Ok(LineCountResult {
        success: true,
        line_count,
        file_size,
        error: None,
        duration_ms: duration.as_millis(),
    })
}

fn extract_data(path: &str, pattern: &str) -> Result<ExtractResult, Box<dyn Error>> {
    let start_time = Instant::now();
    let regex = Regex::new(pattern)?;
    let file_path = Path::new(path);
    
    if !file_path.exists() {
        return Ok(ExtractResult {
            success: false,
            data: Vec::new(),
            error: Some("File not found".to_string()),
            duration_ms: start_time.elapsed().as_millis(),
            matches_found: 0,
        });
    }
    
    let mut file = File::open(file_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    
    let mut captures = Vec::new();
    let mut matches_found = 0;
    
    for cap in regex.captures_iter(&content) {
        matches_found += 1;
        let mut groups = Vec::new();
        for i in 0..cap.len() {
            groups.push(cap.get(i).map(|m| m.as_str()).unwrap_or("").to_string());
        }
        captures.push(groups);
    }
    
    let duration = start_time.elapsed();
    
    Ok(ExtractResult {
        success: true,
        data: captures,
        error: None,
        duration_ms: duration.as_millis(),
        matches_found,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    
    let result = match &cli.command {
        Commands::Search {
            pattern,
            path,
            limit,
            ignore_case,
        } => {
            let result = search_files(pattern, path, *limit, *ignore_case)?;
            serde_json::to_string(&result)?
        }
        Commands::LineCount { path } => {
            let result = count_lines(path)?;
            serde_json::to_string(&result)?
        }
        Commands::Extract { path, pattern } => {
            let result = extract_data(path, pattern)?;
            serde_json::to_string(&result)?
        }
    };
    
    println!("{}", result);
    Ok(())
}