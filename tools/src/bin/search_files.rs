use jarvis_tools::files;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: search_files <path> <pattern>");
        std::process::exit(1);
    }
    let path = &args[1];
    let pattern = &args[2];
    match files::search_files(path, pattern).await {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).unwrap());
        }
        Err(e) => {
            let result = jarvis_tools::files::ToolResult {
                exit_code: Some(1),
                stdout: "".to_string(),
                stderr: e.to_string(),
            };
            println!("{}", serde_json::to_string(&result).unwrap());
        }
    }
}