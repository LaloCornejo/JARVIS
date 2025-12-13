use jarvis_tools::files;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: write_file <path> <content> <append>");
        std::process::exit(1);
    }
    let path = &args[1];
    let content = &args[2];
    let append: bool = args.get(3).map(|s| s.parse().unwrap_or(false)).unwrap_or(false);
    match files::write_file(path, content, append).await {
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