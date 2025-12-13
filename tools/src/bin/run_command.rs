use jarvis_tools::run_command;
use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: run_command <command> <timeout_secs>");
        std::process::exit(1);
    }
    let command = &args[1];
    let timeout_secs: u64 = args[2].parse().unwrap_or(30);

    match run_command::execute(command, timeout_secs).await {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).unwrap());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}