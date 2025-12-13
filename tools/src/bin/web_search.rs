use jarvis_tools::web_search;
use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: web_search <query> <num_results>");
        std::process::exit(1);
    }
    let query = &args[1];
    let num_results: usize = args[2].parse().unwrap_or(5);

    match web_search::execute(query, num_results).await {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).unwrap());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}