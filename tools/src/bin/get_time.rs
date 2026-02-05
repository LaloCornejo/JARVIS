use jarvis_tools::time;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let timezone = if args.len() >= 3 && args[1] == "--timezone" {
        &args[2]
    } else if args.len() >= 2 && !args[1].starts_with("--") {
        &args[1]
    } else {
        "America/Mexico_City"
    };

    match time::execute(timezone) {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).unwrap());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
