use std::env;
use system_control::{lock_computer, sleep_computer, shutdown_computer, restart_computer, turn_off_monitor, set_volume, volume_up, volume_down, toggle_mute};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    let result = match args[1].as_str() {
        "lock" => {
            lock_computer();
            ToolResult::success("Computer locked successfully")
        },
        "sleep" => {
            sleep_computer();
            ToolResult::success("Computer put to sleep")
        },
        "shutdown" => {
            shutdown_computer();
            ToolResult::success("Shutdown initiated")
        },
        "restart" => {
            restart_computer();
            ToolResult::success("Restart initiated")
        },
        "turn_off_monitor" => {
            turn_off_monitor();
            ToolResult::success("Monitor turned off")
        },
        "volume_up" => {
            match volume_up() {
                Ok(()) => ToolResult::success("Volume increased by 10%"),
                Err(e) => ToolResult::error(&e),
            }
        },
        "volume_down" => {
            match volume_down() {
                Ok(()) => ToolResult::success("Volume decreased by 10%"),
                Err(e) => ToolResult::error(&e),
            }
        },
        "mute" => {
            match toggle_mute() {
                Ok(()) => ToolResult::success("Mute state toggled"),
                Err(e) => ToolResult::error(&e),
            }
        },
        "set_volume" => {
            if args.len() < 3 {
                ToolResult::error("Usage: system-control set_volume <level>")
            } else {
                match args[2].parse::<u32>() {
                    Ok(level) => {
                        match set_volume(level) {
                            Ok(()) => ToolResult::success(&format!("Volume set to {}%", level)),
                            Err(e) => ToolResult::error(&e),
                        }
                    },
                    Err(_) => ToolResult::error("Volume level must be a number between 0 and 100"),
                }
            }
        },
        "help" | "-h" | "--help" => {
            print_usage();
            return;
        },
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            ToolResult::error(&format!("Unknown command: {}", args[1]))
        }
    };
    
    // Output JSON result
    println!("{}", serde_json::to_string(&result).unwrap_or_else(|_| r#"{"exit_code": 1, "stdout": "", "stderr": "JSON serialization error"}"#.to_string()));
}

fn print_usage() {
    eprintln!("System Control Tool");
    eprintln!("Usage: system-control <command> [arguments]");
    eprintln!("");
    eprintln!("Available commands:");
    eprintln!("  lock             Lock computer");
    eprintln!("  sleep            Put computer to sleep");
    eprintln!("  shutdown         Shutdown computer");
    eprintln!("  restart          Restart computer");
    eprintln!("  turn_off_monitor Turn off monitor");
    eprintln!("  volume_up        Increase system volume by 10%");
    eprintln!("  volume_down      Decrease system volume by 10%");
    eprintln!("  mute             Toggle mute state");
    eprintln!("  set_volume <0-100> Set volume to specific level");
    eprintln!("  help             Show this help message");
}

#[derive(serde::Serialize)]
struct ToolResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

impl ToolResult {
    fn success(message: &str) -> Self {
        Self {
            exit_code: 0,
            stdout: message.to_string(),
            stderr: String::new(),
        }
    }
    
    fn error(message: &str) -> Self {
        Self {
            exit_code: 1,
            stdout: String::new(),
            stderr: message.to_string(),
        }
    }
}