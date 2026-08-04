use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Test if we can find Obsidian in the expected location
    if let Ok(username) = env::var("USERNAME") {
        let local_programs = format!(r"C:\Users\{}\AppData\Local\Programs", username);
        println!("Checking directory: {}", local_programs);
        
        let base_dir = Path::new(&local_programs);
        if base_dir.exists() {
            println!("Directory exists, listing contents:");
            if let Ok(entries) = fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        println!("Found directory: {:?}", path.file_name());
                        // Check if this is the Obsidian directory
                        if let Some(dir_name) = path.file_name() {
                            let dir_name_str = dir_name.to_string_lossy().to_string();
                            if dir_name_str.to_lowercase().contains("obsidian") {
                                println!("Found Obsidian directory!");
                                // Look for executable
                                let possible_exe_names = vec![
                                    "Obsidian.exe", "obsidian.exe",
                                    "Launcher.exe", "launcher.exe",
                                    "Start.exe", "start.exe",
                                    "App.exe", "app.exe",
                                ];
                                
                                for exe_name in &possible_exe_names {
                                    let exe_path = path.join(exe_name);
                                    if exe_path.exists() {
                                        println!("Found executable: {:?}", exe_path);
                                    } else {
                                        println!("Checked but didn't find: {:?}", exe_path);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                println!("Could not read directory");
            }
        } else {
            println!("Directory does not exist");
        }
    } else {
        println!("Could not get username");
    }
}