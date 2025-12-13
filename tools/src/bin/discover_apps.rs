use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use winreg::enums::*;
use winreg::RegKey;

#[derive(Serialize, Deserialize, Debug)]
struct AppInfo {
    name: String,
    exe_path: String,
    exe_name: String,
    aliases: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct AppDiscoveryResult {
    apps: Vec<AppInfo>,
    count: usize,
}

fn get_installed_apps_from_registry() -> Result<Vec<AppInfo>, Box<dyn std::error::Error>> {
    let mut apps = Vec::new();
    
    // Check both 32-bit and 64-bit registry locations
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let keys_to_check = vec![
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    ];
    
    for key_path in keys_to_check {
        if let Ok(uninstall_key) = hklm.open_subkey(key_path) {
            for subkey_name in uninstall_key.enum_keys().flatten() {
                if let Ok(subkey) = uninstall_key.open_subkey(&subkey_name) {
                    // Skip system components and updates
                    if let Ok(system_component) = subkey.get_value::<u32, _>("SystemComponent") {
                        if system_component == 1 {
                            continue;
                        }
                    }
                    
                    if let Ok(patch_package) = subkey.get_value::<u32, _>("IsMinorUpgrade") {
                        if patch_package == 1 {
                            continue;
                        }
                    }
                    
                    // Get display name
                    if let Ok(display_name) = subkey.get_value::<String, _>("DisplayName") {
                        // Skip entries without executable
                        let has_exe = subkey.get_value::<String, _>("DisplayIcon").is_ok() 
                            || subkey.get_value::<String, _>("InstallLocation").is_ok();
                        
                        if !has_exe {
                            continue;
                        }
                        
                        // Get executable path
                        let mut exe_path = String::new();
                        let mut exe_name = String::new();
                        
                        // Try to get executable from DisplayIcon
                        if let Ok(icon_path) = subkey.get_value::<String, _>("DisplayIcon") {
                            if !icon_path.is_empty() {
                                // Clean up the path (remove quotes and ,0 suffix)
                                let clean_path = icon_path.trim_matches('"').split(',').next().unwrap_or("").to_string();
                                if !clean_path.is_empty() && Path::new(&clean_path).exists() {
                                    exe_path = clean_path.clone();
                                    if let Some(file_name) = Path::new(&exe_path).file_name() {
                                        exe_name = file_name.to_string_lossy().to_string();
                                    }
                                }
                            }
                        }
                        
                        // If we couldn't get it from DisplayIcon, try InstallLocation + DisplayName
                        if exe_path.is_empty() {
                            if let Ok(install_location) = subkey.get_value::<String, _>("InstallLocation") {
                                if !install_location.is_empty() {
                                    let install_path = Path::new(&install_location);
                                    // Look for common executable patterns
                                    let possible_names = vec![
                                        format!("{}.exe", display_name.replace(" ", "")),
                                        format!("{}.exe", display_name),
                                        "launcher.exe".to_string(),
                                        "start.exe".to_string(),
                                    ];
                                    
                                    for name in &possible_names {
                                        let possible_exe = install_path.join(name);
                                        if possible_exe.exists() {
                                            exe_path = possible_exe.to_string_lossy().to_string();
                                            exe_name = name.clone();
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Only add apps with valid executable paths
                        if !exe_path.is_empty() {
                            let mut aliases = vec![display_name.to_lowercase()];
                            
                            // Add common aliases
                            let clean_name = display_name
                                .to_lowercase()
                                .replace("microsoft", "")
                                .replace("(r)", "")
                                .replace("(tm)", "")
                                .trim()
                                .to_string();
                                
                            if clean_name != display_name.to_lowercase() {
                                aliases.push(clean_name);
                            }
                            
                            // Add abbreviated versions
                            let abbreviated = display_name
                                .split_whitespace()
                                .map(|word| word.chars().next().unwrap_or_default().to_string())
                                .collect::<Vec<_>>()
                                .join("");
                                
                            if abbreviated.len() > 1 && abbreviated != display_name.to_lowercase() {
                                aliases.push(abbreviated);
                            }
                            
                            apps.push(AppInfo {
                                name: display_name,
                                exe_path,
                                exe_name: exe_name.trim_end_matches(".exe").to_string(),
                                aliases,
                            });
                        }
                    }
                }
            }
        }
    }
    
    Ok(apps)
}

fn get_apps_from_start_menu() -> Result<Vec<AppInfo>, Box<dyn std::error::Error>> {
    let mut apps = Vec::new();
    
    // Common start menu locations
    let start_menu_paths = vec![
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
        r"C:\Users\Public\Desktop",
    ];
    
    // Also check user-specific start menu
    if let Ok(user_name) = std::env::var("USERNAME") {
        let user_start_menu = format!(r"C:\Users\{}\AppData\Roaming\Microsoft\Windows\Start Menu\Programs", user_name);
        let user_desktop = format!(r"C:\Users\{}\Desktop", user_name);
        let mut user_paths = start_menu_paths.clone();
        user_paths.push(&user_start_menu);
        user_paths.push(&user_desktop);
    }
    
    for base_path in &start_menu_paths {
        let base_dir = Path::new(base_path);
        if base_dir.exists() {
            // Recursively search for .lnk files
            for entry in walkdir::WalkDir::new(base_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "lnk"))
            {
                if let Some(file_name) = entry.path().file_stem() {
                    let app_name = file_name.to_string_lossy().to_string();
                    // For simplicity, we'll use the file name as the executable name
                    // In a real implementation, you'd want to resolve the shortcut target
                    let exe_name = app_name.to_lowercase().replace(" ", "");
                    
                    apps.push(AppInfo {
                        name: app_name.clone(),
                        exe_path: entry.path().to_string_lossy().to_string(),
                        exe_name,
                        aliases: vec![app_name.to_lowercase()],
                    });
                }
            }
        }
    }
    
    Ok(apps)
}

fn get_apps_from_common_directories() -> Vec<AppInfo> {
    let mut apps = Vec::new();
    
    // Check user-specific directories where apps like Obsidian might be installed
    if let Ok(username) = std::env::var("USERNAME") {
        let local_programs = format!(r"C:\Users\{}\AppData\Local\Programs", username);
        
        // Check the Local Programs directory specifically
        let base_dir = Path::new(&local_programs);
        if base_dir.exists() {
            // Look for subdirectories that might contain apps
            if let Ok(entries) = fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    let app_dir = entry.path();
                    if app_dir.is_dir() {
                        // Look for common executable patterns in the app directory
                        let possible_exe_names = vec![
                            "Obsidian.exe", "obsidian.exe",
                            "Launcher.exe", "launcher.exe",
                            "Start.exe", "start.exe",
                            "App.exe", "app.exe",
                        ];
                        
                        // Special handling for Obsidian to ensure it gets added correctly
                        if let Some(dir_name) = app_dir.file_name() {
                            let dir_name_str = dir_name.to_string_lossy().to_string();
                            if dir_name_str.to_lowercase() == "obsidian" {
                                // For Obsidian, use the properly capitalized executable
                                let exe_path = app_dir.join("Obsidian.exe");
                                if exe_path.exists() {
                                    let app_name = "Obsidian".to_string();
                                    let aliases = vec![
                                        "obsidian".to_string(),
                                        "obsidian app".to_string(),
                                    ];
                                    
                                    apps.push(AppInfo {
                                        name: app_name,
                                        exe_path: exe_path.to_string_lossy().to_string(),
                                        exe_name: "Obsidian".to_string(),
                                        aliases,
                                    });
                                    continue; // Skip the general loop for Obsidian
                                }
                            }
                        }
                        
                        // General case for other apps
                        for exe_name in &possible_exe_names {
                            let exe_path = app_dir.join(exe_name);
                            if exe_path.exists() {
                                if let Some(app_display_name) = app_dir.file_name() {
                                    let app_name = app_display_name.to_string_lossy().to_string();
                                    let aliases = vec![
                                        app_name.to_lowercase(),
                                        format!("{} app", app_name.to_lowercase()),
                                    ];
                                    
                                    apps.push(AppInfo {
                                        name: app_name,
                                        exe_path: exe_path.to_string_lossy().to_string(),
                                        exe_name: exe_name.trim_end_matches(".exe").to_string(),
                                        aliases,
                                    });
                                    // Break after finding the first executable to avoid duplicates
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    apps
}

fn get_common_apps() -> Vec<AppInfo> {
    let common_apps = vec![
        ("Notepad", "notepad.exe", vec!["notepad"]),
        ("Calculator", "calc.exe", vec!["calc", "calculator"]),
        ("Paint", "mspaint.exe", vec!["paint", "mspaint"]),
        ("WordPad", "write.exe", vec!["wordpad"]),
        ("Command Prompt", "cmd.exe", vec!["cmd", "command prompt"]),
        ("PowerShell", "powershell.exe", vec!["powershell", "ps"]),
        ("Task Manager", "taskmgr.exe", vec!["task manager", "tasks"]),
        ("Control Panel", "control.exe", vec!["control panel", "control"]),
        ("File Explorer", "explorer.exe", vec!["explorer", "files", "file explorer"]),
        ("Settings", "ms-settings:", vec!["settings", "windows settings"]),
    ];
    
    common_apps
        .into_iter()
        .map(|(name, exe_path, aliases)| AppInfo {
            name: name.to_string(),
            exe_path: exe_path.to_string(),
            exe_name: exe_path.trim_end_matches(".exe").to_string(),
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut all_apps = Vec::new();
    
    // Get apps from registry
    match get_installed_apps_from_registry() {
        Ok(mut apps) => all_apps.append(&mut apps),
        Err(e) => eprintln!("Error getting apps from registry: {}", e),
    }
    
    // Get apps from start menu
    match get_apps_from_start_menu() {
        Ok(mut apps) => all_apps.append(&mut apps),
        Err(e) => eprintln!("Error getting apps from start menu: {}", e),
    }
    
    // Get apps from common directories
    all_apps.append(&mut get_apps_from_common_directories());
    
    // Add common Windows apps
    all_apps.append(&mut get_common_apps());
    
    // Deduplicate apps by executable name
    let mut seen_exes = std::collections::HashSet::new();
    all_apps.retain(|app| {
        if seen_exes.contains(&app.exe_name) {
            false
        } else {
            seen_exes.insert(app.exe_name.clone());
            true
        }
    });
    
    let result = AppDiscoveryResult {
        apps: all_apps,
        count: seen_exes.len(),
    };
    
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}