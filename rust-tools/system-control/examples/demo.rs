use system_control::{lock_computer, sleep_computer, shutdown_computer, restart_computer, turn_off_monitor};

fn main() {
    println!("System Control Demo");
    println!("This demo shows how to use the system control functions.");
    println!("WARNING: Uncommenting the shutdown/restart functions will affect your system!");
    
    // Safe functions to demonstrate:
    lock_computer();  // This will lock your workstation
    turn_off_monitor();  // This will turn off your monitor
    
    // DANGEROUS - Uncomment only if you want to test:
    // sleep_computer();  // Puts computer to sleep
    // shutdown_computer();  // Shuts down computer
    // restart_computer();  // Restarts computer
}