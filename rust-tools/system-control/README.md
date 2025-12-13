# System Control Tool

A Rust library and command-line tool for controlling system behaviors on Windows.

## Features

- Lock workstation
- Put computer to sleep
- Shutdown computer
- Restart computer
- Turn off monitor
- Control system volume (volume up/down, mute, set volume)

## Installation

Make sure you have Rust installed. Then clone this repository and build the project:

```bash
cd JARVIS/rust-tools/system-control
cargo build --release
```

## Usage

### As a Command-Line Tool

```bash
# Lock the workstation
system-control lock

# Put computer to sleep
system-control sleep

# Shutdown computer
system-control shutdown

# Restart computer
system-control restart

# Turn off monitor
system-control turn_off_monitor

# Increase system volume by 10%
system-control volume_up

# Decrease system volume by 10%
system-control volume_down

# Toggle mute state
system-control mute

# Set volume to specific level (0-100)
system-control set_volume 50
```

### As a Library

Add this to your `Cargo.toml`:

```toml
[dependencies]
system-control = { path = "../JARVIS/rust-tools/system-control" }
```

Then use it in your code:

```rust
use system_control::{lock_computer, sleep_computer, shutdown_computer, restart_computer, turn_off_monitor, set_volume, volume_up, volume_down, toggle_mute};

fn main() {
    // Lock the workstation
    lock_computer();
    
    // Turn off monitor
    turn_off_monitor();
    
    // Control volume
    volume_up().unwrap();
    volume_down().unwrap();
    toggle_mute().unwrap();
    set_volume(75).unwrap();
    
    // WARNING: These will affect your system!
    // sleep_computer();
    // shutdown_computer();
    // restart_computer();
}
```

## Safety Warning

⚠️ **Warning**: Some of these functions will immediately affect your system:
- `shutdown_computer()` will immediately shut down your computer
- `restart_computer()` will immediately restart your computer
- `sleep_computer()` will put your computer to sleep

Use these functions with caution!

## Limitations

The volume control functions are currently placeholders. A full implementation would require COM interfaces for audio control, which is more complex to implement safely.

## License

MIT