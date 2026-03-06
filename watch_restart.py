#!/usr/bin/env python3
"""Auto-restart JARVIS when files change."""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class RestartHandler(FileSystemEventHandler):
    def __init__(self, watch_dir: str, process_name: str = "jarvis_wrapper.py"):
        self.watch_dir = Path(watch_dir).resolve()
        self.process_name = process_name
        self.process = None
        self.restarting = False
        self.ignore_patterns = {
            "__pycache__",
            ".pyc",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".env",
            "*.log",
            "jarvis_server_debug.log",
            "jarvis_debug.log",
        }

    def should_ignore(self, path: str) -> bool:
        path_obj = Path(path)
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if path_obj.suffix == pattern[1:]:
                    return True
            if pattern in str(path_obj):
                return True
        return False

    def on_modified(self, event):
        if event.is_directory:
            return
        if self.should_ignore(event.src_path):
            return
        if self.restarting:
            return
        print(f"[WATCH] File changed: {event.src_path}")
        self.restart()

    def on_created(self, event):
        if event.is_directory:
            return
        if self.should_ignore(event.src_path):
            return
        print(f"[WATCH] File created: {event.src_path}")
        self.restart()

    def restart(self):
        self.restarting = True
        print("[WATCH] Restarting JARVIS...")

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        time.sleep(0.5)

        self.process = subprocess.Popen(
            [sys.executable, self.process_name],
            cwd=self.watch_dir,
            env=os.environ.copy(),
        )
        print(f"[WATCH] Started PID: {self.process.pid}")
        self.restarting = False

    def start(self):
        print(f"[WATCH] Watching: {self.watch_dir}")
        print("[WATCH] Press Ctrl+C to stop")

        self.restart()

        observer = Observer()
        observer.schedule(self, str(self.watch_dir), recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
                if self.process.poll() is not None:
                    print("[WATCH] Process died, restarting...")
                    self.restart()
        except KeyboardInterrupt:
            print("\n[WATCH] Stopping...")
            observer.stop()
            if self.process:
                self.process.terminate()
        observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-restart JARVIS on file changes")
    parser.add_argument(
        "--dir",
        "-d",
        default=".",
        help="Directory to watch (default: current directory)",
    )
    args = parser.parse_args()

    handler = RestartHandler(args.dir)
    handler.start()
