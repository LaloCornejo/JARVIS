@echo off
echo Building System Control Tool...
cargo build --release
echo.
echo Build complete!
echo.
echo To use the tool, run:
echo target\release\system-control.exe [command]
echo.
echo For example:
echo target\release\system-control.exe lock
echo.
pause