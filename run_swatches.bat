@echo off
REM ============================================================
REM  Pumpkin Swatch Generator Launcher
REM  Runs create_color_swatches_from_csv.py in this folder
REM ============================================================

echo.
echo ==== Pumpkin Swatch Generator ====
echo.

REM Get the directory of this BAT file
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python not found. Please install Python and add it to PATH.
    pause
    exit /b
)

REM Run the script
python create_color_swatches_from_csv.py

echo.
echo ==== Done! Press any key to close ====
pause
