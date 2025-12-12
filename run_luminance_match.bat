@echo off
echo ===========================================================
echo   Running Luminance Match to Original
echo ===========================================================

REM Get the folder of this BAT file
cd /d "%~dp0"

REM Run the Python script
python luminance_match_to_original.py

echo.
echo ===========================================================
echo   Done! Press any key to close.
echo ===========================================================
pause >nul
