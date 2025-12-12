@echo off
echo ===========================================================
echo   Running Resize Pairs by Variety (PNG + CSV Output)
echo ===========================================================

REM --- Change directory to the script's location ---
cd /d "%~dp0"

REM --- Run the Python script ---
python "resize_pairs_by_object_png.py"

echo.
echo Done! Check the "_resized" folder next to your input folder.
pause
