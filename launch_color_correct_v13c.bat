@echo off
cd /d "%~dp0"
echo ===========================================================
echo   Running Batch Color Correction v13c (Rotation Fix)
echo ===========================================================
echo.

python "batch_color_correct_v13c.py"

echo.
echo -----------------------------------------------------------
echo   Done! Press any key to close this window.
echo -----------------------------------------------------------
pause >nul

