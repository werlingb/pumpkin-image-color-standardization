@echo off
echo ===========================================================
echo   Launching Create Reference NPZ (Fixed Window, v3)
echo ===========================================================
echo.

REM --- automatically find Python and run ---
python "%~dp0create_reference_npz_fixed_window_v3.py"

echo.
echo -----------------------------------------------------------
echo Done. The .npz and preview PNG are saved next to the image.
echo Close this window to exit.
pause
