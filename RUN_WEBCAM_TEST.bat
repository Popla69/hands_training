@echo off
echo ======================================================================
echo SIGN LANGUAGE RECOGNITION - WEBCAM TEST
echo ======================================================================
echo.
echo This will start the sign language recognition system.
echo Using external camera (VideoCapture 1)
echo.
echo Features:
echo   - Simple frame capture (5 frames per second)
echo   - All 29 signs (A-Z, space, del, nothing)
echo   - Voting system for accuracy
echo.
echo How to use:
echo   1. Position your hand in the GREEN BOX
echo   2. Make a sign and hold it steady
echo   3. Press SPACE to capture (takes 5 frames)
echo   4. System votes and adds the letter
echo.
echo Controls:
echo   - SPACE: Capture sign
echo   - C: Clear sequence
echo   - ESC: Exit
echo.
echo Press any key to start...
pause >nul

C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe classify_WORKING.py

echo.
echo ======================================================================
echo Session ended!
echo ======================================================================
echo.
pause
