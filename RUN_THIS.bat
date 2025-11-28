@echo off
cls
echo ============================================================
echo   HACKATHON DEMO - FIXED VERSION
echo ============================================================
echo.
echo IMPORTANT: The "zlibwapi.dll" warning is HARMLESS!
echo The camera will work fine - just ignore that message.
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul
cls
C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe classify_HACKATHON.py
echo.
echo ============================================================
echo   Demo ended
echo ============================================================
pause
