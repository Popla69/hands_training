@echo off
echo ======================================================================
echo RESUMABLE TRAINING - Sign Language Recognition
echo ======================================================================
echo.
echo This training can be interrupted and resumed!
echo.
echo Features:
echo   - Saves checkpoint after each epoch
echo   - Automatically resumes from last checkpoint
echo   - Safe to stop/restart anytime
echo.
echo Press Ctrl+C to cancel, or
pause
echo.
C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe train_RESUMABLE.py
echo.
pause
