@echo off
title Sign Language Recognition - HACKATHON DEMO
color 0A
echo.
echo ========================================
echo   HACKATHON DEMO - Sign Language
echo ========================================
echo.
echo NOTE: You may see a "zlibwapi.dll" warning - IGNORE IT!
echo The camera will still work fine.
echo.
pause
C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe classify_HACKATHON.py 2>nul
pause
