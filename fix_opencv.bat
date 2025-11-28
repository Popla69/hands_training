@echo off
echo ========================================
echo   Fixing OpenCV zlibwapi.dll Issue
echo ========================================
echo.
echo This will reinstall opencv-python...
echo.
pause

C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe -m pip install opencv-python==4.8.1.78

echo.
echo ========================================
echo   Done!
echo ========================================
echo.
pause
