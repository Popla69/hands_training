@echo off
echo ======================================================================
echo CLASS X WEBCAM TEST
echo ======================================================================
echo.
echo This will test Class X recognition in real-time using your webcam.
echo.
echo How to use:
echo   1. Make the X sign with your hand
echo   2. Press SPACE to start tracking accuracy
echo   3. Hold the X sign steady
echo   4. Press SPACE again to stop and see results
echo.
echo Using external camera (VideoCapture 1)
echo.
echo Press any key to start...
pause >nul

C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe test_webcam_class_X.py

echo.
echo ======================================================================
echo Test complete!
echo ======================================================================
echo.
pause
