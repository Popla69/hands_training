@echo off
echo ========================================
echo CLEAN Python Environment Setup
echo ========================================
echo This will create a fresh environment
echo ========================================
pause

echo [1/3] Creating fresh venv...
C:\Users\KISHAN\AppData\Local\Programs\Python\Python310\python.exe -m venv venv_clean

echo [2/3] Installing compatible versions...
venv_clean\Scripts\pip.exe install tensorflow==2.10.0
venv_clean\Scripts\pip.exe install opencv-python==4.8.1.78
venv_clean\Scripts\pip.exe install numpy==1.23.5
venv_clean\Scripts\pip.exe install Pillow

echo [3/3] Done!
echo.
echo To use: venv_clean\Scripts\activate.bat
echo Then run: python train_NO_ZLIB.py
pause
