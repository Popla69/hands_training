@echo off
echo ========================================
echo Setting up Python 3.7 Environment
echo ========================================
echo.

echo This will:
echo 1. Download Python 3.7.9 (last 3.7 version)
echo 2. Install it to python37 folder
echo 3. Create venv with TensorFlow 1.15
echo 4. Install all requirements
echo.
echo This may take 10-15 minutes
echo.
pause

echo.
echo [1/5] Downloading Python 3.7.9...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe' -OutFile 'python-3.7.9-installer.exe'"

if not exist python-3.7.9-installer.exe (
    echo ERROR: Download failed
    echo Please download manually from: https://www.python.org/downloads/release/python-379/
    pause
    exit /b 1
)

echo ✓ Downloaded

echo.
echo [2/5] Installing Python 3.7.9...
echo Installing to: %CD%\python37
python-3.7.9-installer.exe /quiet InstallAllUsers=0 TargetDir=%CD%\python37 PrependPath=0 Include_test=0

timeout /t 30 /nobreak

if not exist python37\python.exe (
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo ✓ Python 3.7 installed

echo.
echo [3/5] Creating virtual environment...
python37\python.exe -m venv venv_py37

if not exist venv_py37\Scripts\activate.bat (
    echo ERROR: venv creation failed
    pause
    exit /b 1
)

echo ✓ venv created

echo.
echo [4/5] Upgrading pip...
venv_py37\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [5/5] Installing TensorFlow 1.15 and dependencies...
venv_py37\Scripts\pip.exe install tensorflow==1.15.0
venv_py37\Scripts\pip.exe install numpy==1.16.4
venv_py37\Scripts\pip.exe install opencv-python==4.1.2.30
venv_py37\Scripts\pip.exe install matplotlib

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use this environment:
echo   venv_py37\Scripts\activate.bat
echo.
echo Or use the provided batch files:
echo   run_SIMPLE_PY37.bat
echo   run_WORKING_PY37.bat
echo.
pause
