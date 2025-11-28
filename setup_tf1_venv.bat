@echo off
echo ========================================
echo Setting up TensorFlow 1.15 Virtual Environment
echo ========================================
echo.

echo [1/5] Creating virtual environment...
python -m venv venv_tf1
if errorlevel 1 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)
echo ✓ Virtual environment created

echo.
echo [2/5] Activating venv...
call venv_tf1\Scripts\activate.bat
echo ✓ Activated

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo ✓ Pip upgraded

echo.
echo [4/5] Installing TensorFlow 1.15...
pip install tensorflow==1.15.0
if errorlevel 1 (
    echo.
    echo ✗ TensorFlow 1.15 failed, trying 1.14...
    pip install tensorflow==1.14.0
)
echo ✓ TensorFlow installed

echo.
echo [5/5] Installing other dependencies...
pip install opencv-python==4.8.1.78
pip install numpy==1.16.4
echo ✓ Dependencies installed

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use this environment:
echo   1. Run: venv_tf1\Scripts\activate.bat
echo   2. Then run your scripts
echo.
echo Or use the provided batch files:
echo   - run_SIMPLE_TF1.bat
echo   - run_WORKING_TF1.bat
echo.
pause
