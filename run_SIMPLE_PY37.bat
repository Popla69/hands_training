@echo off
echo ========================================
echo Running SIMPLE with Python 3.7 + TF 1.15
echo ========================================
echo.

if not exist venv_py37\Scripts\activate.bat (
    echo ERROR: Python 3.7 venv not found!
    echo Please run setup_python37_venv.bat first
    pause
    exit /b 1
)

call venv_py37\Scripts\activate.bat
python classify_SIMPLE.py

pause
