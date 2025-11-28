@echo off
echo ========================================
echo Running SIMPLE Classifier with TF1 venv
echo ========================================
echo.

if not exist venv_tf1\Scripts\activate.bat (
    echo ERROR: venv_tf1 not found!
    echo Please run setup_tf1_venv.bat first
    pause
    exit /b 1
)

call venv_tf1\Scripts\activate.bat
python classify_SIMPLE.py

pause
