@echo off
cd /d "%~dp0"

echo ==========================================
echo   Data Analysis Agent - Setup
echo ==========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    echo   Download: https://www.python.org/downloads/
    echo   Check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] venv created
) else (
    echo [OK] venv already exists, skipping
)
echo.

echo Installing dependencies (may take 3-5 minutes)...
set "ACTIVATE=venv\Scripts\activate.bat"
call %ACTIVATE%
pip install -r requirements.txt -q
echo.
echo [OK] Dependencies installed!
echo.
echo ==========================================
echo       Setup complete!
echo ==========================================
echo   Start:       run_server.bat
echo   Open page:   chat.html
echo ==========================================
echo.
pause
