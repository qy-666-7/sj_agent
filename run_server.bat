@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [WARN] Not initialized. Please run setup.bat first.
    pause
    exit /b 1
)

echo Starting API server...
echo URL: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo.
call venv\Scripts\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
pause
