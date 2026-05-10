@echo off
setlocal
chcp 65001 >nul
echo ====================================
echo    Agentic RAG System - Web Mode
echo ====================================
echo.

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please activate the virtual environment first.
    pause
    exit /b 1
)

echo Installing/updating dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Checking config...
if not exist "config.json" (
    echo WARNING: config.json was not found.
    echo Copy config.json.template to config.json or set DEEPSEEK_API_KEY.
    set /p choice="Continue startup? [y/N]: "
    if /i not "%choice%"=="y" exit /b 1
)

echo.
echo Starting web server...
echo URL:  http://127.0.0.1:5000
echo Chat: http://127.0.0.1:5000/chat
echo.
echo Press Ctrl+C to stop the server.
echo ====================================

python app.py

echo.
echo Server stopped.
pause
