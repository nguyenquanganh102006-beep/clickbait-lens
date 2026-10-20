@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    call scripts\setup_windows.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
call scripts\run_windows.bat
if errorlevel 1 pause
