@echo off
setlocal
cd /d "%~dp0\.."

if not exist .venv\Scripts\python.exe (
    echo Environment not found. Run scripts\setup_windows.bat first.
    exit /b 1
)

.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
