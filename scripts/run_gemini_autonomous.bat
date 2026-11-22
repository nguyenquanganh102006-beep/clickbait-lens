@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\run_gemini_autonomous.py" %*
