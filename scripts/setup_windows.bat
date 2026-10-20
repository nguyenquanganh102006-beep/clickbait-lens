@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON_EXE="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))"
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)

rem A new Python install may not be visible to an already-open terminal yet.
if not defined PYTHON_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
        if exist "%%~D\python.exe" (
            "%%~D\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 11))"
            if not errorlevel 1 set "PYTHON_EXE=%%~D\python.exe"
        )
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(sys.version_info < (3, 11))"
        if not errorlevel 1 set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo Python 3.11+ was not found. Install Python and try again.
    exit /b 1
)

if "%PYTHON_EXE%"=="py -3" (
    py -3 -m venv .venv
) else if "%PYTHON_EXE%"=="python" (
    python -m venv .venv
) else (
    "%PYTHON_EXE%" -m venv .venv
)
if errorlevel 1 (
    echo Could not create .venv. Make sure Python 3.11+ is installed.
    exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Setup complete. Run scripts\run_windows.bat to start the app.
