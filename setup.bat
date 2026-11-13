@echo off
REM One-time setup on Windows. Needs Python 3.10 or newer from python.org ("Add python.exe to PATH" ticked).
cd /d "%~dp0"
python --version || (echo Python not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH". & pause & exit /b 1)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (echo Python 3.10 or newer is needed. & pause & exit /b 1)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
echo.
echo Setup done. Double-click run_app.bat to start the demo.
pause
