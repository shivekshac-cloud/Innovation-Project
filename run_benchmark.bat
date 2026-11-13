@echo off
REM Rebuilds every result. Quick synthetic run (~2-3 min on an i3) plus the real-data evaluation (~1 min).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m cleanagent.benchmark --quick
python -m cleanagent.realeval
pause
