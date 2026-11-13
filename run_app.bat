@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
echo Starting CleanAgent. Your browser will open in a few seconds.
echo KEEP THIS WINDOW OPEN during the demo. Close it to stop the app.
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
python -m streamlit run app\streamlit_app.py
pause
