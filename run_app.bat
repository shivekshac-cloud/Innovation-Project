@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m streamlit run app\streamlit_app.py
