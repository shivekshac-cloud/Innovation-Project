@echo off
REM Run ONCE with Ollama installed (https://ollama.com) to record LLM answers for the offline demo.
REM Then commit llm_cache.json. During the presentation tick "Use recorded LLM answers" in the app.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set CLEANAGENT_LLM_BASE_URL=http://localhost:11434/v1
set CLEANAGENT_LLM_MODEL=qwen2.5:1.5b
ollama pull qwen2.5:1.5b
python scripts\check_llm.py
python scripts\record_llm.py
python -m cleanagent.benchmark --llm --quick
pause
