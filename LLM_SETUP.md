# Recording real LLM answers (project owner, optional, about 20 minutes)

Do this on any computer with internet, for example your Mac. The demo laptop then only replays the recorded answers
(`llm_cache.json`), so it needs no model and no internet.

1. Install Ollama from https://ollama.com/download and open it once.
2. In Terminal, inside the project folder:
   ```bash
   ollama pull qwen2.5:1.5b            # about 1 GB; small enough that the i3 laptop could also run it live
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   export CLEANAGENT_LLM_BASE_URL=http://localhost:11434/v1
   export CLEANAGENT_LLM_MODEL=qwen2.5:1.5b
   python scripts/check_llm.py         # should print a reply and a parsed JSON mapping
   python scripts/record_llm.py        # records answers for every bundled dataset -> llm_cache.json
   python -m cleanagent.benchmark --llm --quick   # LLM vs rules -> reports/RESULTS_llm.md
   ```
3. Commit and push `llm_cache.json` and `reports/RESULTS_llm.md`.
4. In the presentation, say which model produced the answers (qwen2.5:1.5b) and that they are replayed.

On Windows the same steps are automated by `record_llm_answers.bat` (needs Ollama for Windows installed first).
