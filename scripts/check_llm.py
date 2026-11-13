"""Check that a local LLM server answers and how fast it is on this laptop.

    python scripts/check_llm.py                      # uses CLEANAGENT_LLM_BASE_URL / CLEANAGENT_LLM_MODEL
    python scripts/check_llm.py --model qwen2.5:1.5b --url http://localhost:11434/v1
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanagent.llm import OpenAICompatLLM, extract_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url")
ap.add_argument("--model")
a = ap.parse_args()
llm = OpenAICompatLLM(a.url, a.model)
print(f"Asking {llm.model} at {llm.base_url} ...")
prompt = ('Canonical values: ["Pune", "Kochi", "Bhubaneswar", "Seed/ Angel Funding"]. Rare values: ["Poona", "Cochin", '
          '"Bhubneswar", "Angel / Seed Funding", "Nairobi"]. Map each rare value that is an alias, old name or misspelling '
          'of a canonical value. Leave out the rest. Reply with JSON only: {"mapping": {"rare": "canonical"}}')
t = time.time()
try:
    reply = llm.complete([{"role": "user", "content": prompt}])
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not reach the model: {e}\nIs Ollama / LM Studio running, and is the model name right?")
print(f"Reply in {time.time() - t:.1f} s:\n{reply}\n")
try:
    print("Parsed:", extract_json(reply))
except ValueError as e:
    print("Could not parse JSON from the reply:", e, "(the agent will fall back to rules)")
