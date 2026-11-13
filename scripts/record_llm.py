"""Run the hybrid agent on every bundled dataset with a live model, saving answers to llm_cache.json.
Afterwards the app can replay them offline ("Use recorded LLM answers")."""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from cleanagent.agent import CleaningAgent  # noqa: E402
from cleanagent.data import corrupt, make_clean  # noqa: E402
from cleanagent.llm import CachedLLM, OpenAICompatLLM  # noqa: E402
from cleanagent.planner import LLMPlanner, RulePlanner  # noqa: E402

llm = CachedLLM(OpenAICompatLLM(), ROOT / "llm_cache.json")
tables = {name: pd.read_csv(ROOT / "data/real" / name, dtype=str, encoding="utf-8").astype(object)
          for name in ("startup_funding.csv", "bengaluru_house.csv", "cardekho_cars.csv")}
clean, tier = make_clean(1500, 7)
tables["synthetic demo (seed 7, 8%)"] = corrupt(clean, tier, rate=0.08, seed=7)[0].reset_index(drop=True)
for name, df in tables.items():
    for label, planner in (("hybrid", RulePlanner(llm=llm)), ("llm", LLMPlanner(llm))):
        t = time.time()
        res = CleaningAgent(planner).run(df)
        print(f"{name:35s} {label:7s} {time.time() - t:6.1f}s  LLM errors: {len(planner.llm_errors)}  "
              f"issues left: {len(res.issues_after)}")
print(f"Saved {len(llm.cache)} answers to llm_cache.json")
