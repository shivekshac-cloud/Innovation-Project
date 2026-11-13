"""python -m cleanagent messy.csv -o cleaned.csv [--report report.md] [--mode rule|hybrid|llm]"""
import argparse

import pandas as pd

from .agent import CleaningAgent
from .planner import LLMPlanner, RulePlanner
from .report import to_markdown

ap = argparse.ArgumentParser(description="Clean a CSV with the CleanAgent")
ap.add_argument("csv")
ap.add_argument("-o", "--out", default="cleaned.csv")
ap.add_argument("--report", default="cleaning_report.md")
ap.add_argument("--mode", choices=["rule", "hybrid", "llm"], default="rule")
ap.add_argument("--impute", action="store_true")
a = ap.parse_args()
planner = RulePlanner(impute=a.impute)
if a.mode != "rule":
    from .llm import OpenAICompatLLM
    llm = OpenAICompatLLM()
    planner = RulePlanner(llm=llm, impute=a.impute) if a.mode == "hybrid" else LLMPlanner(llm)
res = CleaningAgent(planner, verbose=True).run(pd.read_csv(a.csv, dtype=str, encoding="utf-8").astype(object))
res.df.to_csv(a.out, index=False, encoding="utf-8-sig")  # -sig so Excel on Windows shows ₹ correctly
open(a.report, "w", encoding="utf-8").write(to_markdown(res))
print(f"Wrote {a.out} and {a.report}")
