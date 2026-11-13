"""Score the recorded LLM answers (llm_cache.json) on the REAL data, with no model running.

    python scripts/llm_real_eval.py   ->  reports/real/LLM_REAL_RESULTS.md
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)
import pandas as pd  # noqa: E402

from cleanagent.agent import CleaningAgent  # noqa: E402
from cleanagent.llm import CachedLLM  # noqa: E402
from cleanagent.planner import LLMPlanner, RulePlanner  # noqa: E402
from cleanagent.realeval import GOLD_MAP, score_startup  # noqa: E402

MODEL = "gpt-oss-20b (LM Studio, M5 MacBook Air)"


def run(df, kind):
    llm = CachedLLM(None, ROOT / "llm_cache.json", mode="replay")
    planner = {"rules": RulePlanner(), "hybrid": RulePlanner(llm=llm), "llm": LLMPlanner(llm)}[kind]
    res = CleaningAgent(planner).run(df)
    return res, planner, llm


def llm_mappings(res):
    out = []
    for c in res.changes:
        if c[1] is not None and str(c[5]).startswith("LLM"):
            out.append((c[1], str(c[2]), str(c[3])))
    return Counter(out)


def main():
    gold = pd.read_csv("data/real/startup_gold.csv", dtype=str, keep_default_na=False)
    L = [f"## LLM results on real data ({MODEL})", "",
         "Recorded answers replayed from `llm_cache.json`, so these numbers are reproducible without the model.", ""]
    rows = []
    for name in ("startup_funding.csv", "bengaluru_house.csv", "cardekho_cars.csv"):
        df = pd.read_csv(f"data/real/{name}", dtype=str, encoding="utf-8").astype(object)
        for kind in ("rules", "hybrid"):
            res, planner, llm = run(df, kind)
            log = pd.DataFrame(res.log)
            st = log.status.value_counts().to_dict() if len(log) else {}
            r = dict(dataset=name, planner=kind, steps_applied=st.get("applied", 0),
                     steps_refused=st.get("invalid", 0) + st.get("rejected", 0) + st.get("error", 0),
                     llm_failures=len(planner.llm_errors), issues_left=len(res.issues_after),
                     cells_changed=sum(1 for c in res.changes if c[1] is not None))
            if name == "startup_funding.csv":
                sc, _ = score_startup(res.df, GOLD_MAP, gold)
                r.update(gold_all=sc["all"], gold_city=sc["city"], gold_round=sc["investment_type"])
            if kind == "hybrid":
                maps = llm_mappings(res)
                r["llm_merges"] = maps
            rows.append(r)
    L += ["| Dataset | Planner | Steps applied | Steps refused by validation/guardrails | LLM questions without a usable answer | Cells changed | Issues left | Gold accuracy (startup, 140 rows) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = f"{r['gold_all']:.1%}" if "gold_all" in r else ""
        L.append(f"| {r['dataset'].split('.')[0]} | {r['planner']} | {r['steps_applied']} | {r['steps_refused']} | "
                 f"{r['llm_failures']} | {r['cells_changed']} | {r['issues_left']} | {g} |")
    L += ["", "On the startup gold rows, hybrid mode scores slightly lower than rules alone: the LLM merged distinct "
          "funding-round labels that the labelling rules keep apart (some defensible, like `Seed Round` → `Seed Funding`; "
          "some wrong, like `Crowd funding` → `Debt Funding`). Its useful contribution is world knowledge the rules lack "
          "(`Kormangala` → `Bengaluru`; on synthetic data, old city names like `Poona` and `Cochin`).", "",
          "Development notes (first version vs now): the first hybrid version put LLM merges in the same step as rule "
          "merges, so the guardrail rejected the whole city step (91.2%); it also accepted LLM merges that collapse "
          "multi-city entries (`Bangalore / USA` → `Bangalore`). LLM suggestions are now a separate step and obey the same "
          "safety rules (98.8%).", "", "### What the LLM changed in hybrid mode (accepted answers only)", ""]
    for r in rows:
        if r["planner"] == "hybrid":
            m = r["llm_merges"]
            L.append(f"**{r['dataset'].split('.')[0]}:** " + (", ".join(f"`{a}` → `{b}` ({n})" for (c, a, b), n in m.most_common(25))
                                                                 if m else "no merges (nothing the rules left unresolved was a clear variant)"))
            L.append("")
    L += ["### LLM planning everything by itself (original recorded run on the Mac)", "",
          "Measured live while recording; not replayable from the cache because of a prompt-ordering bug fixed afterwards.", "",
          "| Dataset | Time | Unusable replies | Issues left (LLM alone) | Issues left (rules) |", "|---|---|---|---|---|",
          "| startup_funding | 90 s | 1 | 4 | 4 |", "| bengaluru_house | 28 s | 0 | 4 | 1 |",
          "| cardekho_cars | 74 s | 0 | 2 | 2 |", "| synthetic demo (1,500 rows) | 8 s | 0 | 26 | 1 |", "",
          "The model alone planned few useful steps and left more problems than the rules; on the synthetic benchmark it "
          "fixed only about 10% of fixable errors (see `reports/RESULTS_llm.md`). It never damaged data, because every "
          "step it proposed went through validation and the guardrails.", ""]
    out = ROOT / "reports/real/LLM_REAL_RESULTS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
