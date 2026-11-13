"""python -m cleanagent.benchmark          rule-based agent vs baselines (offline, about 2 minutes)
python -m cleanagent.benchmark --llm    also run the LLM and hybrid planners (needs an LLM server, see llm.py)"""
import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agent import CleaningAgent
from .baselines import FaultyPlanner, SinglePassPlanner, naive_pandas, no_cleaning
from .data import corrupt, make_clean
from .planner import LLMPlanner, RulePlanner
from .score import downstream, score

SEEDS = range(5)
RATE = 0.06
SWEEP = [0.03, 0.06, 0.12, 0.2, 0.3]
SUBTYPE_NAMES = {"case_space": "Case / extra spaces", "variant": "Category spelling variants", "num_text": "Number with unit ('34 yrs')",
                 "alias_known": "Old city name, in rule dictionary", "alias_unknown": "Old city name, NOT in rule dictionary",
                 "typo": "City misspelling", "code": "State code ('MH')", "pin_format": "PIN code format",
                 "phone_format": "Phone format (+91, 0, dashes)", "money_format": "Money format (₹1,20,000 / 1.2 lakh / 120k)",
                 "date_format": "Date format (day-first variants)", "missing": "Disguised missing (NA, -, ?)",
                 "outlier": "Extra-zero outlier", "sentinel": "Sentinel credit score (9999, 0, -1)"}


def methods(seed, llm=None):
    m = {"No cleaning": no_cleaning, "Naive pandas (dropna)": naive_pandas,
         "Agent, single pass": lambda d: CleaningAgent(SinglePassPlanner()).run(d).df,
         "Agent (rule planner)": lambda d: CleaningAgent(RulePlanner()).run(d).df,
         "Faulty planner + guardrails": lambda d: CleaningAgent(FaultyPlanner(seed)).run(d).df,
         "Faulty planner, no guardrails": lambda d: CleaningAgent(FaultyPlanner(seed), guard=None).run(d).df}
    if llm:
        m["Agent (hybrid: rules + LLM)"] = lambda d: CleaningAgent(RulePlanner(llm=llm)).run(d).df
        m["Agent (LLM planner)"] = lambda d: CleaningAgent(LLMPlanner(llm)).run(d).df
    return m


def ms(v):
    v = [x for x in v if x == x]
    return [float(np.mean(v)), float(np.std(v))] if v else [float("nan"), float("nan")]


def main(use_llm=False):
    llm = None
    if use_llm:
        from .llm import OpenAICompatLLM
        llm = OpenAICompatLLM()
    test, test_tier = make_clean(3000, 1000, id_offset=90000)
    tt = test_tier.to_numpy()
    rows, subtype_rows, tier_rows, timing = [], [], [], {}
    clean_auc = []
    for seed in SEEDS:
        clean, tier = make_clean(3000, seed)
        clean_auc.append(downstream(clean, clean, {}, test, tt, seed)["auc"])
        dirty, ledger, dup_of = corrupt(clean, tier, rate=RATE, seed=seed)
        for name, fn in methods(seed, llm).items():
            t0 = time.time()
            out = fn(dirty)
            timing.setdefault(name, []).append(time.time() - t0)
            s = score(out, clean, tier, ledger, dup_of)
            d = downstream(out, clean, dup_of, test, tt, seed)
            rows.append(dict(method=name, seed=seed, auc=d["auc"], **{k: v for k, v in s.items() if isinstance(v, float)}))
            for t in (1, 2, 3):
                tier_rows.append(dict(method=name, seed=seed, tier=t, retention=s["retention_by_tier"][t],
                                      fixed=s["fixed_by_tier"].get(t, np.nan), auc=d["auc_by_tier"][t]))
            if name in ("Agent (rule planner)", "Agent (hybrid: rules + LLM)", "Agent (LLM planner)"):
                for k, v in s["by_subtype"].items():
                    subtype_rows.append(dict(method=name, seed=seed, subtype=k, ok=v))
    df = pd.DataFrame(rows)
    sweep = []
    for rate in SWEEP:
        for seed in range(3):
            clean, tier = make_clean(3000, seed)
            dirty, ledger, dup_of = corrupt(clean, tier, rate=rate, seed=seed)
            for name in ("No cleaning", "Naive pandas (dropna)", "Agent (rule planner)"):
                out = methods(seed)[name](dirty)
                s = score(out, clean, tier, ledger, dup_of)
                sweep.append(dict(rate=rate, method=name, cell_accuracy=s["cell_accuracy"], retention=s["retention"],
                                  auc=downstream(out, clean, dup_of, test, tt, seed)["auc"]))
    sw = pd.DataFrame(sweep)
    metrics = ["cell_accuracy", "recoverable_fixed", "unrecoverable_neutralized", "collateral_damage", "retention",
               "duplicate_groups_resolved", "auc"]
    res = dict(settings=dict(seeds=len(SEEDS), rows=3000, error_rate=RATE, tier3_error_multiplier=2.0, duplicate_rate=0.03,
                             llm=bool(llm), llm_model=getattr(llm, "model", None)),
               clean_data_auc=ms(clean_auc),
               methods={m: {k: ms(g[k]) for k in metrics} | {"seconds": ms(timing[m])} for m, g in df.groupby("method", sort=False)},
               by_subtype={m: {k: ms(g.ok) for k, g in d.groupby("subtype")} for m, d in pd.DataFrame(subtype_rows).groupby("method")},
               by_tier={m: {int(t): {k: ms(g[k]) for k in ("retention", "fixed", "auc")} for t, g in d.groupby("tier")}
                        for m, d in pd.DataFrame(tier_rows).groupby("method", sort=False)},
               sweep={m: {str(r): {k: ms(g[k]) for k in ("cell_accuracy", "retention", "auc")} for r, g in d.groupby("rate")}
                      for m, d in sw.groupby("method", sort=False)})
    suffix = "_llm" if llm else ""
    json.dump(res, open(f"reports/results{suffix}.json", "w"), indent=2)
    write_markdown(res, f"reports/RESULTS{suffix}.md")
    if not llm:
        make_plots(res)
    print(open(f"reports/RESULTS{suffix}.md").read())


def f(v, pct=True):
    if v[0] != v[0]:
        return "n/a"
    return f"{v[0] * 100:.1f}% ± {v[1] * 100:.1f}" if pct else f"{v[0]:.3f} ± {v[1]:.3f}"


def write_markdown(res, path):
    s = res["settings"]
    L = [f"### Main comparison ({s['rows']} applications, {s['error_rate']:.0%} cell error rate, 3% near-duplicates; "
         f"mean ± std over {s['seeds']} datasets)\n",
         "| Method | Correct cells (all applicants) | Fixable errors fixed | Bad values blanked | Clean cells damaged | Applicants kept | Duplicates resolved | Downstream AUROC |",
         "|---|---|---|---|---|---|---|---|"]
    for m, v in res["methods"].items():
        L.append(f"| {m} | {f(v['cell_accuracy'])} | {f(v['recoverable_fixed'])} | {f(v['unrecoverable_neutralized'])} | "
                 f"{f(v['collateral_damage'])} | {f(v['retention'])} | {f(v['duplicate_groups_resolved'])} | {f(v['auc'], False)} |")
    L.append(f"\nAUROC of the same model trained on the original clean data: {f(res['clean_data_auc'], False)}.")
    for m, d in res["by_subtype"].items():
        L += ["", f"### Errors fixed by type: {m}\n", "| Error type | Fixed or correctly blanked |", "|---|---|"]
        for k in SUBTYPE_NAMES:
            if k in d:
                L.append(f"| {SUBTYPE_NAMES[k]} | {f(d[k])} |")
    L += ["", "### Fairness: who loses data? (tier-3 towns have 2x the error rate in this simulation)\n",
          "| Method | Kept: tier 1 | Kept: tier 2 | Kept: tier 3 | Fixed: tier 1 | Fixed: tier 3 |", "|---|---|---|---|---|---|"]
    for m, d in res["by_tier"].items():
        if m.startswith("Faulty"):
            continue
        L.append(f"| {m} | {f(d[1]['retention'])} | {f(d[2]['retention'])} | {f(d[3]['retention'])} | {f(d[1]['fixed'])} | {f(d[3]['fixed'])} |")
    L += ["", "### Error-rate sweep (3 datasets per rate)\n", "| Error rate | " + " | ".join(f"{m}: correct cells" for m in res["sweep"]) + " | Naive: applicants kept |",
          "|---|" + "---|" * (len(res["sweep"]) + 1)]
    for r in [str(x) for x in SWEEP]:
        L.append(f"| {float(r):.0%} | " + " | ".join(f(res['sweep'][m][r]['cell_accuracy']) for m in res["sweep"]) +
                 f" | {f(res['sweep']['Naive pandas (dropna)'][r]['retention'])} |")
    L += ["", "| Seconds per run | " + " | ".join(f"{m}: {v['seconds'][0]:.1f}" for m, v in res["methods"].items()) + " |"]
    open(path, "w").write("\n".join(L) + "\n")


def make_plots(res):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    show = ["No cleaning", "Naive pandas (dropna)", "Agent, single pass", "Agent (rule planner)"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    v = [res["methods"][m]["cell_accuracy"] for m in show]
    ax.barh(show[::-1], [x[0] * 100 for x in v[::-1]], xerr=[x[1] * 100 for x in v[::-1]], color=["#2e86c1"] * 3 + ["#7f8c8d"])
    for i, x in enumerate(v[::-1]):
        ax.text(x[0] * 100 + 1, i, f"{x[0] * 100:.1f}%", va="center")
    ax.set(xlim=(0, 110), xlabel="Correct cells over all applicants (%)", title="How much of the true data each method recovers")
    fig.tight_layout(); fig.savefig("reports/cell_accuracy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m, d in res["sweep"].items():
        rates = [float(r) for r in d]
        ax.errorbar([r * 100 for r in rates], [d[r]["cell_accuracy"][0] * 100 for r in d],
                    yerr=[d[r]["cell_accuracy"][1] * 100 for r in d], marker="o", capsize=3, label=m)
    ax.set(xlabel="Share of cells corrupted (%)", ylabel="Correct cells (%)", title="Robustness as data gets dirtier", ylim=(0, 102))
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("reports/error_rate_sweep.png", dpi=150); plt.close(fig)

    d = res["by_subtype"]["Agent (rule planner)"]
    keys = [k for k in SUBTYPE_NAMES if k in d]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    vals = [d[k][0] * 100 for k in keys]
    ax.barh([SUBTYPE_NAMES[k] for k in keys][::-1], vals[::-1], color=["#c0392b" if x < 90 else "#2e86c1" for x in vals[::-1]])
    for i, x in enumerate(vals[::-1]):
        ax.text(x + 1, i, f"{x:.0f}%", va="center", fontsize=8, color="#c0392b" if x < 90 else "black")
    ax.set(xlim=(0, 112), xlabel="Fixed or correctly blanked (%)", title="Rule-based agent, by error type (red = gap an LLM could fill)")
    fig.tight_layout(); fig.savefig("reports/by_error_type.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    show2 = ["Naive pandas (dropna)", "Agent (rule planner)"]
    x = np.arange(3)
    for i, m in enumerate(show2):
        ax.bar(x + (i - .5) * .38, [res["by_tier"][m][t]["retention"][0] * 100 for t in (1, 2, 3)], .38, label=m)
    ax.set_xticks(x, ["Tier 1 cities", "Tier 2 cities", "Tier 3 towns"])
    ax.set(ylabel="Applicants kept (%)", ylim=(0, 105), title="Dropping dirty rows silently removes small-town applicants")
    ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig("reports/fairness_retention.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    show3 = ["Agent (rule planner)", "Faulty planner + guardrails", "Faulty planner, no guardrails"]
    v = [res["methods"][m]["collateral_damage"] for m in show3]
    ax.bar(range(3), [a[0] * 100 for a in v], yerr=[a[1] * 100 for a in v], capsize=3, color=["#2e86c1", "#27ae60", "#c0392b"])
    ax.set_xticks(range(3), ["Correct planner", "Faulty + guardrails", "Faulty, no guardrails"])
    ax.set(ylabel="Clean cells damaged (%)", title="Guardrails contain a planner's mistakes")
    fig.tight_layout(); fig.savefig("reports/guardrails.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also benchmark LLM and hybrid planners")
    main(ap.parse_args().llm)
