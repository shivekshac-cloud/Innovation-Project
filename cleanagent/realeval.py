"""Evaluation on REAL data (python -m cleanagent.realeval). Writes reports/real/.

1. Startup funding (3,044 real rows): accuracy against 140 hand-labelled gold rows, for no cleaning, a typical
   pandas notebook, a cleaning published by a human analyst on GitHub, and the agent.
2. Bengaluru house prices (13,320 real rows): does cleaning help a price model? Coverage and error on held-out houses.
3. CarDekho used cars (8,128 real rows): blind hold-out, run once with the code frozen (see the frozen report).
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .agent import CleaningAgent
from .parsers import fix_text, is_null
from .planner import RulePlanner

REAL = Path("data/real")
OUT = Path("reports/real")
GOLD_MAP = {"date": "Date dd/mm/yyyy", "city": "City  Location", "investment_type": "InvestmentnType",
            "amount_usd": "Amount in USD"}
HUMAN_MAP = {"date": "date", "city": "city_location", "investment_type": "investment_type", "amount_usd": "amount_usd"}


def read_raw(name):
    return pd.read_csv(REAL / name, dtype=str, encoding="utf-8").astype(object)


def _blank(v):
    return is_null(v) or str(v).strip() in ("", "nan", "NaT")


OLD_NEW = [{"bangalore", "bengaluru"}, {"gurgaon", "gurugram"}, {"new delhi", "delhi"}]


def same_value(gold, val, numeric, lenient=False):
    if _blank(gold):
        return _blank(val)
    if _blank(val):
        return False
    if numeric:
        try:
            return abs(float(str(val).replace(",", "")) - float(gold)) <= 0.5
        except ValueError:
            return False
    a, b = re.sub(r"\s+", " ", fix_text(str(val))).strip().lower(), re.sub(r"\s+", " ", str(gold)).strip().lower()
    if lenient and any(a in grp and b in grp for grp in OLD_NEW):
        return True
    return a == b


def naive_startup(df):
    """A typical notebook: strip commas, coerce numbers and dates, drop rows missing the key fields."""
    d = df.drop_duplicates().copy()
    d["Amount in USD"] = pd.to_numeric(d["Amount in USD"].astype(str).str.replace(",", ""), errors="coerce")
    d["Date dd/mm/yyyy"] = pd.to_datetime(d["Date dd/mm/yyyy"], dayfirst=True, errors="coerce", format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    return d.dropna(subset=["Date dd/mm/yyyy", "City  Location", "Amount in USD"])


def score_startup(table, colmap, gold, key="Sr No"):
    t = table.copy()
    t[key] = t[key].astype(str).str.replace(r"\.0$", "", regex=True)
    t = t.set_index(key)
    per, wrong = {}, []
    for g, c in colmap.items():
        hits = []
        for _, r in gold.iterrows():
            sr = str(r["Sr No"])
            val = t.at[sr, c] if sr in t.index else np.nan
            ok = sr in t.index and same_value(r[g], val, g == "amount_usd")
            hits.append(ok)
            if not ok:
                wrong.append(dict(sr_no=sr, column=g, raw=r[f"raw_{g}"] if f"raw_{g}" in r else "", gold=r[g],
                                  got="(row deleted)" if sr not in t.index else val))
        per[g] = float(np.mean(hits))
        if g == "city":
            per["city_lenient"] = float(np.mean([sr in t.index and same_value(r[g], t.at[sr, c], False, lenient=True)
                                                 for sr, (_, r) in zip(gold["Sr No"].astype(str), gold.iterrows())]))
    per["all"] = float(np.mean([per[g] for g in colmap]))
    per["rows_kept"] = float(gold["Sr No"].astype(str).isin(t.index).mean())
    return per, wrong


def startup_eval():
    raw = read_raw("startup_funding.csv")
    gold = pd.read_csv(REAL / "startup_gold.csv", dtype=str, keep_default_na=False)
    gold = gold.rename(columns={"raw_date": "raw_date", "raw_city": "raw_city"})
    human = pd.read_csv(REAL / "startup_funding_cln.csv", dtype=str).rename(columns={"sr_no": "Sr No"})
    t0 = time.time()
    res = CleaningAgent(RulePlanner()).run(raw)
    secs = time.time() - t0
    agent = res.df
    out, errors = {}, {}
    for name, table, cmap in [("No cleaning", raw, GOLD_MAP), ("Typical pandas notebook", naive_startup(raw), GOLD_MAP),
                              ("Published human cleaning (GitHub)", human, HUMAN_MAP), ("Agent (rule planner)", agent, GOLD_MAP)]:
        out[name], errors[name] = score_startup(table, cmap, gold)
    # whole-file facts (no gold needed)
    city_raw, city_ag = raw["City  Location"], agent["City  Location"]
    facts = {
        "rows": len(raw), "seconds": secs,
        "distinct_cities_before": int(city_raw.dropna().nunique()), "distinct_cities_after": int(city_ag.dropna().nunique()),
        "distinct_rounds_before": int(raw["InvestmentnType"].dropna().nunique()),
        "distinct_rounds_after": int(agent["InvestmentnType"].dropna().nunique()),
        "amounts_usable_before": int(pd.to_numeric(raw["Amount in USD"], errors="coerce").notna().sum()),
        "amounts_usable_after": int(pd.to_numeric(agent["Amount in USD"], errors="coerce").notna().sum()),
        "dates_parsed": int(agent["Date dd/mm/yyyy"].notna().sum()),
        "human_amounts_imputed_with_mean": int(human["amount_usd"].astype(float).round(2).value_counts().iloc[0]),
        "human_mean_value": float(human["amount_usd"].astype(float).round(2).value_counts().index[0]),
        "bengaluru_share_before": float((city_raw.map(lambda v: str(v).strip()) == "Bengaluru").mean()),
        "bengaluru_share_after": float((city_ag == "Bengaluru").mean()),
        "cells_changed": sum(1 for c in res.changes if c[1] is not None),
        "steps": len(res.log), "issues_before": len(res.issues_before), "issues_after": len(res.issues_after),
    }
    pd.DataFrame(errors["Agent (rule planner)"]).to_csv(OUT / "startup_agent_errors.csv", index=False)
    agent.to_csv(OUT / "startup_cleaned_by_agent.csv", index=False, encoding="utf-8")
    from .report import to_markdown
    (OUT / "startup_cleaning_report.md").write_text(to_markdown(res, "Startup funding: cleaning report"), encoding="utf-8")
    return dict(scores=out, facts=facts, agent_errors=errors["Agent (rule planner)"])


# ---------------------------------------------------------------- house prices
def _house_features(df, enc=None):
    X = pd.DataFrame(index=df.index)
    for c in ("total_sqft", "bath", "balcony", "bhk"):
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    X["ready"] = (df["availability"].astype(str).str.strip().str.lower() == "ready to move").astype(float)
    for a in ("super built-up area", "built-up area", "plot area", "carpet area"):
        X[f"type_{a[:5]}"] = (df["area_type"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True) == a).astype(float)
    X["loc"] = df["location"].astype(str).map(enc) if enc is not None else np.nan
    return X


def _loc_encoding(loc, y, m=5):
    g = pd.DataFrame({"loc": loc.astype(str), "y": y}).groupby("loc").y.agg(["mean", "count"])
    prior = y.mean()
    return ((g["mean"] * g["count"] + prior * m) / (g["count"] + m)).to_dict()


def house_versions(raw):
    """Each cleaning method returns a table indexed like raw (rows it cannot use are dropped)."""
    none = raw.copy()
    none["bhk"] = pd.to_numeric(none["size"], errors="coerce")
    nb = raw.dropna(subset=["location", "size", "bath"]).copy()  # the common tutorial recipe
    nb["bhk"] = nb["size"].str.split(" ").str[0].astype(float)
    nb["total_sqft"] = pd.to_numeric(nb["total_sqft"], errors="coerce")
    nb = nb.dropna(subset=["total_sqft"])
    nb["location"] = nb["location"].str.strip()
    t0 = time.time()
    res = CleaningAgent(RulePlanner()).run(raw)
    ag = res.df.rename(columns={"size": "bhk"})
    # the agent never deletes incomplete rows; an analyst may still choose to train only on complete listings
    ag_f = ag.dropna(subset=["location", "bhk", "bath"])
    return {"No cleaning": none, "Typical tutorial recipe": nb, "Agent (rule planner)": ag,
            "Agent + same row filter as tutorial": ag_f}, time.time() - t0, res


def house_eval(seeds=range(5)):
    raw = read_raw("bengaluru_house.csv")
    raw = raw[~raw.duplicated()]  # remove exact duplicates for EVERY method, so no method gains from leakage
    versions, secs, res = house_versions(raw)
    rows = []
    rng_idx = raw.index.to_numpy()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test_idx = set(rng.choice(rng_idx, int(0.2 * len(rng_idx)), replace=False))
        covered = {}
        preds = {}
        for name, d in versions.items():
            d = d[pd.to_numeric(d["price"], errors="coerce").notna()]
            y_all = np.log(pd.to_numeric(d["price"]).astype(float))
            tr = d[~d.index.isin(test_idx)]
            te = d[d.index.isin(test_idx)]
            enc = _loc_encoding(tr["location"], y_all.loc[tr.index])
            Xtr, Xte = _house_features(tr, enc), _house_features(te, enc)
            use = [c for c in Xtr.columns if Xtr[c].notna().sum() >= 20 and Xtr[c].nunique() > 1]  # drop unusable features
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed, early_stopping=False)
            m.fit(Xtr[use], y_all.loc[tr.index])
            p = pd.Series(np.exp(m.predict(Xte[use])), index=te.index)
            preds[name] = p
            covered[name] = set(te.index)
        common = set.intersection(*covered.values())
        only_agent = sorted(covered["Agent + same row filter as tutorial"] - covered["Typical tutorial recipe"])
        truth = pd.to_numeric(raw["price"]).astype(float)
        for name, p in preds.items():
            c = sorted(common)
            rows.append(dict(method=name, seed=seed, coverage=len(covered[name]) / len(test_idx),
                             mae_own=float(np.mean(np.abs(p - truth.loc[p.index]))),
                             mape_own=float(np.mean(np.abs(p - truth.loc[p.index]) / truth.loc[p.index])),
                             mae_common=float(np.mean(np.abs(p.loc[c] - truth.loc[c]))),
                             mape_common=float(np.mean(np.abs(p.loc[c] - truth.loc[c]) / truth.loc[c])),
                             mape_dropped_rows=float(np.mean(np.abs(p.loc[only_agent] - truth.loc[only_agent]) / truth.loc[only_agent]))
                             if name == "Agent + same row filter as tutorial" else float("nan")))
    df = pd.DataFrame(rows)
    summary = {m: {k: [float(g[k].mean()), float(g[k].std())] for k in ("coverage", "mae_own", "mape_own", "mae_common", "mape_common", "mape_dropped_rows")}
               for m, g in df.groupby("method", sort=False)}
    piv = df.pivot(index="seed", columns="method", values="mape_common")
    for a in ("Agent (rule planner)", "Agent + same row filter as tutorial"):
        diff = piv[a] - piv["Typical tutorial recipe"]
        summary[f"paired::{a}"] = [float(diff.mean()), float(diff.std()), int((diff < 0).sum()), len(diff)]
    ag = versions["Agent (rule planner)"]
    sq = raw["total_sqft"].astype(str)
    facts = dict(rows=len(raw), seconds=secs,
                 sqft_ranges=int(sq.str.contains(r"\d\s*-\s*\d").sum()),
                 sqft_units=int(sq.str.contains(r"[A-Za-z]").sum()),
                 sqft_recovered=int((pd.to_numeric(sq, errors="coerce").isna() & ag["total_sqft"].reindex(raw.index).notna()).sum()),
                 locations_before=int(raw["location"].dropna().nunique()),
                 locations_after=int(ag["location"].dropna().nunique()))
    return dict(summary=summary, facts=facts)


def h_cov(house):
    a, t = house["summary"]["Agent + same row filter as tutorial"]["coverage"][0], house["summary"]["Typical tutorial recipe"]["coverage"][0]
    return f"{(a - t) * 100:.1f}% of"


def f(v, pct=False, d=1):
    return f"{v[0] * 100:.{d}f}% ± {v[1] * 100:.{d}f}" if pct else f"{v[0]:.{d}f} ± {v[1]:.{d}f}"


def write(startup, house, holdout):
    L = ["## Real data 1: Indian startup funding (3,044 rows, original Kaggle file)", "",
         "Accuracy on 140 hand-labelled rows (see `data/real/LABELING_GUIDE.md`; 40 of the rows are deliberately hard cases). "
         "A row the method deleted counts as wrong.", "",
         "| Method | Date | City (accepting old names) | Funding round | Amount | All four | Rows kept |", "|---|---|---|---|---|---|---|"]
    for m, s in startup["scores"].items():
        L.append(f"| {m} | {s['date']:.1%} | {s['city']:.1%} ({s['city_lenient']:.0%}) | {s['investment_type']:.1%} | {s['amount_usd']:.1%} | **{s['all']:.1%}** | {s['rows_kept']:.0%} |")
    fa = startup["facts"]
    L += ["", f"Whole file: distinct city spellings {fa['distinct_cities_before']} -> {fa['distinct_cities_after']}, "
          f"funding-round labels {fa['distinct_rounds_before']} -> {fa['distinct_rounds_after']}, usable amounts "
          f"{fa['amounts_usable_before']} -> {fa['amounts_usable_after']}, all {fa['dates_parsed']} dates parsed, "
          f"{fa['cells_changed']} cells changed, {fa['seconds']:.1f} s.",
          f"The published human cleaning filled {fa['human_amounts_imputed_with_mean']} undisclosed amounts with the "
          f"average (${fa['human_mean_value'] / 1e6:.1f}M); the agent leaves them blank.",
          "", "**Read this table with care.** The agent's rules were extended after inspecting this file (escape codes, broken "
          "dates, misspelled cities), so this is a development score, not a blind test. The blind test is the CarDekho file "
          "below. The city score also depends on the naming convention: the human analyst kept old names like Bangalore, "
          "which the bracketed score accepts. 'No cleaning' scores 0% on dates only because they are not in YYYY-MM-DD.",
          "", "Remaining agent errors on the gold rows are listed in `startup_agent_errors.csv`.", "",
          "## Real data 2: Bengaluru house prices (13,320 rows)", "",
          "A price model (gradient boosting on log price) trained on each cleaned version, tested on the same 20% of houses "
          "(5 random splits). Exact duplicate rows were removed for every method first, so no method gains from leakage.", "",
          "| Method | Test houses it can price | Mean abs. error (lakh), own rows | MAPE, own rows | Mean abs. error, rows all methods cover | MAPE, common rows |",
          "|---|---|---|---|---|---|"]
    for m, s in house["summary"].items():
        if m.startswith("paired::"):
            continue
        L.append(f"| {m} | {f(s['coverage'], True)} | {f(s['mae_own'])} | {f(s['mape_own'], True)} | {f(s['mae_common'])} | {f(s['mape_common'], True)} |")
    h = house["facts"]
    d1 = house["summary"]["paired::Agent (rule planner)"]
    d2 = house["summary"]["paired::Agent + same row filter as tutorial"]
    ag = house["summary"]["Agent + same row filter as tutorial"]
    L += ["", "Paired over the 5 splits (MAPE on common rows, minus the tutorial's):", "",
          f"- Agent alone: {d1[0] * 100:+.1f} points (better in {d1[2]} of {d1[3]} splits). **Worse**, and we traced why: the agent "
          "keeps 74 listings with no bathroom or size information, and training on them misleads this model. The converted "
          "unit rows are not the problem (removing them changes nothing).",
          f"- Agent + the tutorial's row filter (drop listings missing location, size or bath): {d2[0] * 100:+.1f} points "
          f"(better in {d2[2]} of {d2[3]} splits), while still pricing the {h_cov(house)} houses the tutorial throws away, "
          f"with MAPE {ag['mape_dropped_rows'][0]:.0%} on those (mostly large plots in acres or ranges).",
          "", "Lesson: fixing values and choosing training rows are separate decisions. The agent does the first and never "
          "deletes data silently; the second belongs to whoever builds the model."]
    L += ["", f"The agent converted {h['sqft_recovered']} areas the tutorial recipe throws away ({h['sqft_ranges']} ranges like "
          f"`1133 - 1384`, {h['sqft_units']} with units like `34.46Sq. Meter` or `5.31Acres`). Location spellings "
          f"{h['locations_before']} -> {h['locations_after']}. {h['seconds']:.1f} s.", "",
          "## Real data 3: CarDekho used cars (8,128 rows), blind hold-out", "",
          f"Run once with the code frozen (fingerprint `{holdout['freeze']}`) before anyone looked at the output. "
          f"Every one of its {holdout['cell_changes']} cell changes was checked by hand: all correct, but all trivial (case and spacing in "
          f"`torque`, e.g. `151NM@ 4850rpm` -> `151Nm@ 4850rpm`). It removed {holdout['dups']} exact duplicate rows. "
          "**It did not convert `23.4 kmpl`, `1248 CC` or `74 bhp` to numbers**, because those units were not in its "
          "dictionary. This is the clearest evidence of the rule-based agent's limits on unseen data.", ""]
    if holdout.get("posthoc"):
        p = holdout["posthoc"]
        L += [f"**After the hold-out (post-hoc fix, not a blind result):** a generic `extract_number` tool now converts a column whose "
              f"values share one unit, and only flags columns that mix units. On this file it converted {', '.join(p['converted'])} "
              f"and flagged {', '.join(p['flagged']) or 'nothing'} for a human or LLM to decide "
              "(mileage mixes `kmpl` and `km/kg`, which are not comparable).", ""]
    (OUT / "REAL_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    startup = startup_eval()
    house = house_eval()
    fr = json.loads((OUT / "cardekho_holdout_frozen_run.json").read_text(encoding="utf-8"))
    ch = pd.read_csv(OUT / "cardekho_holdout_frozen_changes.csv")
    holdout = dict(freeze=fr["freeze"], cell_changes=int(ch.column.notna().sum()), dups=int(ch.column.isna().sum()))
    post = OUT / "cardekho_posthoc.json"
    if post.exists():
        holdout["posthoc"] = json.loads(post.read_text(encoding="utf-8"))
    json.dump(dict(startup=startup, house=house, holdout=holdout), open(OUT / "real_results.json", "w", encoding="utf-8"),
              indent=1, default=str)
    print(write(startup, house, holdout))


if __name__ == "__main__":
    main()
