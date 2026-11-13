"""Score a cleaned table against the ground truth recorded by the corrupter."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import EMPLOYMENT, NUMERIC, RECOVERABLE, TIER
from .parsers import is_null

CHECK_COLS = [c for c in ["name", "gender", "age", "city", "state", "pincode", "phone", "email", "employment",
                          "monthly_income", "loan_amount", "credit_score", "existing_emis", "application_date"]]


def same(truth, val, numeric):
    if is_null(val) or (isinstance(val, str) and val.strip() == ""):
        return False
    if numeric:
        try:
            return abs(float(val) - float(truth)) <= 0.01 + 1e-9 * abs(float(truth))
        except (TypeError, ValueError):
            return False
    return str(val) == str(truth)


def score(out, clean, tier, ledger, dup_of):
    """out: cleaned DataFrame indexed by rid (rows may be missing)."""
    orig = {r: dup_of.get(r, r) for r in out.index}
    kept = set(out.index)
    # rows
    groups = {}
    for d, o in dup_of.items():
        groups.setdefault(o, [o]).append(d)
    dup_ok = np.mean([sum(r in kept for r in g) == 1 for g in groups.values()]) if groups else np.nan
    singles = [r for r in clean.index if r not in groups]
    entities_kept = sum(r in kept for r in singles) + sum(any(r in kept for r in g) for g in groups.values())
    retention = entities_kept / len(clean)
    # cells
    led = ledger.set_index(["rid", "column"])
    corrupted = set(led.index)
    res = {"cells": [], "by_kind": []}
    rec_hits, rec_n, neut_hits, neut_n, dmg, clean_n, all_hits, all_n = 0, 0, 0, 0, 0, 0, 0, 0
    kind_rows = []
    for rid in out.index:
        o = orig[rid]
        for c in CHECK_COLS:
            num = c in NUMERIC
            t, v = clean.at[o, c], out.at[rid, c]
            ok = same(t, v, num)
            all_hits += ok
            all_n += 1
            if (rid, c) in corrupted:
                row = led.loc[(rid, c)]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                if row.kind in RECOVERABLE:
                    rec_hits += ok
                    rec_n += 1
                    kind_rows.append((row.subtype, ok, int(tier.at[o])))
                else:
                    neutral = ok or is_null(v) or (isinstance(v, str) and v.strip() == "")
                    neut_hits += neutral
                    neut_n += 1
                    kind_rows.append((row.subtype, neutral, int(tier.at[o])))
            else:
                clean_n += 1
                dmg += not ok
    kinds = pd.DataFrame(kind_rows, columns=["subtype", "ok", "tier"])
    tier_kept = {t: float(np.mean([any(r in kept for r in groups.get(i, [i])) for i in clean.index[tier == t]]))
                 for t in (1, 2, 3)}
    # correct cells per entity (first surviving row of each), over ALL entities: deleted rows count as lost
    first = {}
    for rid in out.index:
        first.setdefault(orig[rid], rid)
    usable = 0
    for o, rid in first.items():
        usable += sum(same(clean.at[o, c], out.at[rid, c], c in NUMERIC) for c in CHECK_COLS)
    return dict(cell_accuracy=usable / (len(clean) * len(CHECK_COLS)), kept_row_accuracy=all_hits / max(all_n, 1), recoverable_fixed=rec_hits / max(rec_n, 1),
                unrecoverable_neutralized=neut_hits / max(neut_n, 1), collateral_damage=dmg / max(clean_n, 1),
                retention=retention, duplicate_groups_resolved=float(dup_ok),
                by_subtype=kinds.groupby("subtype").ok.mean().to_dict(),
                fixed_by_tier=kinds[kinds.subtype.isin(RECOVERABLE | {"alias_known", "alias_unknown"})].groupby("tier").ok.mean().to_dict(),
                retention_by_tier=tier_kept)


def features(df):
    X = pd.DataFrame(index=df.index)
    for c in ["age", "monthly_income", "loan_amount", "credit_score", "existing_emis"]:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["female"] = (df["gender"].astype(str) == "Female").astype(float).where(df["gender"].astype(str).isin(["Male", "Female"]))
    for e in EMPLOYMENT:
        X[f"emp_{e}"] = (df["employment"].astype(str) == e).astype(float)
    X["tier"] = df["city"].map(TIER)
    return X


def downstream(out, clean, dup_of, test, test_tier, seed=0):
    """Train a default-risk model on the cleaned data, test on a separate clean set. Returns AUROC overall and by tier."""
    y = np.array([clean.at[dup_of.get(r, r), "defaulted"] for r in out.index])
    if len(out) < 200 or len(set(y)) < 2:
        return dict(auc=float("nan"), auc_by_tier={t: float("nan") for t in (1, 2, 3)})
    Xtr, Xte = features(out), features(test)
    usable = [c for c in Xtr.columns if Xtr[c].notna().sum() >= 20 and Xtr[c].nunique() > 1]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(Xtr[usable], y)
    p = m.predict_proba(Xte[usable])[:, 1]
    yt = test.defaulted.to_numpy()
    return dict(auc=roc_auc_score(yt, p), auc_by_tier={t: roc_auc_score(yt[test_tier == t], p[test_tier == t]) for t in (1, 2, 3)})
