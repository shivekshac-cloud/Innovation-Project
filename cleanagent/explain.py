"""Explainability for the agent: for every step, WHY it was chosen, the EVIDENCE behind it, how much of the result
it is responsible for, and how much we should trust it.

This is the agent's counterpart of SHAP/LIME. SHAP explains a model's prediction; CleanAgent makes decisions
instead, so we explain each decision: trigger (the issue found), evidence (real examples from the data),
attribution (share of all changed cells), and a plain confidence label with the rule behind it. No numbers are
invented: everything is read from the agent's own log.
"""
from __future__ import annotations

import pandas as pd

# Which issue kind(s) normally trigger which tool; used to pick the matching evidence for a step.
_TRIGGERS = {
    "fix_encoding": ("encoding_artifacts",),
    "standardize_missing": ("disguised_missing",),
    "parse_number": ("number_stored_as_text",),
    "parse_date": ("date_format",),
    "normalize_text": ("text_format",),
    "extract_number": ("mixed_units", "unit_in_text"),
    "map_values": ("category_variants",),
    "fix_range": ("out_of_range",),
    "flag_outliers": ("outliers",),
    "drop_duplicates": ("duplicates",),
}


def _confidence(rec):
    """Plain, rule-based trust label. The rule is shown to the user so it can be checked, not trusted blindly."""
    st_, src = rec["status"], rec["source"]
    if st_ == "rejected":
        return "Blocked", "A guardrail stopped this step: " + str(rec["why"])
    if st_ in ("invalid", "error"):
        return "Refused", "The step was malformed or failed: " + str(rec["why"])
    if rec["n_changed"] == 0:
        return "No effect", "The step ran but changed nothing."
    if rec["rows_dropped"]:
        return "Medium", "Rows were removed (duplicates only); the log keeps them."
    if src == "llm":
        return "Medium", "Suggested by the LLM; accepted only because every answer maps onto a value already in the column."
    if src == "fallback":
        return "Medium", "The LLM reply was unusable, so the rule planner's step was used instead."
    if rec["n_blanked"]:
        return "Medium", "Some values could not be read and were blanked; originals stay in the change log."
    return "High", "Deterministic rule; every change is a reformat that passed all guardrails."


def _matches(tool, kind):
    return kind in _TRIGGERS.get(tool, ()) or (tool.startswith("normalize_") and kind.startswith(tool[len("normalize_"):]))


def explain_steps(res) -> pd.DataFrame:
    """One row per step the agent tried (applied, rejected or refused)."""
    total = max(sum(r["n_changed"] for r in res.log if r["status"] == "applied"), 1)
    out = []
    for r in res.log:
        col = r["column"]
        issues = [i for i in res.issues_before if i["column"] == col]
        match = [i for i in issues if _matches(r["tool"], i["kind"])] or issues
        if r["iteration"] == 1 and match:
            trigger = "; ".join(f"{i['kind']} ({i['count']} cells)" for i in match)
            evidence = " | ".join(str(e) for i in match for e in i["examples"][:2])
        else:
            trigger = "found after re-profiling (iteration %d)" % r["iteration"]
            evidence = " | ".join(f"{o} -> {n}" for _, o, n in r["examples"][:2])
        label, because = _confidence(r)
        out.append(dict(
            step=r["step"], column=col if col is not None else "(whole table)",
            trigger=trigger, evidence=evidence or r["reason"], chosen_by=r["source"],
            decision=r["status"], cells_changed=r["n_changed"],
            share_of_all_changes=round(100.0 * r["n_changed"] / total, 1) if r["status"] == "applied" else 0.0,
            confidence=label, confidence_rule=because))
    return pd.DataFrame(out)


def attribution(res) -> pd.DataFrame:
    """Share of all changed cells by tool: which operations did the work (the pipeline-level analogue of feature importance)."""
    df = explain_steps(res)
    df = df[df.decision == "applied"]
    if df.empty:
        return pd.DataFrame(columns=["tool", "cells_changed", "share_%"])
    df = df.assign(tool=df.step.str.split("(").str[0])
    g = df.groupby("tool", as_index=False).cells_changed.sum()
    g["share_%"] = (100 * g.cells_changed / max(g.cells_changed.sum(), 1)).round(1)
    return g.sort_values("cells_changed", ascending=False).reset_index(drop=True)


def explain_cell(res, row, column) -> list:
    """Every change made to one cell, in order: before, after, tool, reason."""
    return [dict(before=str(o), after=str(n), tool=t, reason=why) for i, c, o, n, t, why in res.changes
            if i == row and c == column]
