"""Baselines the agent is compared with, and a deliberately faulty planner to test the guardrails."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import NUMERIC
from .planner import RulePlanner, Step


def no_cleaning(df):
    return df


def naive_pandas(df):
    """What a quick notebook often does: drop exact duplicates, coerce numbers, drop rows with any missing value."""
    d = df.drop_duplicates()
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna()


class SinglePassPlanner(RulePlanner):
    """Rule planner that plans once and never re-observes the data (ablation of the agent loop)."""
    name = "single-pass"

    def plan(self, df, prof, issues, done):
        return [] if done else super().plan(df, prof, issues, done)


class FaultyPlanner(RulePlanner):
    """Rule planner plus plausible-looking but wrong steps, like an LLM that misreads the data."""
    name = "faulty"

    def __init__(self, seed=0, n_faults=4, **kw):
        super().__init__(**kw)
        self.rng = np.random.default_rng(seed)
        self.n_faults = n_faults

    def plan(self, df, prof, issues, done):
        steps = super().plan(df, prof, issues, done)
        if done:
            return steps
        cats = [c for c, p in prof.items() if p.kind == "categorical"]
        money = [c for c, p in prof.items() if p.kind == "money"]
        faults = [
            Step("parse_date", {"col": str(self.rng.choice(money)), "dayfirst": True}, "looks like a date", "faulty"),
            Step("parse_number", {"col": str(self.rng.choice(cats))}, "looks numeric", "faulty"),
            Step("drop_duplicates", {"subset": [str(self.rng.choice(cats))]}, "repeated values", "faulty"),
            Step("fix_range", {"col": "age", "lo": 25, "hi": 55}, "plausible working age", "faulty"),
            Step("map_values", {"col": "city", "mapping": {c: "Mumbai" for c in df["city"].dropna().unique()[:25]}},
                 "merge city variants", "faulty"),
            Step("fix_range", {"col": "credit_score", "lo": 600, "hi": 900}, "low scores look wrong", "faulty"),
        ]
        pick = self.rng.choice(len(faults), self.n_faults, replace=False)
        # faults run first, before the correct steps that touch the same columns
        return [faults[i] for i in pick] + steps
