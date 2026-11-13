"""The agent loop: observe (profile) -> plan -> act (whitelisted tool) -> verify (guardrails) -> repeat."""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd

from .planner import validate
from .profiler import find_issues, profile_df
from .tools import diff, run_tool


@dataclass
class Guard:
    """Reject a step that destroys too much: drops many rows, blanks many values, or rewrites most of a column."""
    max_row_drop: float = 0.15
    max_nullify: float = 0.25
    max_change: float = 0.5

    def check(self, before, step, info):
        if info.rows_dropped > self.max_row_drop * len(before):
            return f"would drop {info.rows_dropped} of {len(before)} rows"
        if info.n_nonnull_before and info.n_nullified > self.max_nullify * info.n_nonnull_before:
            return f"would blank {info.n_nullified} of {info.n_nonnull_before} values"
        if step.tool in ("map_values", "normalize_text") and info.n_nonnull_before and \
                info.n_rewritten > self.max_change * info.n_nonnull_before:
            return f"would rewrite {info.n_rewritten} of {info.n_nonnull_before} values"
        return None


@dataclass
class Result:
    df: pd.DataFrame
    log: list = field(default_factory=list)
    issues_before: list = field(default_factory=list)
    issues_after: list = field(default_factory=list)
    changes: list = field(default_factory=list)


class CleaningAgent:
    def __init__(self, planner, max_iters=4, guard=Guard(), verbose=False):
        self.planner, self.max_iters, self.guard, self.verbose = planner, max_iters, guard, verbose

    def run(self, df):
        df = df.copy()
        res = Result(df)
        done = set()
        for it in range(self.max_iters):
            prof = profile_df(df)
            issues = find_issues(df, prof)
            if it == 0:
                res.issues_before = [i.as_dict() for i in issues]
            steps = self.planner.plan(df, prof, issues, done)
            if not steps:
                break
            for step in steps:
                rec = dict(iteration=it + 1, step=step.label(), tool=step.tool, column=step.args.get("col"),
                           reason=step.reason, source=step.source, status="applied", why="", n_changed=0,
                           n_blanked=0, rows_dropped=0, examples=[])
                err = validate(step, df)
                if err:
                    rec.update(status="invalid", why=err)
                    res.log.append(rec)
                    done.add(step.key())
                    continue
                try:
                    new = run_tool(df, step.tool, step.args)
                except Exception as e:  # noqa: BLE001
                    rec.update(status="error", why=str(e))
                    res.log.append(rec)
                    done.add(step.key())
                    continue
                info = diff(df, new, step.tool, step.args)
                rec.update(n_changed=info.n_changed, n_blanked=info.n_nullified, rows_dropped=info.rows_dropped,
                           examples=[(str(i), str(o), str(n)) for i, o, n in info.examples])
                why = self.guard.check(df, step, info) if self.guard else None
                done.add(step.key())
                if why:
                    rec.update(status="rejected", why=why)
                else:
                    reasons = step.meta.get("reasons", {})
                    c = step.args.get("col")
                    for i in info.changed_index:
                        if c is None:
                            res.changes.append((i, None, "row", "dropped", step.tool, step.reason))
                        else:
                            old = df.at[i, c]
                            res.changes.append((i, c, old, new.at[i, c], step.tool, reasons.get(str(old), step.reason)))
                    df = new
                res.log.append(rec)
                if self.verbose:
                    print(f"[{it + 1}] {rec['status']:8s} {step.label()} changed={info.n_changed} {rec['why']}")
        res.df = df
        res.issues_after = [i.as_dict() for i in find_issues(df, profile_df(df))]
        return res
