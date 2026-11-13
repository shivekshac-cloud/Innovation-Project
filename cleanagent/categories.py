"""Find spelling, case, spacing and alias variants of the same category value."""
import difflib

import pandas as pd

from .knowledge import VALUE_ALIASES
from .parsers import clean_ws, fingerprint, is_blank


def category_mapping(s: pd.Series, aliases=VALUE_ALIASES, cutoff=0.8):
    """Returns (mapping, unresolved, canonical_values).

    mapping: {raw value: (canonical value, reason)} for every raw value that should change.
    unresolved: rare values that matched nothing (candidates for an LLM to resolve).
    """
    raw = s[~s.map(is_blank)].astype(str)
    if raw.empty:
        return {}, []
    counts = raw.value_counts()
    min_support = max(3, int(0.003 * counts.sum()))
    groups = {}
    for v, c in counts.items():
        groups.setdefault(fingerprint(v), []).append((v, int(c)))
    groups.pop("", None)
    total = {fp: sum(c for _, c in items) for fp, items in groups.items()}
    canon = {}
    for fp, items in groups.items():
        tidy = [(v, c) for v, c in items if v == clean_ws(v)] or items
        canon[fp] = clean_ws(max(tidy, key=lambda t: t[1])[0])
    frequent = [fp for fp in groups if total[fp] >= min_support]
    target, reason = {fp: fp for fp in groups}, {}
    for fp in groups:
        alias = aliases.get(fp)
        if alias and fingerprint(alias) in groups and fingerprint(alias) != fp:
            target[fp], reason[fp] = fingerprint(alias), "known alias"
        elif fp not in frequent and len(fp) >= 4:
            best = difflib.get_close_matches(fp, [f for f in frequent if f != fp], n=1, cutoff=cutoff)
            if best:
                r = difflib.SequenceMatcher(None, fp, best[0]).ratio()
                target[fp], reason[fp] = best[0], f"likely misspelling (similarity {r:.2f})"
    mapping = {}
    for fp, items in groups.items():
        tgt = canon[target[fp]]
        for v, _ in items:
            if v != tgt:
                mapping[v] = (tgt, reason.get(fp, "case or spacing variant"))
    unresolved = sorted(canon[fp] for fp in groups if fp not in frequent and target[fp] == fp)
    canonical_values = sorted(canon[fp] for fp in frequent)
    return mapping, unresolved, canonical_values
