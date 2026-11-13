"""Find spelling, case, spacing and alias variants of the same category value."""
from __future__ import annotations
import difflib
import re

import pandas as pd

from .knowledge import VALUE_ALIASES
from .parsers import clean_ws, fingerprint, fix_text, is_blank

MULTI = re.compile(r"/|&|\band\b|,\s*\w|;")


def _tokens(v):
    return [t for t in re.split(r"[^a-z0-9]+", fix_text(str(v)).lower()) if t]


def typo_compatible(a, b):
    """True if a and b differ only by misspellings INSIDE words. Rejects a different code letter or number
    ('Series E' vs 'Series D', '21-Jan' vs '21-Jun'), an added or missing word ('Seed Funding Round'), and a value
    listing several things being matched to one ('Bangalore / USA' -> 'Bangalore')."""
    if MULTI.search(a) and not MULTI.search(b):
        return False
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) != len(tb):
        # allow a split or joined word ('Private Equity' vs 'PrivateEquity'), nothing else
        return "".join(ta) == "".join(tb)
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if max(len(x), len(y)) < 3 or x.isdigit() or y.isdigit():
            return False
        if difflib.SequenceMatcher(None, x, y).ratio() < 0.75:
            return False
    return True


def category_mapping(s: pd.Series, aliases=VALUE_ALIASES, cutoff=0.8, fuzzy=True):
    """Returns (mapping, unresolved, canonical_values).

    mapping: {raw value: (canonical value, reason)} for every raw value that should change.
    unresolved: rare values that matched nothing (candidates for an LLM to resolve).
    Safety rules: only rare values are fuzzy-matched, and only onto values at least twice as common (so 'Female' is
    never matched to 'Male'); never match values whose digits differ ('Phase 1' vs 'Phase 2') or values that list
    several things ('Bangalore / USA').
    """
    raw = s[~s.map(is_blank)].astype(str)
    if raw.empty:
        return {}, [], []
    counts = raw.value_counts()
    min_support = max(3, int(0.003 * counts.sum()))
    groups = {}
    for v, c in counts.items():
        groups.setdefault(fingerprint(fix_text(v)), []).append((v, int(c)))
    groups.pop("", None)
    total = {fp: sum(c for _, c in items) for fp, items in groups.items()}
    canon = {}
    for fp, items in groups.items():
        tidy = [(v, c) for v, c in items if v == clean_ws(v) and fix_text(v) == v] or items
        canon[fp] = clean_ws(fix_text(max(tidy, key=lambda t: (t[1], t[0]))[0]))
    frequent = [fp for fp in groups if total[fp] >= min_support]
    target, reason = {fp: fp for fp in groups}, {}
    for fp in groups:
        alias = aliases.get(fp)
        if alias and fingerprint(alias) in groups and fingerprint(alias) != fp:
            target[fp], reason[fp] = fingerprint(alias), "known alias"
        elif fuzzy and fp not in frequent and len(fp) >= 4:
            cands = [f for f in groups if f != fp and total[f] >= max(3, 2 * total[fp])
                     and typo_compatible(canon[fp], canon[f])]
            best = difflib.get_close_matches(fp, cands, n=1, cutoff=cutoff)
            if best:
                r = difflib.SequenceMatcher(None, fp, best[0]).ratio()
                target[fp], reason[fp] = best[0], f"likely misspelling (similarity {r:.2f})"
    for fp in groups:  # follow chains like 'Nw Delhi' -> 'New Delhi' -> 'Delhi'
        seen = {fp}
        while target[target[fp]] != target[fp] and target[fp] not in seen:
            seen.add(target[fp])
            target[fp] = target[target[fp]]
    mapping = {}
    for fp, items in groups.items():
        tgt = canon[target[fp]]
        for v, _ in items:
            if v != tgt:
                mapping[v] = (tgt, reason.get(fp, "case, spacing or encoding variant"))
    unresolved = sorted(canon[fp] for fp in groups if fp not in frequent and target[fp] == fp)
    canonical_values = sorted(canon[fp] for fp in frequent)
    return mapping, unresolved, canonical_values
