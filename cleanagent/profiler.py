"""Observe step: infer what each column is and list the data-quality issues in it."""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .categories import category_mapping
from .knowledge import NAME_HINTS, RANGES
from .parsers import (clean_ws, fingerprint, is_blank, is_missing_token, is_null, is_plain_number, normalize_email,
                      normalize_phone, normalize_pincode, parse_date, parse_number)

PARSERS = {"phone": normalize_phone, "pincode": normalize_pincode, "email": normalize_email, "date": parse_date}


@dataclass
class ColumnProfile:
    name: str
    kind: str
    n: int
    n_null: int
    n_unique: int
    top: list
    case_style: str | None = None


@dataclass
class Issue:
    column: str | None
    kind: str
    count: int
    examples: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(column=self.column, kind=self.kind, count=self.count, examples=[str(e) for e in self.examples[:5]])


def name_tokens(name):
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def name_hint(name):
    for t in name_tokens(name):
        if t in NAME_HINTS:
            return NAME_HINTS[t]
    return None


def column_range(name):
    toks = set(name_tokens(name))
    for keys, rng in RANGES:
        if set(keys) <= toks:
            return rng
    return None


def _rate(values, fn):
    return float(np.mean([not is_null(fn(v)) for v in values])) if len(values) else 0.0


def dominant_case(values, share=0.7):
    vals = [clean_ws(v) for v in values if isinstance(v, str) and any(ch.isalpha() for ch in v)]
    if not vals:
        return None
    for case, f in (("title", str.title), ("lower", str.lower), ("upper", str.upper)):
        if np.mean([v == f(v) for v in vals]) >= share:
            return case
    return None


def infer_kind(name, s):
    vals = s[~s.map(is_blank)]
    if vals.empty:
        return "empty"
    hint = name_hint(name)
    numeric_dtype = pd.api.types.is_numeric_dtype(s)
    sample = vals.sample(min(400, len(vals)), random_state=0).tolist()
    uniq = vals.astype(str).map(fingerprint).nunique() / len(vals)
    if hint == "id" or (name.lower().endswith("id") and uniq > 0.9):
        return "id"
    if numeric_dtype:
        return "money" if hint == "money" else "numeric"
    if hint in PARSERS and _rate(sample, PARSERS[hint]) >= 0.5:
        return hint
    if hint in ("money", "numeric") and _rate(sample, parse_number) >= 0.5:
        return hint
    if hint == "name":
        return "name"
    if _rate(sample, parse_date) >= 0.8:
        return "date"
    if _rate(sample, normalize_email) >= 0.8:
        return "email"
    if _rate(sample, parse_number) >= 0.9:
        return "numeric"
    if vals.astype(str).map(fingerprint).nunique() <= max(50, 0.05 * len(s)):
        return "categorical"
    return "text"


def profile_df(df):
    out = {}
    for c in df.columns:
        s = df[c]
        kind = infer_kind(c, s)
        vals = s[~s.map(is_blank)]
        top = vals.astype(str).value_counts().head(10)
        out[c] = ColumnProfile(c, kind, len(s), int(s.map(is_blank).sum()), int(vals.astype(str).nunique()),
                               list(zip(top.index, top.astype(int))),
                               dominant_case(vals.tolist()) if kind in ("name", "text", "id") else None)
    return out


def _examples(series, mask, k=5):
    return series[mask].astype(str).drop_duplicates().head(k).tolist()


def find_issues(df, prof):
    issues = []
    for c, p in prof.items():
        s = df[c]
        is_text = s.dtype == object or pd.api.types.is_string_dtype(s)
        tok = s.map(is_missing_token)
        if tok.any():
            issues.append(Issue(c, "disguised_missing", int(tok.sum()), _examples(s, tok)))
        present = ~s.map(is_blank)
        if is_text and p.kind in ("money", "numeric"):
            bad = present & ~s.map(is_plain_number)
            issues.append(Issue(c, "number_stored_as_text", int(present.sum()), _examples(s, bad) or _examples(s, present),
                                dict(non_plain=int(bad.sum()))))
        elif is_text and p.kind in PARSERS:
            fn = PARSERS[p.kind]
            norm = s.map(lambda v: fn(v) if not is_blank(v) else v)
            bad = present & (norm.astype(str) != s.astype(str))
            if bad.any():
                issues.append(Issue(c, f"{p.kind}_format", int(bad.sum()), _examples(s, bad)))
        elif is_text and p.kind in ("name", "text", "id"):
            target = p.case_style
            norm = s.map(lambda v: v if is_blank(v) else (getattr(clean_ws(v), target)() if target else clean_ws(v)))
            bad = present & (norm.astype(str) != s.astype(str))
            if bad.any():
                issues.append(Issue(c, "text_format", int(bad.sum()), _examples(s, bad), dict(case=target)))
        elif is_text and p.kind == "categorical":
            mapping, unresolved, canon = category_mapping(s)
            if mapping or unresolved:
                n = int(s.astype(str).map(clean_ws).isin(set(map(clean_ws, mapping)) | set(unresolved)).sum())
                issues.append(Issue(c, "category_variants", n, [f"{k} -> {v[0]}" for k, v in list(mapping.items())[:5]],
                                    dict(mapping=mapping, unresolved=unresolved, canonical=canon)))
        if pd.api.types.is_numeric_dtype(s):
            x = s.astype(float)
            rng = column_range(c) or ((0, None) if p.kind == "money" else None)
            if rng:
                lo, hi = rng
                bad = (x < lo) | ((x > hi) if hi is not None else False)
                if bad.any():
                    issues.append(Issue(c, "out_of_range", int(bad.sum()), _examples(s, bad), dict(lo=lo, hi=hi)))
            if p.kind == "money":
                pos = x[x > 0]
                if len(pos) > 20:
                    lx = np.log(pos)
                    mad = 1.4826 * np.median(np.abs(lx - np.median(lx)))
                    if mad > 0:
                        z = np.abs(np.log(x.where(x > 0)) - np.median(lx)) / mad
                        bad = z > 5
                        if bad.any():
                            issues.append(Issue(c, "outlier", int(bad.sum()), _examples(s, bad), dict(z=5.0)))
    key = next((c for c, p in prof.items() if p.kind == "id"), None)
    subset = [key] if key else list(df.columns)
    fp = df[subset].astype(str).apply(lambda r: "|".join(fingerprint(v) for v in r), axis=1)
    dup = fp.duplicated()
    if dup.any():
        issues.append(Issue(None, "duplicates", int(dup.sum()), [str(i) for i in df.index[dup][:5]], dict(subset=subset)))
    return issues
