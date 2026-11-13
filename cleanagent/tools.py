"""Act step: the only operations the agent (rule-based or LLM) is allowed to run. No free-form code."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .parsers import (apply_case, fix_text, fingerprint, is_blank, is_missing_token, is_null, is_plain_number, normalize_email, normalize_phone,
                      normalize_pincode, parse_date, parse_number)

# tool -> {arg: type}. Used to validate LLM output.
TOOLS = {
    "fix_encoding": {"col": str},
    "standardize_missing": {"col": str},
    "normalize_text": {"col": str, "case": (str, type(None))},
    "parse_number": {"col": str},
    "parse_date": {"col": str, "dayfirst": bool},
    "normalize_phone": {"col": str},
    "normalize_pincode": {"col": str},
    "normalize_email": {"col": str},
    "map_values": {"col": str, "mapping": dict},
    "extract_number": {"col": str, "unit": str},
    "fix_range": {"col": str, "lo": (int, float, type(None)), "hi": (int, float, type(None))},
    "flag_outliers": {"col": str, "z": (int, float), "above": (int, float)},
    "drop_duplicates": {"subset": list},
    "impute": {"col": str, "strategy": str},
}
OPTIONAL = {"normalize_text": {"case"}, "parse_date": {"dayfirst"}, "fix_range": {"lo", "hi"}, "flag_outliers": {"z", "above"}}
TOOL_DOCS = {
    "fix_encoding": "repair export damage: literal \\xc2\\xa0 escapes, literal \\n, non-breaking spaces, mojibake",
    "standardize_missing": "turn placeholders like NA, -, ?, null, 'not available' into real missing values",
    "normalize_text": "trim/collapse spaces and set case ('title', 'lower', 'upper' or null)",
    "parse_number": "convert text like '₹1,20,000', '1.2 lakh', '120k', '34 yrs' to numbers",
    "parse_date": "convert any day-first date format to YYYY-MM-DD",
    "normalize_phone": "Indian mobile to 10 digits; invalid -> missing",
    "normalize_pincode": "6-digit PIN code; invalid -> missing",
    "normalize_email": "lowercase/trim email; invalid -> missing",
    "map_values": "replace raw category values using {raw: canonical}",
    "extract_number": "turn '1248 CC' into 1248 when every value uses the same `unit`; other values -> missing",
    "fix_range": "set numeric values outside [lo, hi] to missing",
    "flag_outliers": "blank values with robust log z-score above z AND above `above` (order-of-magnitude errors)",
    "drop_duplicates": "drop rows whose normalized values in `subset` repeat an earlier row",
    "impute": "fill missing values with 'median' (numbers) or 'mode'",
}
PARSE_TOOLS = {"extract_number", "parse_number", "parse_date", "normalize_phone", "normalize_pincode", "normalize_email"}


@dataclass
class StepInfo:
    n_changed: int = 0
    n_nullified: int = 0
    rows_dropped: int = 0
    n_nonnull_before: int = 0
    n_rewritten: int = 0
    examples: list = field(default_factory=list)
    changed_index: list = field(default_factory=list)


def _apply(s, fn, dtype=object):
    return pd.Series([v if is_null(v) else fn(v) for v in s], index=s.index, dtype=dtype)


def _to_float(s):
    return pd.Series([parse_number(v) for v in s], index=s.index, dtype=float)


def run_tool(df, tool, args):
    """Returns the new DataFrame. Raises ValueError for bad arguments."""
    df = df.copy()
    c = args.get("col")
    if c is not None and c not in df.columns:
        raise ValueError(f"unknown column {c!r}")
    if tool == "fix_encoding":
        df[c] = _apply(df[c], fix_text)
    elif tool == "standardize_missing":
        df[c] = _apply(df[c], lambda v: np.nan if is_missing_token(v) else v)
    elif tool == "normalize_text":
        df[c] = _apply(df[c], lambda v: apply_case(v, args.get("case")) if isinstance(v, str) else v)
    elif tool == "parse_number":
        df[c] = _to_float(df[c])
    elif tool == "parse_date":
        df[c] = _apply(df[c], lambda v: parse_date(v, args.get("dayfirst", True)))
    elif tool == "normalize_phone":
        df[c] = _apply(df[c], normalize_phone)
    elif tool == "normalize_pincode":
        df[c] = _apply(df[c], normalize_pincode)
    elif tool == "normalize_email":
        df[c] = _apply(df[c], normalize_email)
    elif tool == "extract_number":
        u = args["unit"].strip().lower()

        def ex(v):
            m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(.*?)\s*$", str(v))
            return float(m.group(1)) if m and m.group(2).strip().lower() in (u, "") else np.nan
        df[c] = pd.Series([np.nan if is_blank(v) else ex(v) for v in df[c]], index=df.index, dtype=float)
    elif tool == "map_values":
        m = {str(k): (v[0] if isinstance(v, (list, tuple)) else v) for k, v in args["mapping"].items()}
        df[c] = _apply(df[c], lambda v: m.get(str(v), m.get(str(v).strip(), v)))
    elif tool == "fix_range":
        x = _to_float(df[c])
        lo, hi = args.get("lo"), args.get("hi")
        bad = ((x < lo) if lo is not None else False) | ((x > hi) if hi is not None else False)
        df[c] = x.mask(bad)
    elif tool == "flag_outliers":
        x = _to_float(df[c])
        pos = x[x > 0]
        lx = np.log(pos)
        mad = 1.4826 * np.median(np.abs(lx - np.median(lx))) or 1.0
        z = np.abs(np.log(x.where(x > 0)) - np.median(lx)) / mad
        df[c] = x.mask(((z > args.get("z", 5.0)) & (x > args.get("above", 0))) | (x <= 0))
    elif tool == "drop_duplicates":
        sub = args["subset"]
        if not sub or any(col not in df.columns for col in sub):
            raise ValueError(f"bad subset {sub}")
        key = df[sub].astype(str).apply(lambda r: "|".join(fingerprint(v) for v in r), axis=1)
        df = df[~key.duplicated()]
    elif tool == "impute":
        s = df[c]
        if args.get("strategy") == "median":
            x = _to_float(s)
            df[c] = x.fillna(x.median())
        else:
            mode = s[~s.map(is_blank)].mode()
            if len(mode):
                df[c] = _apply(s, lambda v: v).where(~s.map(is_blank), mode.iloc[0])
    else:
        raise ValueError(f"unknown tool {tool!r}")
    return df


def diff(before, after, tool, args):
    info = StepInfo(rows_dropped=len(before) - len(after))
    c = args.get("col")
    if c is None:
        info.changed_index = [i for i in before.index if i not in after.index]
        info.examples = [(i, "row", "dropped") for i in info.changed_index[:5]]
        return info
    old, new = before[c].loc[after.index], after[c]
    old_null = old.map(is_null) | old.map(is_missing_token)
    new_null = new.map(is_null)
    same = (old.map(is_null) & new_null) | (old.astype(str) == new.astype(str))
    if tool in PARSE_TOOLS or tool in ("fix_range", "flag_outliers", "impute"):
        plain_equal = pd.Series([is_plain_number(o) and not is_null(n) and isinstance(n, (int, float))
                                 and float(o) == float(n) for o, n in zip(old, new)], index=old.index)
        same = same | plain_equal
    changed = ~same
    info.n_nonnull_before = int((~old_null).sum())
    info.n_changed = int(changed.sum())
    # changes beyond case/spacing/punctuation: what the guard counts as rewriting a value
    info.n_rewritten = int((changed & (old.map(lambda v: fingerprint(fix_text(v)) if isinstance(v, str) else v)
                                        != new.map(lambda v: fingerprint(v) if isinstance(v, str) else v))).sum())
    info.n_nullified = int((~old_null & new_null).sum())
    info.changed_index = list(old.index[changed])
    info.examples = [(i, old[i], new[i]) for i in info.changed_index[:5]]
    return info
