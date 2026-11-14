"""Safe CSV reading and writing for user-supplied files."""
from __future__ import annotations

import io
import re

import pandas as pd

MAX_ROWS = 300_000          # the agent is meant for laptop-size files
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")   # Excel on Windows often saves cp1252
_NUMERIC = re.compile(r"^[+-]?[\d.,]+(e[+-]?\d+)?$", re.I)


def read_csv_safely(source, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Read a CSV as text, trying common Windows/Excel encodings; raise ValueError with a plain message otherwise."""
    raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw.strip():
        raise ValueError("The file is empty.")
    last = None
    for enc in ENCODINGS:
        try:
            head = raw[:4096].decode(enc, errors="ignore").splitlines()[0] if raw.strip() else ""
            sep = max((",", ";", "\t"), key=head.count)   # comma unless the header clearly uses ; or tab
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc, sep=sep, on_bad_lines="skip")
            break
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            last = e
    else:
        raise ValueError(f"Could not read this file as a CSV ({last}). Save it as 'CSV UTF-8' from Excel and try again.")
    if df.shape[1] < 2:
        raise ValueError("The file has fewer than 2 columns; is it really a CSV?")
    if len(df) > max_rows:
        raise ValueError(f"The file has {len(df):,} rows; the demo limit is {max_rows:,}.")
    return df.astype(object)


def neutralize_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix text cells that a spreadsheet would run as a formula (=, +, -, @ at the start) with an apostrophe.
    Genuine numbers such as -5 or +91 are left alone. Used only for downloads; the cleaned data itself is unchanged."""
    def fix(v):
        if isinstance(v, str) and v[:1] in "=+-@\t\r" and not _NUMERIC.match(v.replace(" ", "")):
            return "'" + v
        return v
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(fix)
    return out
