"""Value-level parsers for Indian data formats. Each returns np.nan when a value cannot be parsed."""
import math
import re
from datetime import datetime

import numpy as np

from .knowledge import MISSING_TOKENS

_WS = re.compile(r"\s+")
_CUR = re.compile(r"^(?:₹|rs\.?|inr)\s*")
_SUFFIX = re.compile(r"\s*(?:/-|yrs?|years?|y/o|/month|per month|p\.?m\.?)$")
_NUM = re.compile(r"^([-+]?\d+(?:\.\d+)?)\s*([a-z]+)?$")
_PLAIN = re.compile(r"^-?\d+(?:\.\d+)?$")
_EMAIL = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
UNITS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
         "k": 1e3, "thousand": 1e3}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%d %B %Y",
                "%d-%b-%y", "%d/%m/%y", "%d-%m-%y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d"]
US_DATE_FORMATS = [f.replace("%d/%m", "%m/%d").replace("%d-%m", "%m-%d") for f in DATE_FORMATS]


def is_null(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        return v is np.nan or (not isinstance(v, str) and bool(np.isnan(v)))
    except (TypeError, ValueError):
        return False


def is_missing_token(v):
    return isinstance(v, str) and v.strip().lower() in MISSING_TOKENS


def is_blank(v):
    return is_null(v) or is_missing_token(v)


def clean_ws(s):
    return _WS.sub(" ", str(s)).strip()


def fingerprint(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def is_plain_number(v):
    return isinstance(v, str) and bool(_PLAIN.match(v.strip()))


def parse_number(v):
    """'₹1,20,000', 'Rs. 120000/-', '1.2 lakh', '1.2L', '120k', '2 crore', '34 yrs' -> float."""
    if is_null(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower()
    if s in MISSING_TOKENS:
        return np.nan
    s = _CUR.sub("", s).replace(",", "").strip()
    for _ in range(2):
        s = _SUFFIX.sub("", s).strip()
    m = _NUM.match(s)
    if not m:
        return np.nan
    x, unit = float(m.group(1)), m.group(2)
    if unit:
        if unit not in UNITS:
            return np.nan
        x *= UNITS[unit]
    return round(x, 2)


def parse_date(v, dayfirst=True):
    """Any common Indian (day-first) date format -> 'YYYY-MM-DD'."""
    if is_blank(v):
        return np.nan
    s = clean_ws(v)
    for f in (DATE_FORMATS if dayfirst else US_DATE_FORMATS):
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return np.nan


def _digits(v):
    s = re.sub(r"\.0$", "", str(v).strip())
    return re.sub(r"\D", "", s)


def normalize_phone(v):
    """Indian mobile -> 10 digits starting 6-9. Strips +91 / 91 / leading 0."""
    if is_blank(v):
        return np.nan
    d = _digits(v)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d if len(d) == 10 and d[0] in "6789" else np.nan


def normalize_pincode(v):
    if is_blank(v):
        return np.nan
    d = _digits(v)
    return d if len(d) == 6 and d[0] != "0" else np.nan


def normalize_email(v):
    if is_blank(v):
        return np.nan
    s = clean_ws(v).lower().replace(" ", "")
    return s if _EMAIL.match(s) else np.nan


def apply_case(s, case):
    s = clean_ws(s)
    return {"title": s.title(), "lower": s.lower(), "upper": s.upper()}.get(case, s)
