"""Value-level parsers for Indian data formats. Each returns np.nan when a value cannot be parsed."""
import math
import re
from collections import Counter
from datetime import datetime

import numpy as np

from .knowledge import MISSING_TOKENS

_WS = re.compile(r"\s+")
_CUR = re.compile(r"^(?:₹|rs\.?|inr)\s*")
_SUFFIX = re.compile(r"\s*(?:/-|\+|/month|per month|p\.?m\.?)$")
_N = r"[-+]?\d+(?:\.\d+)?"
_RANGE = re.compile(rf"^({_N})\s*(?:-|to|–)\s*({_N})\s*(.*)$")
_ONE = re.compile(rf"^({_N})\s*(.*)$")
_PLAIN = re.compile(r"^-?\d+(?:\.\d+)?$")
_EMAIL = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
_ESC = re.compile(r"(?:\\+x[0-9a-fA-F]{2})+")
_ESC_NL = re.compile(r"\\+[nrt]")
UNITS = {  # multiplier to the base unit (rupees, square feet, count, years)
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "k": 1e3, "thousand": 1e3,
    "sqft": 1, "sqfeet": 1, "sqmeter": 10.7639, "sqmeters": 10.7639, "sqm": 10.7639, "sqyards": 9, "sqyard": 9,
    "sqyd": 9, "acre": 43560, "acres": 43560, "cents": 435.6, "cent": 435.6, "guntha": 1089, "gunthas": 1089,
    "grounds": 2400.35, "ground": 2400.35, "perch": 272.25,
    "bhk": 1, "rk": 1, "bedroom": 1, "bedrooms": 1, "bed": 1, "beds": 1,
    "yrs": 1, "yr": 1, "years": 1, "year": 1, "yo": 1,
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%d %B %Y",
                "%d-%b-%y", "%d/%m/%y", "%d-%m-%y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d"]
US_DATE_FORMATS = [f.replace("%d/%m", "%m/%d").replace("%d-%m", "%m-%d").replace("%d.%m", "%m.%d") for f in DATE_FORMATS]


def is_null(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        return v is np.nan or (not isinstance(v, str) and bool(np.isnan(v)))
    except (TypeError, ValueError):
        return False


def fix_text(v):
    """Repair export damage: literal byte escapes ('\\xc2\\xa0', '\\xe2\\x80\\x99'), literal '\\n', non-breaking
    spaces and UTF-8 read as Windows-1252 ('â€™')."""
    if not isinstance(v, str):
        return v
    s = _ESC.sub(lambda m: bytes(int(h, 16) for h in re.findall(r"x([0-9a-fA-F]{2})", m.group(0)))
                 .decode("utf-8", errors="ignore"), v)
    s = _ESC_NL.sub(" ", s).replace("\xa0", " ")
    if "â" in s or "Ã" in s:
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if s != v:
        s = _WS.sub(" ", s).strip()
    return s


def has_text_artifacts(v):
    return isinstance(v, str) and fix_text(v) != v


def is_missing_token(v):
    return isinstance(v, str) and fix_text(v).strip().lower() in MISSING_TOKENS


def is_blank(v):
    return is_null(v) or is_missing_token(v)


def clean_ws(s):
    return _WS.sub(" ", str(s)).strip()


def fingerprint(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def is_plain_number(v):
    return isinstance(v, str) and bool(_PLAIN.match(v.strip()))


def _unit(rest):
    if re.search(r"\d", rest):
        return None
    u = re.sub(r"[^a-z]", "", rest)
    return 1.0 if u == "" else UNITS.get(u)


def parse_number(v):
    """'₹1,20,000', 'Rs. 120000/-', '1.2 lakh', '120k', '2 crore', '34 yrs', '14,342,000+', '1133 - 1384' (midpoint),
    '34.46Sq. Meter' (to sq ft), '5.31Acres', '2 BHK' -> float."""
    if is_null(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = fix_text(str(v)).strip().lower()
    if s in MISSING_TOKENS:
        return np.nan
    s = _CUR.sub("", s).replace(",", "").strip()
    for _ in range(2):
        s = _SUFFIX.sub("", s).strip()
    m = _RANGE.match(s)
    if m and not m.group(1).startswith("-") and 0 < float(m.group(1)) <= float(m.group(2)) <= 10 * float(m.group(1)):
        mult = _unit(m.group(3))
        return np.nan if mult is None else _scaled((float(m.group(1)) + float(m.group(2))) / 2, mult)
    m = _ONE.match(s)
    if not m:
        return np.nan
    mult = _unit(m.group(2))
    return np.nan if mult is None else _scaled(float(m.group(1)), mult)


def _scaled(x, mult):
    return x if mult == 1 else round(x * mult, 4)


def _try_formats(s, fmts):
    for f in fmts:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_date(v, dayfirst=True):
    """Common date formats -> 'YYYY-MM-DD'. Also repairs mixed or doubled separators ('12/05.2015', '22/01//2015')
    and a missing separator ('05/072018')."""
    if is_blank(v):
        return np.nan
    s = clean_ws(fix_text(v))
    fmts = DATE_FORMATS if dayfirst else US_DATE_FORMATS
    out = _try_formats(s, fmts)
    if out is None and re.fullmatch(r"[\d./\- ]+", s):
        s2 = re.sub(r"[./\- ]+", "/", s).strip("/")
        out = _try_formats(s2, fmts)
        d = re.sub(r"\D", "", s)
        if out is None and len(d) == 8:
            out = _try_formats(d, ["%Y%m%d"] if d[:2] in ("19", "20") and int(d[4:6]) <= 12 else
                               (["%d%m%Y"] if dayfirst else ["%m%d%Y"]))
    return np.nan if out is None else out


def date_order(values):
    """Guess whether numeric dates are day-first. Returns ('dayfirst' | 'monthfirst' | 'unknown', evidence counts)."""
    c = Counter()
    for v in values:
        m = re.match(r"^\s*(\d{1,2})[./\-](\d{1,2})[./\-]", fix_text(str(v)))
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12 >= b:
                c["dayfirst"] += 1
            elif b > 12 >= a:
                c["monthfirst"] += 1
    if c["dayfirst"] and not c["monthfirst"]:
        return "dayfirst", dict(c)
    if c["monthfirst"] and not c["dayfirst"]:
        return "monthfirst", dict(c)
    if not c:
        return "unknown", {}
    return ("dayfirst" if c["dayfirst"] >= c["monthfirst"] else "monthfirst"), dict(c)


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
