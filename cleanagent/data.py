"""Benchmark data: a clean synthetic Indian loan-application table, and a corrupter that injects realistic
errors and records exactly what it changed (the ground truth for scoring).

Simulation assumption, stated openly: applications from tier-3 towns are twice as likely to contain errors
(more paper forms and assisted data entry). This lets us check whether cleaning harms one group more.
"""
import io

import numpy as np
import pandas as pd

from .knowledge import CITY_ALIASES

CITIES = [("Mumbai", "Maharashtra", 1), ("Delhi", "Delhi", 1), ("Bengaluru", "Karnataka", 1), ("Chennai", "Tamil Nadu", 1),
          ("Kolkata", "West Bengal", 1), ("Hyderabad", "Telangana", 1), ("Pune", "Maharashtra", 1), ("Ahmedabad", "Gujarat", 1),
          ("Jaipur", "Rajasthan", 2), ("Lucknow", "Uttar Pradesh", 2), ("Kochi", "Kerala", 2), ("Thiruvananthapuram", "Kerala", 2),
          ("Visakhapatnam", "Andhra Pradesh", 2), ("Varanasi", "Uttar Pradesh", 2), ("Indore", "Madhya Pradesh", 2),
          ("Bhopal", "Madhya Pradesh", 2), ("Gurugram", "Haryana", 2), ("Mysuru", "Karnataka", 2), ("Vadodara", "Gujarat", 2),
          ("Coimbatore", "Tamil Nadu", 2), ("Nagpur", "Maharashtra", 2), ("Patna", "Bihar", 2),
          ("Hubballi", "Karnataka", 3), ("Kota", "Rajasthan", 3), ("Bareilly", "Uttar Pradesh", 3), ("Durgapur", "West Bengal", 3),
          ("Sangli", "Maharashtra", 3), ("Hosur", "Tamil Nadu", 3), ("Bhilai", "Chhattisgarh", 3), ("Ajmer", "Rajasthan", 3),
          ("Thrissur", "Kerala", 3), ("Guntur", "Andhra Pradesh", 3), ("Gaya", "Bihar", 3), ("Jhansi", "Uttar Pradesh", 3)]
TIER = {c: t for c, _, t in CITIES}
# every old name used by the corrupter; only some are in knowledge.CITY_ALIASES
ALL_ALIASES = {"Mumbai": ["Bombay"], "Bengaluru": ["Bangalore"], "Chennai": ["Madras"], "Kolkata": ["Calcutta"],
               "Gurugram": ["Gurgaon"], "Mysuru": ["Mysore"], "Vadodara": ["Baroda"], "Delhi": ["New Delhi"],
               "Pune": ["Poona"], "Kochi": ["Cochin"], "Thiruvananthapuram": ["Trivandrum"], "Visakhapatnam": ["Vizag"],
               "Varanasi": ["Banaras", "Benares"], "Hubballi": ["Hubli"], "Thrissur": ["Trichur"]}
STATE_CODE = {"Maharashtra": "MH", "Delhi": "DL", "Karnataka": "KA", "Tamil Nadu": "TN", "West Bengal": "WB", "Telangana": "TG",
              "Gujarat": "GJ", "Rajasthan": "RJ", "Uttar Pradesh": "UP", "Kerala": "KL", "Andhra Pradesh": "AP",
              "Madhya Pradesh": "MP", "Haryana": "HR", "Bihar": "BR", "Chhattisgarh": "CG"}
FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohit", "Pooja", "Arjun", "Neha", "Suresh", "Kavita",
         "Manish", "Divya", "Karthik", "Ritu", "Sanjay", "Meera", "Imran", "Fatima", "Harpreet", "Lakshmi", "Joseph", "Ayesha"]
LAST = ["Sharma", "Gupta", "Kumar", "Reddy", "Iyer", "Patel", "Singh", "Nair", "Das", "Joshi", "Khan", "Mehta", "Pillai",
        "Verma", "Rao", "Banerjee", "Chauhan", "Shetty", "Thomas", "Ansari"]
EMPLOYMENT = ["Salaried", "Self-employed", "Business owner", "Farmer", "Retired"]
COLUMNS = ["applicant_id", "name", "gender", "age", "city", "state", "pincode", "phone", "email", "employment",
           "monthly_income", "loan_amount", "credit_score", "existing_emis", "application_date", "defaulted"]
NUMERIC = ["age", "monthly_income", "loan_amount", "credit_score", "existing_emis", "defaulted"]


def make_clean(n=3000, seed=0, id_offset=0):
    rng = np.random.default_rng(seed)
    city_w = np.array([3.0 if t == 1 else 2.0 if t == 2 else 1.6 for _, _, t in CITIES])
    ci = rng.choice(len(CITIES), n, p=city_w / city_w.sum())
    city = [CITIES[i][0] for i in ci]
    tier = np.array([CITIES[i][2] for i in ci])
    emp = rng.choice(EMPLOYMENT, n, p=[.45, .2, .12, .13, .10])
    emp = np.where((tier == 3) & (rng.random(n) < .2), "Farmer", emp)
    age = np.clip(rng.normal(37, 10, n), 21, 70).round().astype(int)
    age = np.where(emp == "Retired", np.maximum(age, 58 + rng.integers(0, 10, n)), age)
    base = np.exp(rng.normal(10.7 - 0.25 * (tier - 1), 0.55, n))
    income = (np.round(base / 500) * 500).clip(8000, 2_000_000)
    loan = (np.round(income * rng.uniform(3, 30, n) / 5000) * 5000).clip(20000, 50_000_000)
    score = np.clip(rng.normal(715, 70, n), 300, 900).round().astype(int)
    emis = rng.poisson(0.9, n).clip(0, 6)
    burden = (loan / 60 + emis * 0.06 * income) / income
    emp_eff = pd.Series(emp).map({"Salaried": -.35, "Self-employed": .2, "Business owner": .1, "Farmer": .35, "Retired": .15}).to_numpy()
    logit = -2.2 + 4.5 * (burden - 0.35) - 0.022 * (score - 715) + 2 * emp_eff + 0.05 * np.clip(30 - age, 0, None)
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    first, last = rng.choice(FIRST, n), rng.choice(LAST, n)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 640, n), unit="D")
    df = pd.DataFrame({
        "applicant_id": [f"APP{id_offset + i:05d}" for i in range(n)],
        "name": [f"{a} {b}" for a, b in zip(first, last)],
        "gender": rng.choice(["Male", "Female"], n, p=[.58, .42]),
        "age": age, "city": city, "state": [CITIES[i][1] for i in ci],
        "pincode": [str(rng.integers(110001, 855117)) for _ in range(n)],
        "phone": [str(rng.integers(6, 10)) + "".join(map(str, rng.integers(0, 10, 9))) for _ in range(n)],
        "email": [f"{a.lower()}.{b.lower()}{rng.integers(1, 999)}@{rng.choice(['gmail.com', 'yahoo.co.in', 'outlook.com'])}"
                  for a, b in zip(first, last)],
        "employment": emp, "monthly_income": income.astype(float), "loan_amount": loan.astype(float),
        "credit_score": score, "existing_emis": emis, "application_date": dates.strftime("%Y-%m-%d"), "defaulted": default,
    })
    df.index.name = "rid"
    return df, pd.Series(tier, index=df.index, name="tier")


# ---------------------------------------------------------------- corruption
CORRUPTIONS = {
    "name": [("case_space", 1)],
    "gender": [("variant", 1), ("missing", .3)],
    "age": [("num_text", 1), ("outlier", .6), ("missing", .6)],
    "city": [("case_space", 1), ("alias", 1.2), ("typo", .8), ("missing", .3)],
    "state": [("code", 1), ("case_space", .6)],
    "pincode": [("pin_format", 1), ("missing", .3)],
    "phone": [("phone_format", 1.5), ("missing", .3)],
    "email": [("case_space", 1)],
    "employment": [("variant", 1), ("missing", .3)],
    "monthly_income": [("money_format", 1.5), ("outlier", .5), ("missing", .6)],
    "loan_amount": [("money_format", 1.5), ("missing", .4)],
    "credit_score": [("sentinel", .6), ("missing", .6)],
    "existing_emis": [("missing", .4)],
    "application_date": [("date_format", 1.5), ("missing", .3)],
}
RECOVERABLE = {"case_space", "variant", "num_text", "alias", "typo", "code", "pin_format", "phone_format",
               "money_format", "date_format"}
UNRECOVERABLE = {"missing", "outlier", "sentinel"}
MISSING_FORMS = ["NA", "-", "?", "null", "N/A", "", "not available", "nil", "--"]


def inr(v):
    s = str(int(round(v)))
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail]) if parts else tail


def _case_space(v, rng):
    s = str(v)
    for _ in range(5):
        out = rng.choice([s.upper(), s.lower(), f"  {s}", f"{s} ", s.replace(" ", "  ") if " " in s else f" {s} ",
                          s.swapcase()])
        if out != s:
            return out
    return f" {s} "


def _typo(v, rng):
    s = str(v)
    for _ in range(10):
        i = int(rng.integers(1, len(s) - 1))
        op = rng.integers(0, 3)
        out = s[:i] + s[i + 1] + s[i] + s[i + 2:] if op == 0 else s[:i] + s[i + 1:] if op == 1 else s[:i] + s[i] + s[i:]
        if out.lower() != s.lower():
            return out
    return s + "a"


def corrupt_value(col, v, kind, rng):
    if kind == "missing":
        return rng.choice(MISSING_FORMS)
    if kind == "case_space":
        return _case_space(v, rng)
    if kind == "typo":
        return _typo(v, rng)
    if kind == "alias":
        a = rng.choice(ALL_ALIASES[v])
        return a if rng.random() < .7 else _case_space(a, rng)
    if kind == "variant":
        if col == "gender":
            return rng.choice([v[0], v.lower(), v.upper(), f" {v}", v[0].lower()])
        return rng.choice([v.lower(), v.upper(), v.replace("-", " "), v.replace("-", " ").title(), f"{v} ", v.replace(" ", "-")]) \
            if ("-" in v or " " in v) else _case_space(v, rng)
    if kind == "num_text":
        return rng.choice([f"{v} yrs", f"{v} years", f"{v}yrs"])
    if kind == "outlier":
        return str(int(v) * (10 if col == "age" else 100))
    if kind == "sentinel":
        return rng.choice(["9999", "0", "-1", "999"])
    if kind == "code":
        c = STATE_CODE[v]
        return c if rng.random() < .7 else c.lower()
    if kind == "pin_format":
        return rng.choice([f"{v[:3]} {v[3:]}", f"{v}.0", f" {v}"])
    if kind == "phone_format":
        return rng.choice([f"+91 {v[:5]} {v[5:]}", f"+91-{v}", f"0{v}", f"91{v}", f"{v[:5]}-{v[5:]}", f"(+91) {v[:5]} {v[5:]}"])
    if kind == "money_format":
        forms = [f"₹{inr(v)}", f"Rs. {int(v)}", f"INR {inr(v)}/-", f"Rs.{inr(v)}"]
        if v % 1000 == 0:
            forms.append(f"{int(v // 1000)}k")
        if v % 1000 == 0 and v >= 100000:
            forms += [f"{v / 1e5:g} lakh", f"{v / 1e5:g}L"]
        if v % 100000 == 0 and v >= 1e7:
            forms.append(f"{v / 1e7:g} crore")
        return rng.choice(forms)
    if kind == "date_format":
        d = pd.Timestamp(v)
        return d.strftime(rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y", "%d-%b-%y"]))
    raise ValueError(kind)


def corrupt(clean, tier, rate=0.06, dup_rate=0.03, tier3_mult=2.0, seed=0):
    """Returns (dirty, ledger, dup_of). dirty is read back from CSV as text, like a real file.

    ledger: one row per corrupted cell (rid, column, kind, subtype, dirty value).
    dup_of: {duplicate rid: original rid}.
    """
    rng = np.random.default_rng(seed)
    dirty = clean.astype(str).astype(object)
    rows = []
    mult = np.where(tier.to_numpy() == 3, tier3_mult, 1.0)
    for col, kinds in CORRUPTIONS.items():
        hit = rng.random(len(clean)) < rate * mult
        for rid in clean.index[hit]:
            v = clean.at[rid, col]
            opts = [(k, w) for k, w in kinds if not (k == "alias" and v not in ALL_ALIASES)]
            ks, ws = zip(*opts)
            kind = rng.choice(ks, p=np.array(ws) / sum(ws))
            dv = corrupt_value(col, v, kind, rng)
            if dv == str(v):
                continue
            sub = kind
            if kind == "alias":
                from .parsers import fingerprint
                sub = "alias_known" if fingerprint(dv) in CITY_ALIASES else "alias_unknown"
            dirty.at[rid, col] = dv
            rows.append((rid, col, kind, sub, dv))
    ledger = pd.DataFrame(rows, columns=["rid", "column", "kind", "subtype", "dirty"])
    # near-duplicate rows: re-submitted applications with small formatting differences
    n_dup = int(dup_rate * len(clean))
    src = rng.choice(clean.index, n_dup, replace=False)
    dups, dup_of, extra = [], {}, []
    next_rid = int(clean.index.max()) + 1
    for k, r in enumerate(src):
        new = dirty.loc[r].copy()
        nr = next_rid + k
        for col in rng.choice(["name", "email", "city"], 1):
            new[col] = _case_space(clean.at[r, col], rng)
            extra.append((nr, col, "case_space", "case_space", new[col]))
        for _, row in ledger[ledger.rid == r].iterrows():
            if row.column != new.index[0] and not any(e[0] == nr and e[1] == row.column for e in extra):
                extra.append((nr, row.column, row.kind, row.subtype, row.dirty))
        new.name = nr
        dups.append(new)
        dup_of[nr] = int(r)
    dirty = pd.concat([dirty, pd.DataFrame(dups)])
    ledger = pd.concat([ledger, pd.DataFrame(extra, columns=ledger.columns)], ignore_index=True)
    dirty = dirty.sample(frac=1, random_state=seed)
    dirty.index.name = "rid"
    buf = io.StringIO()
    dirty.to_csv(buf)
    buf.seek(0)
    dirty = pd.read_csv(buf, dtype=str, index_col="rid").astype(object)
    dirty.index = dirty.index.astype(int)
    return dirty, ledger, dup_of
