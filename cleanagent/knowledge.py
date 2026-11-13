"""Domain knowledge available to the rule-based planner.

This is deliberately incomplete (it lists only common Indian city renames). The benchmark also uses
older names that are NOT listed here, to measure what a rule-based agent misses and an LLM could fix.
"""
from __future__ import annotations
MISSING_TOKENS = {"", "na", "n/a", "n.a.", "nan", "null", "none", "-", "--", "?", "nil", "not available",
                  "unknown", "#n/a", "missing", "not applicable", "undisclosed",
                  "not disclosed", "n.a", "na."}

CITY_ALIASES = {"bombay": "Mumbai", "bangalore": "Bengaluru", "madras": "Chennai", "calcutta": "Kolkata",
                "gurgaon": "Gurugram", "mysore": "Mysuru", "baroda": "Vadodara", "newdelhi": "Delhi"}

STATE_CODES = {"mh": "Maharashtra", "dl": "Delhi", "ka": "Karnataka", "tn": "Tamil Nadu", "wb": "West Bengal",
               "tg": "Telangana", "ts": "Telangana", "gj": "Gujarat", "rj": "Rajasthan", "up": "Uttar Pradesh",
               "kl": "Kerala", "ap": "Andhra Pradesh", "mp": "Madhya Pradesh", "hr": "Haryana", "pb": "Punjab",
               "br": "Bihar", "od": "Odisha", "or": "Odisha", "as": "Assam", "jh": "Jharkhand", "ga": "Goa",
               "cg": "Chhattisgarh", "ct": "Chhattisgarh", "uk": "Uttarakhand", "hp": "Himachal Pradesh"}

VALUE_ALIASES = {**CITY_ALIASES, **STATE_CODES, "m": "Male", "f": "Female"}

# column-name tokens -> semantic type
NAME_HINTS = {
    "phone": "phone", "mobile": "phone", "contact": "phone",
    "pin": "pincode", "pincode": "pincode", "zip": "pincode", "postal": "pincode",
    "email": "email", "mail": "email",
    "date": "date", "dob": "date",
    "income": "money", "amount": "money", "salary": "money", "price": "money", "loan": "money",
    "fee": "money", "cost": "money", "balance": "money", "rent": "money",
    "age": "numeric", "score": "numeric", "count": "numeric", "qty": "numeric", "emis": "numeric",
    "name": "name", "id": "id", "sqft": "numeric", "bath": "numeric", "balcony": "numeric", "usd": "money",
}

# plausible ranges, matched on column-name tokens (first match wins)
RANGES = [(("credit", "score"), (300, 900)), (("cibil",), (300, 900)), (("age",), (18, 100)),
          (("emis",), (0, 20)), (("percent",), (0, 100)), (("pct",), (0, 100))]
