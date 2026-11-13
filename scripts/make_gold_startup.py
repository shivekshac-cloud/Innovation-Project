"""Builds data/real/startup_gold.csv: hand-assigned correct values for 140 sampled rows of the real startup file.

Written BEFORE running the improved agent on this file. Every city and funding-type value in the sample is
labelled by hand in the tables below; irregular dates and amounts are labelled by hand too. Only plain
dd/mm/yyyy dates and plain comma-grouped integers are converted mechanically. See data/real/LABELING_GUIDE.md.
"""
import re
import sys

import pandas as pd

CITY = {  # raw -> correct. Multi-location entries stay as they are (only junk characters removed).
    "Pune": "Pune", "Mumbai": "Mumbai", "Bengaluru": "Bengaluru", "Bangalore": "Bengaluru", "Singapore": "Singapore",
    "Jaipur": "Jaipur", "Gurugram": "Gurugram", "Gurgaon": "Gurugram", "Hyderabad": "Hyderabad",
    "Ahemadabad": "Ahmedabad", "Ahmedabad": "Ahmedabad", "Chennai": "Chennai", "Delhi": "Delhi", "New Delhi": "Delhi",
    "Noida": "Noida", "Kozhikode": "Kozhikode", "Chandigarh": "Chandigarh", "nan": "", "": "", "Udupi": "Udupi",
    "\\\\xc2\\\\xa0New Delhi": "Delhi", "\\\\xc2\\\\xa0Noida": "Noida", "Nagpur": "Nagpur", "Panaji": "Panaji",
    "Kochi": "Kochi", "Taramani": "Taramani", "New Delhi / US": "New Delhi / US", "California": "California",
    "Amritsar": "Amritsar", "Pune/Seattle": "Pune/Seattle", "New York/ India": "New York/ India",
    "Delhi & Cambridge": "Delhi & Cambridge", "USA/India": "USA/India", "Nairobi": "Nairobi",
    "Mumbai/Bengaluru": "Mumbai/Bengaluru", "Kerala": "Kerala", "San Francisco": "San Francisco", "London": "London",
    "Bhopal": "Bhopal", "Mumbai / UK": "Mumbai / UK",
}
SEED_ANGEL = "Seed/ Angel Funding"  # most common raw spelling of this family (60 rows)
ROUND = {
    "Series A": "Series A", "Series B": "Series B", "Series C": "Series C", "Series E": "Series E", "Series H": "Series H",
    "Pre-Series A": "Pre-Series A", "Private Equity": "Private Equity", "Private\\\\nEquity": "Private Equity",
    "Seed / Angel Funding": SEED_ANGEL, "Seed/Angel Funding": SEED_ANGEL, "Seed/ Angel Funding": SEED_ANGEL,
    "Seed / Angle Funding": SEED_ANGEL, "Angel Funding": "Angel Funding", "Seed Funding": "Seed Funding",
    "Seed\\\\nFunding": "Seed Funding", "Debt Funding": "Debt Funding", "Private": "Private",
    "Crowd funding": "Crowd Funding", "Seed": "Seed", "Venture Round": "Venture Round", "": "",
    "Seed Funding Round": "Seed Funding Round", "Private Equity Round": "Private Equity Round", "Venture": "Venture",
}
AMOUNT = {"N/A": "", "Undisclosed": "", "undisclosed": "", "": "", "27,39,034.68": "2739034.68",
          "1,51,09,500.00": "15109500", "\\\\xc2\\\\xa010,000,000": "10000000", "\\\\xc2\\\\xa020,000,000": "20000000"}
DATE = {"05/072018": "2018-07-05", "13/04.2015": "2015-04-13"}


def main():
    raw = pd.read_csv("data/real/startup_funding.csv", dtype=str, keep_default_na=False)
    ids = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/real/startup_gold_sample_ids.csv", dtype=str, keep_default_na=False)["Sr No"]
    s = raw[raw["Sr No"].isin(ids)]
    rows = []
    for _, r in s.iterrows():
        d, c, t, a = r["Date dd/mm/yyyy"], r["City  Location"], r["InvestmentnType"], r["Amount in USD"]
        if d in DATE:
            gd = DATE[d]
        else:
            m = re.fullmatch(r"(\d\d)/(\d\d)/(\d{4})", d)
            assert m, f"unlabelled date {d!r}"
            gd = f"{m[3]}-{m[2]}-{m[1]}"
        if a in AMOUNT:
            ga = AMOUNT[a]
        else:
            assert re.fullmatch(r"[\d,]+", a), f"unlabelled amount {a!r}"
            ga = str(int(a.replace(",", "")))
        assert c in CITY, f"unlabelled city {c!r}"
        assert t in ROUND, f"unlabelled round {t!r}"
        rows.append({"Sr No": r["Sr No"], "date": gd, "city": CITY[c], "investment_type": ROUND[t], "amount_usd": ga,
                     "raw_date": d, "raw_city": c, "raw_investment_type": t, "raw_amount": a})
    out = pd.DataFrame(rows)
    out.to_csv("data/real/startup_gold.csv", index=False)
    print(len(out), "rows labelled")


if __name__ == "__main__":
    main()
