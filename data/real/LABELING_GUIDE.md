# Gold labels for the real startup-funding file

`startup_gold.csv` holds the correct value of 4 columns (date, city, funding round, amount) for 140 rows of
`startup_funding.csv` (the original Kaggle file, 3,044 rows). The labels were written **before** the agent was
run on this file, using the rules below. Please review them yourself before presenting; disagreements are expected
on judgement calls and should be fixed in `scripts/make_gold_startup.py`.

**Sample:** 60 random rows that contain at least one non-standard value, 40 random rows from the rest, and 40
deliberately hard rows (text escapes like `\xc2\xa0`, broken dates, rare or misspelled cities, multi-city entries,
unusual amounts or round names). The hard stratum is over-represented on purpose, so overall scores on this sample
are lower than on the full file.

| Column | Convention |
|---|---|
| date | `YYYY-MM-DD`. Broken but readable dates are recovered (`05/072018` -> `2018-07-05`). |
| city | One current official name per city: Bangalore -> Bengaluru, Gurgaon -> Gurugram, New Delhi -> Delhi (judgement call), misspellings fixed. Multi-location entries (`Pune/Seattle`) and non-cities (`Kerala`, `USA/India`) are kept as written, minus junk characters. `nan` -> blank. |
| funding round | Spelling, spacing and typo variants of the same label become its most common spelling (`Seed / Angel Funding`, `Seed / Angle Funding` -> `Seed/ Angel Funding`). Different labels are NOT merged even if similar in meaning (`Seed`, `Seed Funding Round`, `Angel Funding` stay). |
| amount | Number in USD. `N/A`, `undisclosed`, `unknown` -> blank (never a made-up number). |

Comparison ignores letter case and repeated spaces, nothing else.

## Review record

All 140 rows were re-checked against the raw file and these rules: both irregular dates (`05/072018`,
`13/04.2015`) and every amount match their raw values; every city and round follows the table above. Judgement calls
to be aware of: `New Delhi` -> `Delhi`; `Taramani` (a Chennai neighbourhood) and `Kerala` (a state) kept as written
because they are not city names; `Seed`, `Seed Funding Round`, `Angel Funding` kept separate from `Seed Funding`;
`Crowd funding` vs `Crowd Funding` is a tie in the raw data (comparison ignores case, so it does not matter).
The review was done by the same AI assistant that drafted the labels, so it is not independent.
