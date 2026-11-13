## Real data 1: Indian startup funding (3,044 rows, original Kaggle file)

Accuracy on 140 hand-labelled rows (see `data/real/LABELING_GUIDE.md`; 40 of the rows are deliberately hard cases). A row the method deleted counts as wrong.

| Method | Date | City (accepting old names) | Funding round | Amount | All four | Rows kept |
|---|---|---|---|---|---|---|
| No cleaning | 0.0% | 70.0% (99%) | 97.9% | 97.1% | **66.2%** | 100% |
| Typical pandas notebook | 62.1% | 43.6% (61%) | 60.0% | 62.1% | **57.0%** | 62% |
| Published human cleaning (GitHub) | 96.4% | 57.1% (86%) | 95.7% | 70.7% | **80.0%** | 97% |
| Agent (rule planner) | 100.0% | 100.0% (100%) | 100.0% | 100.0% | **100.0%** | 100% |

Whole file: distinct city spellings 112 -> 99, funding-round labels 55 -> 42, usable amounts 1 -> 2073, all 3043 dates parsed, 7048 cells changed, 2.9 s.
The published human cleaning filled 974 undisclosed amounts with the average ($18.4M); the agent leaves them blank.

**Read this table with care.** The agent's rules were extended after inspecting this file (escape codes, broken dates, misspelled cities), so this is a development score, not a blind test. The blind test is the CarDekho file below. The city score also depends on the naming convention: the human analyst kept old names like Bangalore, which the bracketed score accepts. 'No cleaning' scores 0% on dates only because they are not in YYYY-MM-DD.

Remaining agent errors on the gold rows are listed in `startup_agent_errors.csv`.

## Real data 2: Bengaluru house prices (13,320 rows)

A price model (gradient boosting on log price) trained on each cleaned version, tested on the same 20% of houses (5 random splits). Exact duplicate rows were removed for every method first, so no method gains from leakage.

| Method | Test houses it can price | Mean abs. error (lakh), own rows | MAPE, own rows | Mean abs. error, rows all methods cover | MAPE, common rows |
|---|---|---|---|---|---|
| No cleaning | 100.0% ± 0.0 | 32.2 ± 1.7 | 24.9% ± 1.5 | 32.0 ± 1.7 | 24.6% ± 1.5 |
| Typical tutorial recipe | 97.7% ± 0.1 | 30.7 ± 1.2 | 22.6% ± 0.6 | 30.8 ± 1.2 | 22.6% ± 0.6 |
| Agent (rule planner) | 100.0% ± 0.0 | 31.9 ± 2.1 | 24.4% ± 1.6 | 31.7 ± 1.9 | 24.1% ± 1.5 |
| Agent + same row filter as tutorial | 99.3% ± 0.1 | 30.4 ± 1.0 | 22.5% ± 0.6 | 30.5 ± 1.1 | 22.4% ± 0.6 |

Paired over the 5 splits (MAPE on common rows, minus the tutorial's):

- Agent alone: +1.6 points (better in 0 of 5 splits). **Worse**, and we traced why: the agent keeps 74 listings with no bathroom or size information, and training on them misleads this model. The converted unit rows are not the problem (removing them changes nothing).
- Agent + the tutorial's row filter (drop listings missing location, size or bath): -0.1 points (better in 4 of 5 splits), while still pricing the 1.6% of houses the tutorial throws away, with MAPE 24% on those (mostly large plots in acres or ranges).

Lesson: fixing values and choosing training rows are separate decisions. The agent does the first and never deletes data silently; the second belongs to whoever builds the model.

The agent converted 246 areas the tutorial recipe throws away (200 ranges like `1133 - 1384`, 46 with units like `34.46Sq. Meter` or `5.31Acres`). Location spellings 1305 -> 1283. 7.0 s.

## Real data 3: CarDekho used cars (8,128 rows), blind hold-out

Run once with the code frozen (fingerprint `7a376f4a6226`) before anyone looked at the output. Every one of its 72 cell changes was checked by hand: all correct, but all trivial (case and spacing in `torque`, e.g. `151NM@ 4850rpm` -> `151Nm@ 4850rpm`). It removed 1202 exact duplicate rows. **It did not convert `23.4 kmpl`, `1248 CC` or `74 bhp` to numbers**, because those units were not in its dictionary. This is the clearest evidence of the rule-based agent's limits on unseen data.

**After the hold-out (post-hoc fix, not a blind result):** a generic `extract_number` tool now converts a column whose values share one unit, and only flags columns that mix units. On this file it converted engine, max_power and flagged mileage for a human or LLM to decide (mileage mixes `kmpl` and `km/kg`, which are not comparable).

