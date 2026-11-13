### Main comparison (3000 applications, 6% cell error rate, 3% near-duplicates; mean ± std over 5 datasets)

| Method | Correct cells (all applicants) | Fixable errors fixed | Bad values blanked | Clean cells damaged | Applicants kept | Duplicates resolved | Downstream AUROC |
|---|---|---|---|---|---|---|---|
| No cleaning | 92.3% ± 0.2 | 0.0% ± 0.0 | 34.7% ± 2.1 | 0.0% ± 0.0 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.808 ± 0.005 |
| Naive pandas (dropna) | 66.8% ± 0.9 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 70.4% ± 0.9 | 1.1% ± 0.7 | 0.800 ± 0.004 |
| Agent, single pass | 97.5% ± 0.1 | 99.1% ± 0.2 | 77.2% ± 1.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.806 ± 0.005 |
| Agent (rule planner) | 97.5% ± 0.1 | 99.1% ± 0.2 | 99.6% ± 0.1 | 0.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.808 ± 0.004 |
| Faulty planner + guardrails | 96.7% ± 0.8 | 98.7% ± 0.5 | 99.6% ± 0.1 | 0.9% ± 0.8 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.799 ± 0.008 |
| Faulty planner, no guardrails | 49.6% ± 40.4 | 82.6% ± 7.2 | 99.7% ± 0.2 | 15.0% ± 2.1 | 60.1% ± 48.9 | 60.2% ± 48.7 | 0.775 ± 0.006 |

AUROC of the same model trained on the original clean data: 0.807 ± 0.008.

### Errors fixed by type: Agent (rule planner)

| Error type | Fixed or correctly blanked |
|---|---|
| Case / extra spaces | 100.0% ± 0.0 |
| Category spelling variants | 100.0% ± 0.0 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 |
| Old city name, in rule dictionary | 100.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 0.0% ± 0.0 |
| City misspelling | 93.4% ± 3.7 |
| State code ('MH') | 100.0% ± 0.0 |
| PIN code format | 100.0% ± 0.0 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 |
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 |
| Date format (day-first variants) | 100.0% ± 0.0 |
| Disguised missing (NA, -, ?) | 100.0% ± 0.0 |
| Extra-zero outlier | 95.9% ± 1.4 |
| Sentinel credit score (9999, 0, -1) | 100.0% ± 0.0 |

### Unseen formats: error formats the rules were never written for

Same data and error rate, but money like `Rs 1.2 Lakhs only` / `120000 rupees`, dates like `5th March 2024`, phones like `Mob: 98765...`, cities like `Bombay (MH)`, gender like `Woman`, ages like `Age 34`. Written after the rules; the rules were not changed after seeing these results.

| Method | Correct cells | Fixable errors fixed | Clean cells damaged |
|---|---|---|---|
| Agent (rule planner) | 95.8% ± 0.1 | 66.1% ± 1.2 | 0.0% ± 0.0 |

| Error type (Agent (rule planner)) | Known formats | Unseen formats |
|---|---|---|
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 | 25.0% ± 2.6 |
| Date format (day-first variants) | 100.0% ± 0.0 | 68.8% ± 3.3 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 | 78.2% ± 1.5 |
| Category spelling variants | 100.0% ± 0.0 | 18.2% ± 1.4 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 | 0.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 0.0% ± 0.0 | 0.0% ± 0.0 |

### Fairness: who loses data? (tier-3 towns have 2x the error rate in this simulation)

| Method | Kept: tier 1 | Kept: tier 2 | Kept: tier 3 | Fixed: tier 1 | Fixed: tier 3 |
|---|---|---|---|---|---|
| No cleaning | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Naive pandas (dropna) | 75.7% ± 0.6 | 76.1% ± 1.5 | 55.6% ± 1.7 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Agent, single pass | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.5% ± 0.2 | 98.9% ± 0.3 |
| Agent (rule planner) | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.5% ± 0.2 | 98.9% ± 0.3 |

### Error-rate sweep (3 datasets per rate)

| Error rate | No cleaning: correct cells | Naive pandas (dropna): correct cells | Agent (rule planner): correct cells | Naive: applicants kept |
|---|---|---|---|---|
| 3% | 96.1% ± 0.0 | 81.9% ± 0.2 | 98.8% ± 0.0 | 84.0% ± 0.1 |
| 6% | 92.4% ± 0.0 | 67.1% ± 0.1 | 97.5% ± 0.1 | 70.6% ± 0.1 |
| 12% | 84.9% ± 0.2 | 43.4% ± 0.7 | 94.8% ± 0.1 | 48.0% ± 0.9 |
| 20% | 75.1% ± 0.1 | 25.7% ± 0.5 | 90.9% ± 0.4 | 30.5% ± 0.6 |
| 30% | 62.5% ± 0.2 | 12.3% ± 0.3 | 86.1% ± 0.0 | 15.8% ± 0.3 |

| Seconds per run | No cleaning: 0.0 | Naive pandas (dropna): 0.0 | Agent, single pass: 2.6 | Agent (rule planner): 3.2 | Faulty planner + guardrails: 3.6 | Faulty planner, no guardrails: 2.6 |
