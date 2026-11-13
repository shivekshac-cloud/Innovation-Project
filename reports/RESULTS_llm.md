**LLM: gpt-oss-20b, run locally in LM Studio on an M5 MacBook Air.** Answers recorded once and replayed from `llm_cache.json` (6 questions, all found in the recording). The demo laptop (i3, 8 GB RAM) cannot run a 20B model, so it replays the same answers offline.

### Main comparison (3000 applications, 6% cell error rate, 3% near-duplicates; mean ± std over 2 datasets)

| Method | Correct cells (all applicants) | Fixable errors fixed | Bad values blanked | Clean cells damaged | Applicants kept | Duplicates resolved | Downstream AUROC |
|---|---|---|---|---|---|---|---|
| No cleaning | 92.4% ± 0.0 | 0.0% ± 0.0 | 33.8% ± 1.2 | 0.0% ± 0.0 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.812 ± 0.005 |
| Naive pandas (dropna) | 67.2% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 70.7% ± 0.1 | 1.7% ± 0.6 | 0.803 ± 0.004 |
| Agent, single pass | 97.6% ± 0.0 | 99.1% ± 0.1 | 77.0% ± 1.3 | 0.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.811 ± 0.003 |
| Agent (rule planner) | 97.6% ± 0.0 | 99.1% ± 0.1 | 99.7% ± 0.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.811 ± 0.001 |
| Faulty planner + guardrails | 97.3% ± 0.0 | 99.1% ± 0.1 | 99.7% ± 0.0 | 0.3% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.803 ± 0.001 |
| Faulty planner, no guardrails | 39.6% ± 39.4 | 74.9% ± 4.9 | 99.8% ± 0.2 | 16.0% ± 3.0 | 50.1% ± 49.9 | 50.6% ± 49.4 | 0.780 ± 0.000 |
| Agent (hybrid: rules + LLM) | 97.7% ± 0.0 | 99.9% ± 0.1 | 99.7% ± 0.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.812 ± 0.005 |

AUROC of the same model trained on the original clean data: 0.815 ± 0.002.

### Errors fixed by type: Agent (hybrid: rules + LLM)

| Error type | Fixed or correctly blanked |
|---|---|
| Case / extra spaces | 100.0% ± 0.0 |
| Category spelling variants | 100.0% ± 0.0 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 |
| Old city name, in rule dictionary | 100.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 94.7% ± 5.3 |
| City misspelling | 98.6% ± 1.4 |
| State code ('MH') | 100.0% ± 0.0 |
| PIN code format | 100.0% ± 0.0 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 |
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 |
| Date format (day-first variants) | 100.0% ± 0.0 |
| Disguised missing (NA, -, ?) | 100.0% ± 0.0 |
| Extra-zero outlier | 97.3% ± 0.2 |
| Sentinel credit score (9999, 0, -1) | 100.0% ± 0.0 |

### Errors fixed by type: Agent (rule planner)

| Error type | Fixed or correctly blanked |
|---|---|
| Case / extra spaces | 100.0% ± 0.0 |
| Category spelling variants | 100.0% ± 0.0 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 |
| Old city name, in rule dictionary | 100.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 0.0% ± 0.0 |
| City misspelling | 95.6% ± 0.4 |
| State code ('MH') | 100.0% ± 0.0 |
| PIN code format | 100.0% ± 0.0 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 |
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 |
| Date format (day-first variants) | 100.0% ± 0.0 |
| Disguised missing (NA, -, ?) | 100.0% ± 0.0 |
| Extra-zero outlier | 97.3% ± 0.2 |
| Sentinel credit score (9999, 0, -1) | 100.0% ± 0.0 |

### Unseen formats: error formats the rules were never written for

Same data and error rate, but money like `Rs 1.2 Lakhs only` / `120000 rupees`, dates like `5th March 2024`, phones like `Mob: 98765...`, cities like `Bombay (MH)`, gender like `Woman`, ages like `Age 34`. Written after the rules; the rules were not changed after seeing these results.

| Method | Correct cells | Fixable errors fixed | Clean cells damaged |
|---|---|---|---|
| Agent (rule planner) | 95.9% ± 0.1 | 66.0% ± 0.8 | 0.0% ± 0.0 |
| Agent (hybrid: rules + LLM) | 96.0% ± 0.1 | 68.2% ± 1.2 | 0.0% ± 0.0 |

| Error type (Agent (hybrid: rules + LLM)) | Known formats | Unseen formats |
|---|---|---|
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 | 24.1% ± 2.4 |
| Date format (day-first variants) | 100.0% ± 0.0 | 70.4% ± 4.1 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 | 78.3% ± 1.4 |
| Category spelling variants | 100.0% ± 0.0 | 25.9% ± 3.0 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 | 0.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 94.7% ± 5.3 | 56.9% ± 1.4 |

| Error type (Agent (rule planner)) | Known formats | Unseen formats |
|---|---|---|
| Money format (₹1,20,000 / 1.2 lakh / 120k) | 100.0% ± 0.0 | 24.1% ± 2.4 |
| Date format (day-first variants) | 100.0% ± 0.0 | 70.4% ± 4.1 |
| Phone format (+91, 0, dashes) | 100.0% ± 0.0 | 78.3% ± 1.4 |
| Category spelling variants | 100.0% ± 0.0 | 17.7% ± 0.6 |
| Number with unit ('34 yrs') | 100.0% ± 0.0 | 0.0% ± 0.0 |
| Old city name, NOT in rule dictionary | 0.0% ± 0.0 | 0.0% ± 0.0 |

### Fairness: who loses data? (tier-3 towns have 2x the error rate in this simulation)

| Method | Kept: tier 1 | Kept: tier 2 | Kept: tier 3 | Fixed: tier 1 | Fixed: tier 3 |
|---|---|---|---|---|---|
| No cleaning | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Naive pandas (dropna) | 75.5% ± 0.3 | 77.1% ± 0.7 | 55.1% ± 0.3 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Agent, single pass | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.6% ± 0.0 | 99.0% ± 0.0 |
| Agent (rule planner) | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.6% ± 0.0 | 99.0% ± 0.0 |
| Agent (hybrid: rules + LLM) | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.8% ± 0.2 |

### Error-rate sweep (3 datasets per rate)

| Error rate | No cleaning: correct cells | Naive pandas (dropna): correct cells | Agent (rule planner): correct cells | Naive: applicants kept |
|---|---|---|---|---|
| 3% | 96.1% ± 0.0 | 81.8% ± 0.0 | 98.8% ± 0.0 | 84.0% ± 0.0 |
| 6% | 92.4% ± 0.0 | 67.3% ± 0.0 | 97.6% ± 0.0 | 70.8% ± 0.0 |
| 12% | 84.7% ± 0.0 | 43.9% ± 0.0 | 94.7% ± 0.0 | 48.7% ± 0.0 |
| 20% | 75.0% ± 0.0 | 26.1% ± 0.0 | 90.6% ± 0.0 | 30.9% ± 0.0 |
| 30% | 62.8% ± 0.0 | 12.5% ± 0.0 | 86.1% ± 0.0 | 16.0% ± 0.0 |

| Seconds per run | No cleaning: 0.0 | Naive pandas (dropna): 0.0 | Agent, single pass: 2.6 | Agent (rule planner): 3.3 | Faulty planner + guardrails: 3.8 | Faulty planner, no guardrails: 2.6 | Agent (hybrid: rules + LLM): 3.2 |

### LLM planning everything by itself (original recorded run, 2 datasets)

| Method | Correct cells | Fixable errors fixed | Bad values blanked | Clean cells damaged | Seconds per run |
|---|---|---|---|---|---|
| Agent (LLM planner) | 93.0% ± 0.1 | 10.4% ± 0.6 | 33.8% ± 1.0 | 0.0% | 60 |

Left to plan alone, the model mostly proposed standardising placeholders and a few category merges, then stopped; it fixed about 10% of fixable errors (rules: 99%). It damaged nothing, because every step passed validation and the guardrails. Measured live while recording, before a prompt-ordering fix, so it is not replayed here.
