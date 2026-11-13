## LLM results on real data (gpt-oss-20b (LM Studio, M5 MacBook Air))

Recorded answers replayed from `llm_cache.json`, so these numbers are reproducible without the model.

| Dataset | Planner | Steps applied | Steps refused by validation/guardrails | LLM questions without a usable answer | Cells changed | Issues left | Gold accuracy (startup, 140 rows) |
|---|---|---|---|---|---|---|---|
| startup_funding | rules | 20 | 0 | 0 | 7048 | 4 | 100.0% |
| startup_funding | hybrid | 23 | 0 | 0 | 7101 | 4 | 98.8% |
| bengaluru_house | rules | 11 | 0 | 0 | 28545 | 1 |  |
| bengaluru_house | hybrid | 11 | 0 | 0 | 28545 | 1 |  |
| cardekho_cars | rules | 10 | 0 | 0 | 15885 | 2 |  |
| cardekho_cars | hybrid | 10 | 0 | 0 | 15885 | 2 |  |

On the startup gold rows, hybrid mode scores slightly lower than rules alone: the LLM merged distinct funding-round labels that the labelling rules keep apart (some defensible, like `Seed Round` → `Seed Funding`; some wrong, like `Crowd funding` → `Debt Funding`). Its useful contribution is world knowledge the rules lack (`Kormangala` → `Bengaluru`; on synthetic data, old city names like `Poona` and `Cochin`).

Development notes (first version vs now): the first hybrid version put LLM merges in the same step as rule merges, so the guardrail rejected the whole city step (91.2%); it also accepted LLM merges that collapse multi-city entries (`Bangalore / USA` → `Bangalore`). LLM suggestions are now a separate step and obey the same safety rules (98.8%).

### What the LLM changed in hybrid mode (accepted answers only)

**startup_funding:** `Seed Round` → `Seed Funding` (7), `Private Equity Round` → `Private Equity` (4), `Seed` → `Seed Funding` (4), `Kormangala` → `Bengaluru` (2), `Corporate Round` → `Debt Funding` (2), `Equity` → `Private Equity` (2), `Crowd funding` → `Debt Funding` (2), `Pre-Series A bridge round` → `Pre-Series A` (2), `Maiden Round` → `Seed Funding` (1), `Seed Funding Round` → `Seed Funding` (1), `Single Venture` → `Private Equity` (1), `Angel` → `Seed Funding` (1), `Angel Round` → `Seed Funding` (1), `Bridge Round` → `Debt Funding` (1), `Inhouse Funding` → `Debt Funding` (1), `Debt` → `Debt Funding` (1), `Mezzanine` → `Debt Funding` (1), `Series B (Extension)` → `Series B` (1), `Equity Based Funding` → `Private Equity` (1), `Private Funding` → `Debt Funding` (1), `Private` → `Private Equity` (1), `Structured Debt` → `Debt Funding` (1), `Term Loan` → `Debt Funding` (1), `Angel Funding` → `Seed Funding` (1), `Series B (includes Debt financing)` → `Series B` (1)

**bengaluru_house:** no merges (nothing the rules left unresolved was a clear variant)

**cardekho_cars:** no merges (nothing the rules left unresolved was a clear variant)

### LLM planning everything by itself (original recorded run on the Mac)

Measured live while recording; not replayable from the cache because of a prompt-ordering bug fixed afterwards.

| Dataset | Time | Unusable replies | Issues left (LLM alone) | Issues left (rules) |
|---|---|---|---|---|
| startup_funding | 90 s | 1 | 4 | 4 |
| bengaluru_house | 28 s | 0 | 4 | 1 |
| cardekho_cars | 74 s | 0 | 2 | 2 |
| synthetic demo (1,500 rows) | 8 s | 0 | 26 | 1 |

The model alone planned few useful steps and left more problems than the rules; on the synthetic benchmark it fixed only about 10% of fixable errors (see `reports/RESULTS_llm.md`). It never damaged data, because every step it proposed went through validation and the guardrails.

