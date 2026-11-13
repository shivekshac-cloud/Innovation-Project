# Notes for evaluators

Please read this before judging the numbers. It lists how the evaluation was done and where it is weaker than it looks.

**Data.**
- *Synthetic:* 3,000 generated loan applications with 14 injected error types; every injected error is recorded, so
  accuracy is exact.
- *Real:* three public Indian datasets from Kaggle (Indian Startup Funding, Bengaluru House Prices, CarDekho used
  cars), taken from public GitHub mirrors because Kaggle was not reachable from the build environment. Copies are in
  `data/real/`.

**Gold labels (startup file).** The correct values for 140 sampled rows were written before the improved agent was
run on that file, following the written rules in `data/real/LABELING_GUIDE.md`. They were drafted with an AI
assistant and checked row by row against those rules. Some rules are judgement calls that the agent's dictionary
shares (`New Delhi` → `Delhi`, `Bangalore` → `Bengaluru`, `Gurgaon` → `Gurugram`). The table therefore also shows a
lenient city score that accepts old names.

**Development vs blind results.**
- The startup and house files were inspected while the rules were improved, so their scores are development scores.
  The agent's 100% on the startup gold rows should be read that way.
- The CarDekho file is the only blind test: the code was frozen (fingerprint `7a376f4a6226`) before it ran, and its
  output was reported unchanged. It made only trivial, correct fixes and missed unit columns (`kmpl`, `CC`, `bhp`).
  A unit tool was added afterwards and is reported separately as a post-hoc fix.

**Generalisation.** On synthetic error formats the rules were not written for, the agent fixed 66% of fixable errors
(99% on known formats), without damaging any correct value.

**Downstream.** On real house prices, cleaning did not improve the price model by itself. With the same row filter as
a standard tutorial, accuracy was equal (−0.1 MAPE points) and coverage was 1.6% higher.

**LLM.** The LLM and hybrid planners are implemented and tested with a simulated model. Results with a real model
appear in `reports/RESULTS_llm.md` only if that file exists; otherwise, real-model performance has not been measured.

**Tools used.** Python, pandas, scikit-learn, Streamlit. Development was assisted by an AI coding assistant.
