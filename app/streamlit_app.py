"""CleanAgent - run: streamlit run app/streamlit_app.py   (Windows: double-click run_app.bat)"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from cleanagent.agent import CleaningAgent, Guard
from cleanagent.data import corrupt, make_clean
from cleanagent.llm import CachedLLM, OpenAICompatLLM
from cleanagent.explain import attribution, explain_cell, explain_steps
from cleanagent.planner import LLMPlanner, RulePlanner
from cleanagent.report import changes_frame, to_markdown

st.set_page_config(page_title="CleanAgent", layout="wide")
st.title("CleanAgent: an AI agent that cleans messy Indian data")
st.caption("Observe → plan → act → verify. The agent may only call 14 whitelisted cleaning tools, every step passes "
           "guardrails, and every changed cell is logged with a reason. Runs fully offline on a laptop CPU.")

DATASETS = {
    "Real: Indian startup funding (3,044 rows)": ROOT / "data/real/startup_funding.csv",
    "Real: Bengaluru house prices (13,320 rows)": ROOT / "data/real/bengaluru_house.csv",
    "Real: CarDekho used cars (8,128 rows, blind hold-out)": ROOT / "data/real/cardekho_cars.csv",
    "Synthetic: messy loan applications (with known truth)": None,
    "Upload your own CSV": "upload",
}

with st.sidebar:
    st.header("Planner")
    mode = st.radio("Who plans the steps?", ["Rules (offline)", "Hybrid: rules + LLM", "LLM"],
                    help="Hybrid asks the LLM only about rare values the rules could not match: few, small prompts, "
                         "which suits a laptop CPU.")
    if mode != "Rules (offline)":
        replay = st.checkbox("Use recorded LLM answers (no model needed)", True,
                             help="Replays answers saved in llm_cache.json. Untick to call a running model live.")
        base_url = st.text_input("LLM server", "http://localhost:11434/v1", help="Ollama: http://localhost:11434/v1 · LM Studio: http://localhost:1234/v1")
        model = st.text_input("Model", "qwen2.5:1.5b")
    st.header("Options")
    guard_on = st.checkbox("Guardrails", True)
    impute = st.checkbox("Fill remaining gaps (median / most common)", False)

tab_demo, tab_syn, tab_real, tab_how = st.tabs(["Live demo", "Results: synthetic data", "Results: real data", "How it works"])

with tab_demo:
    choice = st.selectbox("Dataset", list(DATASETS))
    src = DATASETS[choice]
    if src == "upload":
        up = st.file_uploader("CSV file", type="csv")
        if not up:
            st.stop()
        dirty = pd.read_csv(up, dtype=str).astype(object)
    elif src is None:
        c1, c2 = st.columns(2)
        rate = c1.slider("Share of cells corrupted", 0.02, 0.3, 0.08, 0.01)
        seed = c2.number_input("Seed", 0, 999, 7)

        @st.cache_data
        def demo(rate, seed):
            clean, tier = make_clean(1500, seed)
            d, _, _ = corrupt(clean, tier, rate=rate, seed=seed)
            return d.reset_index(drop=True)
        dirty = demo(rate, int(seed))
    else:
        dirty = pd.read_csv(src, dtype=str, encoding="utf-8").astype(object)
    st.dataframe(dirty.head(12), width="stretch")

    if st.button("Run the agent", type="primary"):
        planner = RulePlanner(impute=impute)
        if mode != "Rules (offline)":
            llm = CachedLLM(None if replay else OpenAICompatLLM(base_url, model), ROOT / "llm_cache.json",
                            mode="replay" if replay else "auto")
            planner = RulePlanner(llm=llm, impute=impute) if mode.startswith("Hybrid") else LLMPlanner(llm, RulePlanner(impute=impute))
        t0 = time.time()
        with st.spinner("Observing, planning, cleaning... (up to ~30 s for the house data on a laptop)"):
            res = CleaningAgent(planner, guard=Guard() if guard_on else None).run(dirty)
        st.session_state.update(res=res, secs=time.time() - t0, errs=getattr(planner, "llm_errors", []), shown_for=choice)
        if mode != "Rules (offline)" and replay and not (ROOT / "llm_cache.json").exists():
            st.warning("No recorded LLM answers found (llm_cache.json); the rules were used instead.")

    res = st.session_state.get("res")
    if res is not None and st.session_state.get("shown_for") == choice:
        if st.session_state["errs"]:
            st.info(f"{len(st.session_state['errs'])} LLM question(s) had no recorded answer; rules were used for those.")
        log = pd.DataFrame(res.log)
        m = st.columns(6)
        m[0].metric("Issues found", len(res.issues_before)); m[1].metric("Issues left", len(res.issues_after))
        m[2].metric("Cells changed", sum(1 for c in res.changes if c[1] is not None))
        m[3].metric("Rows dropped", sum(1 for c in res.changes if c[1] is None))
        m[4].metric("Steps rejected", int((log.status != "applied").sum()) if len(log) else 0)
        m[5].metric("Seconds", f"{st.session_state['secs']:.1f}")
        t1, t2, t3, t4, t5 = st.tabs(["Agent trace", "Before → after by column", "Cleaned data", "Issues left", "Why? (explainability)"])
        with t1:
            st.dataframe(log[["iteration", "step", "source", "status", "n_changed", "n_blanked", "rows_dropped", "reason", "why"]],
                         hide_index=True, width="stretch")
            st.caption("Iteration 2+ exists because the agent re-profiles after acting: ranges can only be checked once text "
                       "has become numbers. 'rejected' = stopped by a guardrail.")
        with t2:
            ch = changes_frame(res)
            cols = sorted(ch.column.dropna().unique().tolist())
            if cols:
                col = st.selectbox("Column", cols)
                v = ch[ch.column == col].astype(str).groupby(["before", "after", "reason"]).size().reset_index(name="rows")
                st.dataframe(v.sort_values("rows", ascending=False).head(300), hide_index=True, width="stretch")
        with t3:
            st.dataframe(res.df.head(200), width="stretch")
            a, b = st.columns(2)
            a.download_button("Download cleaned CSV", res.df.to_csv(index=False).encode("utf-8-sig"), "cleaned.csv")
            b.download_button("Download cleaning report", to_markdown(res), "cleaning_report.md")
        with t4:
            st.write("Left for a human or an LLM (the rules deliberately do not guess these):")
            st.dataframe(pd.DataFrame(res.issues_after), hide_index=True, width="stretch")
        with t5:
            st.write("Every decision the agent made: **what triggered it**, the **evidence** from your data, how much of the "
                     "result it is **responsible for**, and how far to **trust** it. (SHAP explains a model's prediction; "
                     "this agent makes decisions, so each decision is explained instead.)")
            ex = explain_steps(res)
            st.dataframe(ex, hide_index=True, width="stretch")
            st.caption("Confidence is a rule, not a probability: High = deterministic reformat that passed all guardrails; "
                       "Medium = LLM-chosen, blanked unreadable values, or removed duplicates; Blocked = a guardrail stopped it.")
            at = attribution(res)
            if len(at):
                st.write("**Which operations did the work** (share of all changed cells):")
                st.bar_chart(at.set_index("tool")["share_%"])
            ch2 = changes_frame(res)
            if len(ch2):
                st.write("**Trace one cell:** pick a row and column to see every change made to it and why.")
                c1, c2 = st.columns(2)
                ccol = c1.selectbox("Column ", sorted(ch2.column.dropna().unique().tolist()), key="why_col")
                rows = ch2[ch2.column == ccol].row.tolist() if "row" in ch2.columns else []
                if rows:
                    rr = c2.selectbox("Row ", rows[:500], key="why_row")
                    st.dataframe(pd.DataFrame(explain_cell(res, rr, ccol)), hide_index=True, width="stretch")


def show_report(md_path, images):
    if md_path.exists():
        st.markdown(md_path.read_text(encoding="utf-8"))
    for img in images:
        p = ROOT / "reports" / img
        if p.exists():
            st.image(str(p))


with tab_syn:
    st.write("3,000 generated loan applications with errors injected on purpose, so every fix can be checked against the truth.")
    llm_md = ROOT / "reports/RESULTS_llm.md"
    if llm_md.exists():
        with st.expander("With a real LLM: gpt-oss-20b, recorded on an M5 MacBook Air (click to open)"):
            st.markdown(llm_md.read_text(encoding="utf-8"))
    show_report(ROOT / "reports/RESULTS.md", ["cell_accuracy.png", "by_error_type.png", "unseen_formats.png",
                                              "fairness_retention.png", "guardrails.png", "error_rate_sweep.png"])

with tab_real:
    st.write("Three real Indian datasets: two used during development, one kept as a blind hold-out.")
    show_report(ROOT / "reports/real/REAL_RESULTS.md", [])
    show_report(ROOT / "reports/real/LLM_REAL_RESULTS.md", [])
    err = ROOT / "reports/real/startup_agent_errors.csv"
    if err.exists() and err.stat().st_size > 5:
        with st.expander("Agent errors on the hand-labelled startup rows"):
            st.dataframe(pd.read_csv(err), hide_index=True)

with tab_how:
    st.markdown("""
**Observe.** Profile every column: guess its type (money, date, phone, PIN, category, free text...) and list issues.

**Plan.** Rules turn issues into steps. In hybrid mode, values the rules cannot match are sent to an LLM
(e.g. `Poona` → `Pune`), and its answer is accepted only if it maps to a value that already exists in the column.

**Act.** Only whitelisted tools run, never code written by the LLM.

**Verify.** Guardrails reject any step that would drop >15% of rows, blank >25% of a column or rewrite >50% of it.
Every change is logged with before, after, tool and reason.

**Repeat** until nothing fixable is left (at most 4 rounds).

Safety rules learned from real data: never merge two common values (`Female`/`Male`), never merge different codes or
numbers (`Series E`/`Series D`, `Phase 1`/`Phase 2`), never merge a multi-city entry into one city
(`Bangalore / USA`), never mix units (`kmpl`/`km/kg`), and blank only order-of-magnitude outliers (real luxury
houses are kept).
""")
