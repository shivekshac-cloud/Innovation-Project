"""CleanAgent - run: streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from cleanagent.agent import CleaningAgent
from cleanagent.data import corrupt, make_clean
from cleanagent.llm import OpenAICompatLLM
from cleanagent.planner import LLMPlanner, RulePlanner
from cleanagent.report import changes_frame, to_markdown

st.set_page_config(page_title="CleanAgent", layout="wide")
st.title("CleanAgent: an AI agent that cleans messy Indian data")
st.caption("Observe, plan, act, verify. The agent can only call whitelisted cleaning tools, every step is checked by "
           "guardrails, and every changed cell is logged with a reason.")

with st.sidebar:
    st.header("Planner")
    mode = st.radio("Who plans the steps?", ["Rules (offline)", "Hybrid: rules + LLM", "LLM"],
                    help="Hybrid asks the LLM only about rare category values the rules could not match.")
    if mode != "Rules (offline)":
        base_url = st.text_input("LLM server (OpenAI-compatible)", "http://localhost:1234/v1",
                                 help="LM Studio: http://localhost:1234/v1. Ollama: http://localhost:11434/v1")
        model = st.text_input("Model name", "local-model")
        api_key = st.text_input("API key (hosted APIs only)", type="password")
    st.header("Options")
    impute = st.checkbox("Fill remaining gaps (median / most common)", False)
    guard_on = st.checkbox("Guardrails", True)

src = st.radio("Data", ["Demo: messy loan applications", "Upload CSV"], horizontal=True)
if src == "Upload CSV":
    up = st.file_uploader("CSV file", type="csv")
    if not up:
        st.stop()
    dirty = pd.read_csv(up, dtype=str).astype(object)
else:
    c1, c2 = st.columns(2)
    rate = c1.slider("Share of cells corrupted", 0.02, 0.3, 0.08, 0.01)
    seed = c2.number_input("Seed", 0, 999, 7)

    @st.cache_data
    def demo(rate, seed):
        clean, tier = make_clean(1500, seed)
        d, _, _ = corrupt(clean, tier, rate=rate, seed=seed)
        return d.reset_index(drop=True)
    dirty = demo(rate, int(seed))

st.subheader("Input")
st.dataframe(dirty.head(15), width="stretch")

if st.button("Run the agent", type="primary"):
    planner = RulePlanner(impute=impute)
    if mode != "Rules (offline)":
        llm = OpenAICompatLLM(base_url, model, api_key)
        planner = RulePlanner(llm=llm, impute=impute) if mode.startswith("Hybrid") else LLMPlanner(llm, RulePlanner(impute=impute))
    from cleanagent.agent import Guard
    with st.spinner("Observing, planning, cleaning..."):
        res = CleaningAgent(planner, guard=Guard() if guard_on else None).run(dirty)
    st.session_state["res"] = res
    st.session_state["llm_errors"] = getattr(planner, "llm_errors", [])

res = st.session_state.get("res")
if res is None:
    st.stop()
if st.session_state.get("llm_errors"):
    st.warning(f"LLM problems (rules were used instead): {st.session_state['llm_errors'][0][:200]}")

log = pd.DataFrame(res.log)
cells = sum(1 for c in res.changes if c[1] is not None)
dropped = sum(1 for c in res.changes if c[1] is None)
m = st.columns(5)
m[0].metric("Issues found", len(res.issues_before)); m[1].metric("Issues left", len(res.issues_after))
m[2].metric("Cells changed", cells); m[3].metric("Rows dropped", dropped)
m[4].metric("Steps rejected", int((log.status != "applied").sum()) if len(log) else 0)

t1, t2, t3, t4 = st.tabs(["Agent trace", "Changed cells", "Cleaned data", "Issues before / after"])
with t1:
    st.dataframe(log[["iteration", "step", "source", "status", "n_changed", "n_blanked", "rows_dropped", "reason", "why"]],
                 hide_index=True, width="stretch")
    st.caption("Iteration 2+ happens because the agent re-profiles the data after acting. For example, ranges can only be "
               "checked once text has been parsed into numbers.")
with t2:
    ch = changes_frame(res)
    col = st.selectbox("Column", ["(all)"] + sorted(ch.column.dropna().unique().tolist()))
    view = ch if col == "(all)" else ch[ch.column == col]
    st.dataframe(view.astype(str).head(500), hide_index=True, width="stretch")
with t3:
    st.dataframe(res.df.head(200), width="stretch")
    a, b = st.columns(2)
    a.download_button("Download cleaned CSV", res.df.to_csv(index=False), "cleaned.csv")
    b.download_button("Download cleaning report (Markdown)", to_markdown(res), "cleaning_report.md")
with t4:
    a, b = st.columns(2)
    a.write("Before"); a.dataframe(pd.DataFrame(res.issues_before), hide_index=True, width="stretch")
    b.write("After"); b.dataframe(pd.DataFrame(res.issues_after), hide_index=True, width="stretch")

rp = ROOT / "reports" / "results.json"
if rp.exists():
    with st.expander("Benchmark results"):
        st.markdown((ROOT / "reports" / "RESULTS.md").read_text())
