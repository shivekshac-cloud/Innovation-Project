import json

import numpy as np
import pandas as pd
import pytest

from cleanagent.agent import CleaningAgent, Guard
from cleanagent.categories import category_mapping
from cleanagent.data import corrupt, make_clean
from cleanagent.parsers import normalize_phone, normalize_pincode, parse_date, parse_number
from cleanagent.planner import LLMPlanner, RulePlanner, Step, validate
from cleanagent.score import score


@pytest.mark.parametrize("raw,val", [("₹1,20,000", 120000), ("Rs. 120000/-", 120000), ("1.2 lakh", 120000),
                                     ("1.2L", 120000), ("120k", 120000), ("2 crore", 2e7), ("INR 1,20,000/-", 120000),
                                     ("34 yrs", 34), ("NA", np.nan), ("abc", np.nan)])
def test_parse_number_indian_formats(raw, val):
    out = parse_number(raw)
    assert (np.isnan(out) and np.isnan(val)) or out == pytest.approx(val)


def test_dates_are_day_first():
    assert parse_date("05/03/2024") == "2024-03-05"
    assert parse_date("5 Mar 2024") == parse_date("05-Mar-24") == "2024-03-05"
    assert parse_date("31/31/2024") != parse_date("31/31/2024")  # nan


def test_phone_and_pincode():
    for raw in ["+91 98765 43210", "098765-43210", "919876543210", "(+91) 98765 43210"]:
        assert normalize_phone(raw) == "9876543210"
    assert normalize_phone("12345") != normalize_phone("12345")
    assert normalize_pincode("400 001") == normalize_pincode("400001.0") == "400001"


def test_category_mapping_merges_case_alias_and_typo():
    s = pd.Series(["Mumbai"] * 20 + ["Pune"] * 20 + [" mumbai", "Bombay", "Mumabi", "Poona"])
    mapping, unresolved, canon = category_mapping(s)
    assert mapping[" mumbai"][0] == mapping["Bombay"][0] == mapping["Mumabi"][0] == "Mumbai"
    assert "Poona" in unresolved  # not in the rule dictionary: left for an LLM


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages):
        return self.reply


def test_hybrid_planner_uses_llm_only_for_valid_canonical_targets():
    s = pd.DataFrame({"city": ["Mumbai"] * 20 + ["Pune"] * 20 + ["Poona", "Cochin"]})
    llm = FakeLLM('```json\n{"mapping": {"Poona": "Pune", "Cochin": "Kochi"}}\n```')
    res = CleaningAgent(RulePlanner(llm=llm)).run(s)
    assert res.df.city.iloc[40] == "Pune"
    assert res.df.city.iloc[41] == "Cochin"  # Kochi is not a value in this column, so the answer is ignored


def test_invalid_llm_steps_are_refused_and_bad_reply_falls_back():
    df = pd.DataFrame({"age": ["34", "NA", "41 yrs"] * 10})
    bad = json.dumps({"steps": [{"tool": "exec_python", "args": {"code": "import os"}},
                                {"tool": "parse_number", "args": {"col": "no_such_column"}},
                                {"tool": "parse_number", "args": {"col": "age"}, "reason": "ok"}]})
    res = CleaningAgent(LLMPlanner(FakeLLM(bad)), max_iters=1).run(df)
    status = [r["status"] for r in res.log]
    assert status[:2] == ["invalid", "invalid"] and "applied" in status
    res2 = CleaningAgent(LLMPlanner(FakeLLM("sorry, I cannot help")), max_iters=1).run(df)
    assert all(r["source"] == "fallback" for r in res2.log)
    assert res2.df.age.dropna().tolist()[:2] == [34.0, 41.0]


def test_guard_rejects_destructive_steps():
    df = pd.DataFrame({"city": ["Mumbai", "Pune", "Delhi"] * 20, "amount": ["100"] * 60})
    guard = Guard()

    class Bad(RulePlanner):
        def plan(self, df, prof, issues, done):
            return [] if done else [Step("drop_duplicates", {"subset": ["city"]}), Step("parse_date", {"col": "amount"}),
                                    Step("map_values", {"col": "city", "mapping": {"Pune": "Mumbai", "Delhi": "Mumbai"}})]
    res = CleaningAgent(Bad(), guard=guard).run(df)
    assert [r["status"] for r in res.log] == ["rejected"] * 3
    assert res.df.equals(df)


def test_end_to_end_agent_beats_no_cleaning():
    clean, tier = make_clean(800, seed=5)
    dirty, ledger, dup_of = corrupt(clean, tier, rate=0.06, seed=5)
    res = CleaningAgent(RulePlanner()).run(dirty)
    s_agent, s_none = score(res.df, clean, tier, ledger, dup_of), score(dirty, clean, tier, ledger, dup_of)
    assert s_agent["cell_accuracy"] > s_none["cell_accuracy"] + 0.03
    assert s_agent["retention"] == 1.0 and s_agent["duplicate_groups_resolved"] > 0.9
    assert s_agent["collateral_damage"] < 0.005
    assert len(res.issues_after) < len(res.issues_before)
