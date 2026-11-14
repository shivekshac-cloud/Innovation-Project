from __future__ import annotations

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


def test_fuzzy_matching_never_merges_two_common_values_or_numbered_or_multi_values():
    s = pd.Series(["Male"] * 30 + ["Female"] * 25 + ["Madhya Pradesh"] * 10 + ["Andhra Pradesh"] * 12
                  + ["BTM 1st Stage"] * 9 + ["BTM 2nd Stage"] * 2 + ["Bangalore"] * 20 + ["Bangalore / USA"])
    mapping, _, _ = category_mapping(s)
    assert "Female" not in mapping and "Madhya Pradesh" not in mapping
    assert "BTM 2nd Stage" not in mapping and "Bangalore / USA" not in mapping
    s2 = pd.Series(["Series D"] * 12 + ["Series E", "Seed Funding Round"] + ["Seed Funding"] * 30 + ["21-Jun"] * 9 + ["21-Jan"]
                   + ["Seed/ Angel Funding"] * 10 + ["Seed / Angle Funding"])
    m2, _, _ = category_mapping(s2)
    assert "Series E" not in m2 and "Seed Funding Round" not in m2 and "21-Jan" not in m2
    assert m2["Seed / Angle Funding"][0] == "Seed/ Angel Funding"


def test_real_world_artifacts():
    from cleanagent.parsers import fix_text, date_order
    assert parse_date("05/072018") == "2018-07-05" and parse_date("12/05.2015") == "2015-05-12"
    assert parse_number("1133 - 1384") == pytest.approx(1258.5)
    assert parse_number("34.46Sq. Meter") == pytest.approx(370.92, abs=0.01)
    assert parse_number("14,342,000+") == 14342000
    assert np.isnan(parse_number("98765-43210"))  # not a range: descending
    assert fix_text("\\\\xc2\\\\xa0Bangalore") == "Bangalore" and fix_text("Seed\\\\nFunding") == "Seed Funding"
    assert fix_text("Byju\\xe2\\x80\\x99s") == "Byju’s"
    assert date_order(["13/01/2024", "05/02/2024"])[0] == "dayfirst"
    assert date_order(["01/13/2024", "02/05/2024"])[0] == "monthfirst"


def test_month_first_file_is_detected():
    df = pd.DataFrame({"order_date": ["03/15/2024", "04/02/2024", "12/25/2023", "01/30/2024"] * 5})
    res = CleaningAgent(RulePlanner()).run(df)
    assert res.df.order_date.iloc[1] == "2024-04-02"  # 2 April, not 4 February


def test_single_unit_columns_are_converted_and_mixed_units_only_flagged():
    df = pd.DataFrame({"engine": ["1248 CC", "1197 CC", "796 CC"] * 30,
                       "mileage": ["23.4 kmpl"] * 80 + ["26.6 km/kg"] * 10})
    res = CleaningAgent(RulePlanner()).run(df)
    assert res.df.engine.iloc[0] == 1248
    assert res.df.mileage.iloc[0] == "23.4 kmpl"  # not converted: kmpl and km/kg are not comparable
    assert any(i["kind"] == "mixed_units" for i in res.issues_after)


def test_llm_cache_records_and_replays(tmp_path):
    from cleanagent.llm import CachedLLM
    path = tmp_path / "cache.json"
    live = CachedLLM(FakeLLM('{"mapping": {"Poona": "Pune"}}'), path)
    s = pd.DataFrame({"city": ["Mumbai"] * 20 + ["Pune"] * 20 + ["Poona"]})
    CleaningAgent(RulePlanner(llm=live)).run(s)
    assert live.misses == 1 and path.exists()
    replay = CachedLLM(None, path, mode="replay")
    res = CleaningAgent(RulePlanner(llm=replay)).run(s)
    assert replay.hits == 1 and res.df.city.iloc[40] == "Pune"


def test_malformed_llm_arguments_are_refused_not_crashing():
    df = pd.DataFrame({"age": ["34", "NA", "41 yrs"] * 10, "city": ["Pune"] * 30})
    bad = json.dumps({"steps": [{"tool": "parse_number", "args": {"col": ["age", "city"]}},
                                {"tool": "drop_duplicates", "args": {"subset": "city"}},
                                {"tool": "map_values", "args": "oops"}, "not a step",
                                {"tool": "parse_number", "args": {"col": "age"}}]})
    res = CleaningAgent(LLMPlanner(FakeLLM(bad)), max_iters=2).run(df)
    assert [r["status"] for r in res.log][:3] == ["invalid", "invalid", "invalid"]
    assert res.df.age.dropna().tolist()[:2] == [34.0, 41.0]


def test_llm_suggestions_obey_the_same_safety_rules():
    s = pd.DataFrame({"city": ["Bengaluru"] * 20 + ["Bangalore"] * 20 + ["Pune"] * 20 + ["Bangalore / USA", "Poona", "Kormangala"]})
    llm = FakeLLM('{"mapping": {"Bangalore / USA": "Bangalore", "Poona": "Pune", "Kormangala": "Bangalore"}}')
    res = CleaningAgent(RulePlanner(llm=llm)).run(s)
    out = res.df.city.tolist()
    assert out[60] == "Bangalore / USA"     # multi-city entry never collapsed, whoever suggests it
    assert out[61] == "Pune"                # LLM world knowledge accepted
    assert out[62] == "Bengaluru"           # LLM target follows the rule rename, old name not brought back
    assert "Bangalore" not in out


def test_explainability_covers_every_step_and_attribution_sums_to_100():
    from cleanagent.explain import attribution, explain_cell, explain_steps
    clean, tier = make_clean(600, seed=3)
    dirty, _, _ = corrupt(clean, tier, rate=0.06, seed=3)
    res = CleaningAgent(RulePlanner()).run(dirty)
    ex = explain_steps(res)
    assert len(ex) == len(res.log) and (ex.evidence.str.len() > 0).all()
    assert set(ex.confidence) <= {"High", "Medium", "Blocked", "Refused", "No effect"}
    assert abs(attribution(res)["share_%"].sum() - 100) < 0.5
    i, c, old, new, *_ = next(ch for ch in res.changes if ch[1] is not None)
    assert explain_cell(res, i, c)[0]["after"] == str(new)


def test_upload_reader_handles_windows_encoding_and_rejects_bad_files(tmp_path):
    import io as _io
    from cleanagent.io import neutralize_formulas, read_csv_safely
    cp = "name,city\nAsha,Pune\nRavi,Chennai \n".encode("cp1252")
    df = read_csv_safely(_io.BytesIO(cp))
    assert list(df.columns) == ["name", "city"] and len(df) == 2
    for bad in (b"", b"onlyonecolumn\n1\n2\n"):
        with pytest.raises(ValueError):
            read_csv_safely(_io.BytesIO(bad))
    d = neutralize_formulas(pd.DataFrame({"a": ["=HYPERLINK(\"http://x\")", "@cmd", "-5", "+91 98765 43210", "ok"]}))
    assert d.a.tolist()[:2] == ["'=HYPERLINK(\"http://x\")", "'@cmd"] and d.a.tolist()[2:] == ["-5", "+91 98765 43210", "ok"]
