"""Plan step. RulePlanner is deterministic; LLMPlanner asks a language model; HybridPlanner uses rules and
asks the LLM only about rare category values the rules could not resolve."""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from .categories import MULTI
from .llm import extract_json
from .tools import OPTIONAL, TOOL_DOCS, TOOLS

ORDER = ["fix_encoding", "standardize_missing", "extract_number", "parse_number", "parse_date", "normalize_phone", "normalize_pincode", "normalize_email",
         "normalize_text", "map_values", "fix_range", "flag_outliers", "impute", "drop_duplicates"]


@dataclass
class Step:
    tool: str
    args: dict
    reason: str = ""
    source: str = "rule"
    meta: dict = field(default_factory=dict)

    def key(self):
        """Hashable identity of the step, even when an LLM sends malformed arguments."""
        if self.meta.get("llm_suggestion"):
            return ("map_values_llm", str(self.args.get("col")))
        col = self.args.get("col") if isinstance(self.args, dict) else None
        if col is None and isinstance(self.args, dict):
            sub = self.args.get("subset", [])
            col = ",".join(map(str, sub)) if isinstance(sub, list) else sub
        return (str(self.tool), col if isinstance(col, str) else json.dumps(col, sort_keys=True, default=str))

    def label(self):
        a = {k: v for k, v in self.args.items() if k != "mapping"}
        if "mapping" in self.args:
            a["mapping"] = f"{len(self.args['mapping'])} values"
        return f"{self.tool}({', '.join(f'{k}={v}' for k, v in a.items())})"


def validate(step, df):
    """Returns an error message, or None if the step is well-formed for this DataFrame."""
    if step.tool not in TOOLS:
        return f"unknown tool {step.tool!r}"
    schema = TOOLS[step.tool]
    for arg, typ in schema.items():
        if arg not in step.args:
            if arg in OPTIONAL.get(step.tool, set()):
                continue
            return f"missing argument {arg!r}"
        if not isinstance(step.args[arg], typ):
            return f"argument {arg!r} has wrong type"
    extra = set(step.args) - set(schema)
    if extra:
        return f"unexpected arguments {sorted(extra)}"
    if "col" in step.args and step.args["col"] not in df.columns:
        return f"unknown column {step.args['col']!r}"
    if "subset" in step.args and (not step.args["subset"] or any(c not in df.columns for c in step.args["subset"])):
        return "bad subset"
    if step.tool == "impute" and step.args["strategy"] not in ("median", "mode"):
        return "strategy must be median or mode"
    return None


def _sorted(steps):
    return sorted(steps, key=lambda s: ORDER.index(s.tool))


class RulePlanner:
    name = "rule"

    def __init__(self, llm=None, impute=False):
        self.llm, self.impute = llm, impute
        self.llm_errors = []

    def plan(self, df, prof, issues, done):
        steps = []
        for iss in issues:
            c = iss.column
            k, why = iss.kind, f"{iss.count} values: {iss.kind.replace('_', ' ')}"
            if k == "encoding_artifacts":
                steps.append(Step("fix_encoding", {"col": c}, f"{iss.count} values contain export junk such as \\xc2\\xa0"))
            elif k == "disguised_missing":
                steps.append(Step("standardize_missing", {"col": c}, why))
            elif k == "number_stored_as_text":
                steps.append(Step("parse_number", {"col": c}, f"{iss.detail['non_plain']} values in non-standard formats; column stored as text"))
            elif k == "date_format":
                order = iss.detail.get("order", "unknown")
                ev = iss.detail.get("evidence", {})
                note = {"dayfirst": f"day-first: {ev.get('dayfirst', 0)} dates have a day above 12",
                        "monthfirst": f"month-first: {ev.get('monthfirst', 0)} dates have a day above 12 in second place",
                        "unknown": "order not provable from the data; assuming Indian day-first"}[order]
                if ev.get("dayfirst") and ev.get("monthfirst"):
                    note += f" (conflicting evidence {ev})"
                steps.append(Step("parse_date", {"col": c, "dayfirst": order != "monthfirst"}, f"{why}; {note}"))
            elif k in ("phone_format", "pincode_format", "email_format"):
                steps.append(Step(f"normalize_{k.split('_')[0]}", {"col": c}, why))
            elif k == "text_format":
                steps.append(Step("normalize_text", {"col": c, "case": iss.detail["case"]}, why))
            elif k == "number_with_unit":
                steps.append(Step("extract_number", {"col": c, "unit": iss.detail["unit"]},
                                  f"every value is a number in '{iss.detail['unit']}'; store the number, unit noted here"))
            elif k == "mixed_units":
                pass  # reported, not fixed: converting needs knowledge of how the units relate (ask a human or an LLM)
            elif k == "category_variants":
                mapping = {raw: v[0] for raw, v in iss.detail["mapping"].items()}
                reasons = {raw: v[1] for raw, v in iss.detail["mapping"].items()}
                if mapping:
                    steps.append(Step("map_values", {"col": c, "mapping": mapping},
                                      f"merge {len(mapping)} variant spellings into canonical values", "rule",
                                      {"reasons": reasons}))
                if self.llm and iss.detail["unresolved"] and ("map_values_llm", str(c)) not in done:  # ask once per column
                    # LLM suggestions are a SEPARATE step: the guardrails judge them on their own, and a bad
                    # suggestion can never cause the (reliable) rule merges to be thrown away with it
                    extra = self._ask_llm(c, iss.detail["unresolved"], iss.detail["canonical"])
                    # the same safety rules apply to LLM answers: never collapse a multi-valued entry
                    # ('Bangalore / USA') into one value; and follow the rule renames ('Bangalore' -> 'Bengaluru')
                    extra = {k: mapping.get(v, v) for k, v in extra.items()
                             if not (MULTI.search(k) and not MULTI.search(v))}
                    extra = {k: v for k, v in extra.items() if k != v}
                    if extra:
                        steps.append(Step("map_values", {"col": c, "mapping": extra},
                                          f"LLM suggests {len(extra)} values are aliases or misspellings", "llm",
                                          {"reasons": {k: f"LLM: alias or misspelling of {v}" for k, v in extra.items()},
                                           "llm_suggestion": True}))
            elif k == "out_of_range":
                steps.append(Step("fix_range", {"col": c, "lo": iss.detail["lo"], "hi": iss.detail["hi"]},
                                  f"{iss.count} values outside plausible range [{iss.detail['lo']}, {iss.detail['hi']}]"))
            elif k == "outlier":
                steps.append(Step("flag_outliers", {"col": c, "z": iss.detail["z"], "above": iss.detail["above"]},
                                  f"{iss.count} values are 10x above the normal range (likely extra zeros or wrong unit)"))
            elif k == "duplicates":
                steps.append(Step("drop_duplicates", {"subset": iss.detail["subset"]}, f"{iss.count} duplicate rows"))
        if self.impute and not steps:
            from .parsers import is_blank
            for c in df.columns:
                if df[c].map(is_blank).any():
                    num = str(df[c].dtype).startswith(("float", "int"))
                    steps.append(Step("impute", {"col": c, "strategy": "median" if num else "mode"}, "fill remaining gaps"))
        return _sorted([s for s in steps if s.key() not in done])

    def _ask_llm(self, col, unresolved, canonical):
        prompt = (f"Column '{col}' of an Indian dataset has these canonical values:\n{json.dumps(canonical)}\n\n"
                  f"These rare values did not match any of them:\n{json.dumps(unresolved[:60])}\n\n"
                  "For each rare value that is an alias, old name, abbreviation or misspelling of a canonical value, "
                  "map it to that canonical value. Leave out values you are not sure about. "
                  'Reply with JSON only: {"mapping": {"rare value": "canonical value"}}')
        try:
            reply = extract_json(self.llm.complete([{"role": "user", "content": prompt}]))
            m = reply.get("mapping", {})
            return {str(k): v for k, v in m.items() if k in unresolved and v in canonical}
        except Exception as e:  # noqa: BLE001 - any LLM failure falls back to rules only
            self.llm_errors.append(str(e))
            return {}


SYSTEM = """You are a careful data-cleaning agent for Indian tabular data.
You may ONLY use these tools (no other code):
{tools}
Rules: dates in India are day-first. Money may be written as '₹1,20,000', '1.2 lakh', '2 crore', '120k'.
Prefer fixing to deleting. Never drop rows except exact duplicates of an ID. Do not repeat a step already done.
Reply with JSON only: {{"steps": [{{"tool": "...", "args": {{...}}, "reason": "..."}}]}}. Return {{"steps": []}} when the data is clean."""


class LLMPlanner:
    """The LLM sees the profile and issues and proposes steps. Invalid steps are rejected; if the reply is not
    usable at all, the agent falls back to the rule planner for that iteration."""
    name = "llm"

    def __init__(self, llm, fallback=None):
        self.llm, self.fallback = llm, fallback or RulePlanner()
        self.llm_errors = []

    def plan(self, df, prof, issues, done):
        tools = "\n".join(f"- {t}({', '.join(TOOLS[t])}): {TOOL_DOCS[t]}" for t in TOOLS)
        cols = [dict(column=p.name, inferred_type=p.kind, missing=p.n_null, unique=p.n_unique,
                     dtype=str(df[p.name].dtype), top_values=[str(v) for v, _ in p.top[:8]]) for p in prof.values()]
        issue_view = []
        for i in issues:
            d = i.as_dict()
            if i.kind == "category_variants":
                d["canonical_values"] = i.detail["canonical"][:40]
                d["unmatched_rare_values"] = i.detail["unresolved"][:40]
            issue_view.append(d)
        user = (f"Columns:\n{json.dumps(cols, ensure_ascii=False)}\n\nDetected issues:\n{json.dumps(issue_view, ensure_ascii=False)}\n\n"
                f"Steps already done: {json.dumps(sorted(list(k) for k in done))}\nPlan the next cleaning steps.")
        try:
            reply = extract_json(self.llm.complete([{"role": "system", "content": SYSTEM.format(tools=tools)},
                                                    {"role": "user", "content": user}]))
            raw = reply.get("steps", [])
            steps = [Step(str(s.get("tool", "")), s.get("args") if isinstance(s.get("args"), dict) else {}, str(s.get("reason", "")), "llm")
                     for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []
        except Exception as e:  # noqa: BLE001
            self.llm_errors.append(str(e))
            return [Step(s.tool, s.args, s.reason + " [fallback: LLM reply unusable]", "fallback", s.meta)
                    for s in self.fallback.plan(df, prof, issues, done)]
        return [s for s in steps if s.key() not in done]
