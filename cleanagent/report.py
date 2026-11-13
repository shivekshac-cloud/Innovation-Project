"""Explain step: human-readable audit of everything the agent did."""
import pandas as pd


def changes_frame(result, limit=None):
    rows = result.changes[:limit] if limit else result.changes
    return pd.DataFrame(rows, columns=["row", "column", "before", "after", "tool", "reason"])


def to_markdown(result, title="Data cleaning report"):
    log = pd.DataFrame(result.log)
    L = [f"# {title}", "", f"Issues found: {len(result.issues_before)}. Issues left: {len(result.issues_after)}.",
         f"Cells changed: {sum(1 for c in result.changes if c[1] is not None)}. "
         f"Rows dropped: {sum(1 for c in result.changes if c[1] is None)}.", "", "## Steps", "",
         "| # | Iter | Step | Status | Changed | Blanked | Why |", "|---|---|---|---|---|---|---|"]
    for k, r in log.iterrows():
        why = r["why"] or r["reason"]
        L.append(f"| {k + 1} | {r['iteration']} | `{r['step']}` | {r['status']} | {r['n_changed']} | {r['n_blanked']} | {why} |")
    L += ["", "## Examples of changes", ""]
    for r in log.itertuples():
        if r.status == "applied" and r.examples:
            ex = "; ".join(f"`{o}` -> `{n}`" for _, o, n in r.examples[:3])
            L.append(f"- **{r.step}**: {ex}")
    if result.issues_after:
        L += ["", "## Issues still open", ""] + [f"- {i['column']}: {i['kind']} ({i['count']})" for i in result.issues_after]
    return "\n".join(L) + "\n"
