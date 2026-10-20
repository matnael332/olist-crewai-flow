"""Data contract: how raw orders became the clean table, and the contract the scientists must follow."""

import pandas as pd
import streamlit as st

from components import cards, charts, layout
from components.format import num
from services import artifacts as art

layout.page_header(
    "The dataset contract",
    "The Data Analyst crew publishes a versioned contract with the clean data; the Data Scientist "
    "crew must follow it, and the Flow checks it twice before any model is trained.",
    kicker="Data contract · the handoff",
)
contract, cleaning = art.contract(), art.cleaning()
if not contract or not cleaning:
    layout.empty_state("No dataset contract in artifacts/ yet.")
    st.stop()

gates = art.gate_summary(art.validation())
contract_gate = gates.get("gate_contract")
c1, c2, c3, c4 = st.columns(4)
cards.kpi(c1, "Contract version", str(contract["contract_version"]))
cards.kpi(c2, "Rows", num(contract["row_count"]), help="One row per delivered order")
cards.kpi(c3, "Columns", num(len(contract["columns"])))
cards.kpi(
    c4,
    "Contract gate",
    f"{contract_gate['n_passed']}/{contract_gate['n_total']}" if contract_gate else "n/a",
    help="Checks that the clean data matches the contract: columns, types, bounds, allowed "
    "values and every binding constraint.",
)

# Cleaning funnel ------------------------------------------------------------------------------
layout.section("From raw orders to one clean row per order")
steps = [{"step": "Raw orders (all statuses)", "rows": cleaning["rules"][0]["rows_before"]}]
steps += [
    {"step": r["rule"].split("_", 1)[1].replace("_", " "), "rows": r["rows_after"]}
    for r in cleaning["rules"]
    if r["rows_removed"] > 0
]
layout.chart_card(
    "Rows remaining after each cleaning rule",
    charts.funnel(steps),
    f"{num(cleaning['rows_in'])} raw orders became {num(cleaning['rows_out'])} clean delivered "
    f"orders. The other {len(cleaning['rules']) - len(steps) + 1} rules aggregate items, payments "
    "and reviews and derive columns without dropping rows.",
)
with st.expander(f"All {len(cleaning['rules'])} cleaning rules"):
    st.table(
        pd.DataFrame(
            [
                {
                    "Rule": r["rule"].split("_", 1)[0],
                    "What it does": r["description"],
                    "Rows removed": num(r["rows_removed"]),
                }
                for r in cleaning["rules"]
            ]
        ).set_index("Rule")
    )

# Schema ---------------------------------------------------------------------------------------
layout.section("Schema", "Every column, with when its value is known")
when = ["checkout", "post_purchase", "label", "identifier"]
c1, c2 = st.columns([3, 2])
selected = c1.pills("Known at", when, selection_mode="multi", default=when, key="known_at") or when
query = c2.text_input("Search columns", placeholder="e.g. state, days, price", key="col_search")


def _rule(column: dict) -> str:
    if column.get("allowed_values"):
        values = column["allowed_values"]
        shown = ", ".join(map(str, values[:6]))
        return f"one of {len(values)}: {shown}{', ...' if len(values) > 6 else ''}"
    bounds = column.get("bounds") or {}
    parts = [
        f"≥ {bounds['min']}" if "min" in bounds else "",
        f"≤ {bounds['max']}" if "max" in bounds else "",
    ]
    if any(parts):
        return " and ".join(p for p in parts if p)
    return f"pattern {column['pattern']}" if column.get("pattern") else ""


schema = pd.DataFrame(
    [
        {
            "Column": c["name"],
            "Type": c["dtype"],
            "Known at": c["available_at"].replace("_", " "),
            "Nullable": c["nullable"],
            "Rule": _rule(c),
            "Description": c.get("description", ""),
        }
        for c in contract["columns"]
        if c["available_at"] in selected
    ]
)
if query:
    mask = schema["Column"].str.contains(query, case=False) | schema["Description"].str.contains(
        query, case=False
    )
    schema = schema[mask]
st.dataframe(
    schema,
    hide_index=True,
    width="stretch",
    height=380,
    column_config={
        "Nullable": st.column_config.CheckboxColumn(),
        "Description": st.column_config.TextColumn(width="large"),
    },
)
st.caption(
    "Post-purchase columns (delivery dates, reviews) exist in the clean data for analysis but can "
    "never become model features."
)

# Features -------------------------------------------------------------------------------------
layout.section(
    "Features: allowed vs forbidden", "Checked again by the feature gate before training"
)
left, right = st.columns(2)
with left, st.container(border=True, height="stretch"):
    st.markdown(
        f":green[:material/check_circle:] **Required features ({len(contract['required_features'])})**"
    )
    st.caption("Known at checkout, so the model can use them when the customer pays.")
    st.markdown(" ".join(f":green-badge[{f['name']}]" for f in contract["required_features"]))
with right, st.container(border=True, height="stretch"):
    st.markdown(
        f":red[:material/block:] **Forbidden features ({len(contract['forbidden_features'])})**"
    )
    st.caption("Known only after purchase; using them would leak the answer.")
    st.markdown(" ".join(f":red-badge[{name}]" for name in contract["forbidden_features"]))

# Constraints ----------------------------------------------------------------------------------
layout.section("Binding constraints", "Each one is a check in the contract gate")
status_by_check = {c["name"]: c for c in (contract_gate or {}).get("checks", [])}
constraints = pd.DataFrame(
    [
        {
            "ID": c["id"],
            "Rule": c["description"],
            "Kind": c["type"].replace("_", " "),
            "Result": "PASS"
            if status_by_check.get(f"constraint_{c['id']}", {}).get("passed")
            else "FAIL",
        }
        for c in contract["constraints"]
    ]
).set_index("ID")
st.table(constraints)

# Narrative ------------------------------------------------------------------------------------
layout.section("Assumptions and limitations", "Template items plus the Data Analyst crew's notes")
left, right = st.columns(2)
for column, (title, block) in zip(
    (left, right, left, right),
    [
        ("Assumptions", contract.get("assumptions")),
        ("Known limitations", contract.get("known_limitations")),
        ("Business rules", contract.get("business_rules")),
        ("Privacy", {"items": [contract.get("privacy", {}).get("note", "")]}),
    ],
    strict=True,
):
    items = [i for i in (block or {}).get("items", []) if i]
    if items:
        with column.expander(title):
            st.markdown("\n".join(f"- {i}" for i in items))

with st.expander("Sample of the clean data (first 100 rows)"):
    sample = art.load_csv("clean_data.csv", nrows=100)
    if sample is not None:
        st.dataframe(sample, hide_index=True, width="stretch")
layout.footer()
