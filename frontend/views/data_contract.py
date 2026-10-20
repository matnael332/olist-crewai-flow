import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from services.artifacts import ARTIFACTS, load_csv, load_json

st.title("Data and dataset contract")
report = load_json("cleaning_report.json")
contract = load_json("dataset_contract.json")
if not report or not contract:
    st.info("No cleaned data in artifacts/ yet.")
    st.stop()

st.subheader("Cleaning: raw tables to one row per delivered order")
st.caption(
    f"{report['rows_in']:,} raw orders became {report['rows_out']:,} clean delivered orders."
)
st.dataframe(
    pd.DataFrame(
        [
            {"rule": r["rule"], "what it does": r["description"], "rows removed": r["rows_removed"]}
            for r in report["rules"]
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("clean_data.csv (first 200 rows)")
sample = load_csv("clean_data.csv", nrows=200)
if sample is not None:
    st.dataframe(sample, hide_index=True, width="stretch", height=300)
with open(ARTIFACTS / "clean_data.csv", "rb") as fh:
    st.download_button("Download clean_data.csv", fh, file_name="clean_data.csv")

st.subheader(f"Dataset contract v{contract['contract_version']}")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{contract['row_count']:,}")
c2.metric("Columns", len(contract["columns"]))
c3.metric("Binding constraints", len(contract["constraints"]))

rows = []
for col in contract["columns"]:
    allowed = col.get("allowed_values")
    rows.append(
        {
            "column": col["name"],
            "dtype": col["dtype"],
            "available at": col["available_at"],
            "nullable": col["nullable"],
            "bounds": col.get("bounds", ""),
            "allowed values": f"{len(allowed)} values"
            if allowed and len(allowed) > 6
            else allowed or "",
            "observed range": col.get("observed_range", ""),
        }
    )
st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")

gate = (load_json("validation_report.json") or {}).get("gates", {}).get("gate_contract")
left, right = st.columns(2)
with left:
    st.markdown("**Constraints and gate results**")
    results = {c["name"]: c for c in gate["checks"]} if gate else {}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "id": c["id"],
                    "rule": c.get("description", ""),
                    "gate": "PASS"
                    if results.get(f"constraint_{c['id']}", {}).get("passed")
                    else "n/a",
                }
                for c in contract["constraints"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
with right:
    st.markdown("**Features the Data Scientist crew must build**")
    st.write(", ".join(f"`{f['name']}`" for f in contract["required_features"]))
    st.markdown("**Forbidden (post-purchase) columns**")
    st.write(", ".join(f"`{f}`" for f in contract["forbidden_features"]))
    st.markdown("**Target**")
    st.write(contract["target"]["definition"])

for block, title in [
    ("assumptions", "Assumptions"),
    ("business_rules", "Business rules"),
    ("known_limitations", "Known limitations"),
]:
    with st.expander(f"{title} ({contract[block]['narrative_source']})"):
        for item in contract[block]["items"]:
            st.markdown(f"- {item}")
with st.expander("Privacy"):
    st.write(contract["privacy"]["note"])
st.download_button(
    "Download dataset_contract.json",
    (ARTIFACTS / "dataset_contract.json").read_bytes(),
    file_name="dataset_contract.json",
)
