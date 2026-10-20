from olist_core.io import read_json
from olist_core.raw import load_raw_tables


def test_two_layer_summary(analyst_out, fixture_raw_dir):
    summary = read_json(analyst_out / "eda_summary.json")
    contract = read_json(analyst_out / "dataset_contract.json")
    raw_orders = load_raw_tables(fixture_raw_dir, ["orders"])["orders"]
    assert summary["all_orders"]["n_orders"] == len(raw_orders)
    assert summary["delivered_population"]["n_orders"] == contract["row_count"]
    splits = summary["delivered_population"]["prevalence_by_split"]
    assert set(splits) == {"train", "test", "censored_buffer"}
    assert summary["cleaning"]["rows_out"] == contract["row_count"]


def test_figures_and_html_are_self_contained(analyst_out):
    summary = read_json(analyst_out / "eda_summary.json")
    assert len(summary["figures"]) == 7
    for name in summary["figures"]:
        assert (analyst_out / "figures" / name).stat().st_size > 5_000
    html = (analyst_out / "eda_report.html").read_text(encoding="utf-8")
    assert html.count('src="data:image/png;base64,') == 7
    assert 'src="http' not in html and "<script" not in html


def test_text_artifacts_have_no_numpy_reprs_or_crlf(analyst_out):
    for name in (
        "eda_summary.json",
        "dataset_contract.json",
        "insights.md",
        "cleaning_report.md",
        "cleaning_report.json",
        "raw_profile.json",
        "eda_report.html",
    ):
        raw = (analyst_out / name).read_bytes()
        assert b"\r" not in raw, name
        assert b"np.float64" not in raw and b"np.int64" not in raw, name


def test_insights_cover_required_sections(analyst_out):
    text = (analyst_out / "insights.md").read_text(encoding="utf-8")
    for heading in (
        "## Key numbers (computed)",
        "## Headline",
        "## Key findings",
        "## Recommendations",
        "## Data caveats",
        "## Implications for modeling",
    ):
        assert heading in text
    assert "narrative_source: template" in text
