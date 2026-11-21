from lv_transorce import harmonise, mapping

PARAMS = {"operation_window": ["2007-07-01", "2013-04-30"], "randomisation_lag_days": [28, 92]}
ANALYSIS = {"missing_sex": "exclude", "fallback_surgery_year": 2006, "score_agreement_min": 0.9}


def _row(item: str, **analysis) -> dict:
    rows = mapping.mapping_rows(PARAMS, ANALYSIS | analysis)
    (row,) = [r for r in rows if r["item"] == item]
    return row


def test_t_stage_rule_comes_from_the_loader_map(monkeypatch):
    row = _row("t_stage")
    assert row["source"] == "T stage"
    for raw, mapped in harmonise.T_STAGE.items():
        assert f"{raw} -> {mapped}" in row["rule"]
    # Worked example: the workbook's own spelling reaches the pT3a coefficient.
    assert "pT3a - 4 -> pT3a" in row["rule"]
    assert '"pT3a - 4"' in row["raw_coding"]
    monkeypatch.setitem(harmonise.T_STAGE, "pT3a - 4", "pT3b")
    assert "pT3a - 4 -> pT3b" in _row("t_stage")["rule"]


def test_thresholds_settings_and_mapping_path_come_from_code_and_params():
    size = _row("size_ge_10cm")
    assert "Tumour Size" in size["source"]
    assert f"{harmonise.SIZE_THRESHOLD_MM} mm" in size["rule"]
    assert '"exclude"' in _row("sex")["missing"]
    assert "90%" in _row("leibovich_score_recorded", score_agreement_min=0.9)["rule"]
    rule = _row("event of interest")["rule"]
    assert "nodal, lung, bone or liver flags" in rule
    assert "packages/transorce/src/lv_transorce/other_details_mapping.yaml" in rule


def test_other_details_table_matches_the_approved_mapping():
    classes = {t: e["class"] for t, e in harmonise.load_other_details_mapping().items()}
    table = mapping.mapping_tables()["table_s2_other_details"]
    assert dict(zip(table["other_details"], table["class"], strict=True)) == classes
