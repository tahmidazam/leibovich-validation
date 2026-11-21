from lv_vasudev import harmonise, mapping

ANALYSIS = {"missing_sex": "exclude", "fallback_surgery_year": 2007, "score_agreement_min": 0.9}


def _row(item: str) -> dict:
    (row,) = [r for r in mapping.mapping_rows({}, ANALYSIS) if r["item"] == item]
    return row


def test_t_stage_rule_comes_from_the_loader_map():
    row = _row("t_stage")
    for raw, mapped in harmonise.T_STAGE.items():
        assert f"{raw} -> {mapped}" in row["rule"]
    # Unsplit TI is absent from the loader's map, so it counts as missing.
    assert "TI" in row["missing"]


def test_settings_come_from_params():
    assert "2007" in _row("operation_year")["rule"]
    assert "90%" in _row("leibovich_score_recorded")["rule"]
