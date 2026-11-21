import lv_datasets
import polars as pl
import pytest
import yaml

from leibovich_validation import leibovich, params, tables, tables_latex


def synthetic_model(tmp_path, **overrides) -> params.ModelParams:
    """Model settings loaded from a synthetic params.yaml, like the real loader."""
    model = {
        "life_table": "lifetables/synthetic-qx.xlsx",
        "start_year": 2030,
        "round_other_cause_to_percent": False,
    }
    model.update(overrides)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"model": model}))
    return params.load_model(path)


def section(table: pl.DataFrame, name: str) -> dict[str, str]:
    """The `item` -> `value` lookup for one section of Table S1."""
    rows = table.filter(pl.col("section") == name)
    return dict(zip(rows["item"], rows["value"], strict=True))


def test_coefficients_match_leibovich_table_3(tmp_path):
    table = tables.table_s1(synthetic_model(tmp_path))
    coef = section(table, "Leibovich linear predictor coefficients")
    for stage, value in leibovich.T_COEF.items():
        expected = "0 (reference)" if value == 0 else f"{value:.3f}"
        assert coef[f"T stage {stage}"] == expected
    # Worked example: published Table 3 values for a high-risk tumour.
    assert coef["T stage pT3b"] == "2.014"
    assert coef["Regional lymph node metastasis"] == "0.924"
    assert coef["Nuclear grade 3"] == "0.640"
    assert coef["Tumour necrosis"] == "0.678"
    sources = table.filter(pl.col("section") == "Leibovich linear predictor coefficients")
    assert all("10.1002/cncr.11234" in s and "Table 3" in s for s in sources["source"])


def test_points_match_leibovich_table_4(tmp_path):
    table = tables.table_s1(synthetic_model(tmp_path))
    points = section(table, "Leibovich risk score points")
    for stage, value in lv_datasets.T_POINTS.items():
        assert points[f"T stage {stage}"] == str(value)
    assert points["Regional lymph node metastasis"] == str(lv_datasets.NODE_POSITIVE_POINTS)
    assert points["Tumour size 10 cm or greater"] == str(lv_datasets.SIZE_GE_10CM_POINTS)
    assert points["Nuclear grade 4"] == "3"
    assert points["Tumour necrosis"] == "1"
    sources = table.filter(pl.col("section") == "Leibovich risk score points")
    assert all("Table 4" in s for s in sources["source"])


def test_baseline_rows_equal_the_recovered_values(tmp_path):
    table = tables.table_s1(synthetic_model(tmp_path))
    baseline = section(table, "Baseline cumulative hazard")
    for years, value in leibovich.BASELINE_CUMULATIVE_HAZARD.items():
        item = f"Baseline cumulative hazard at {years} year{'' if years == 1 else 's'}"
        assert baseline[item] == f"{value:.6g}"
    sources = table.filter(pl.col("section") == "Baseline cumulative hazard")["source"]
    assert all("geometric mean" in s and "10.1101/2025.05.22.25328132" in s for s in sources)


def test_life_table_rows_come_from_params(tmp_path):
    table = tables.table_s1(
        synthetic_model(tmp_path, start_year=2031, round_other_cause_to_percent=False)
    )
    life = section(table, "Life table")
    assert life["Mortality rates"] == "lifetables/synthetic-qx.xlsx"
    assert "2031" in life["Calendar years"]
    assert life["Other-cause risk"] == "not rounded"

    rounded = tables.table_s1(synthetic_model(tmp_path, round_other_cause_to_percent=True))
    assert section(rounded, "Life table")["Other-cause risk"] == "rounded to a whole percent"


def test_table_specifies_enough_to_reproduce_preprint_table_1(tmp_path):
    table = tables.table_s1(synthetic_model(tmp_path))
    items = set(table["item"])
    assert {"Leibovich (original)", "Leibovich Plus (published)"} == set(
        section(table, "Model equations")
    )
    assert not any("azard-based" in item for item in items)
    profiles = section(table, "Preprint tumour profiles")
    assert set(profiles) == {f"{name}-risk profile" for name in leibovich.PREPRINT_PROFILES}
    text = "\n".join(" | ".join(str(v) for v in row) for row in table.iter_rows())
    for needle in ("Table 3", "Table 4", "geometric mean", "piecewise constant", "qx"):
        assert needle in text


def test_write_table_emits_a_csv(tmp_path):
    table = tables.table_s1(synthetic_model(tmp_path))
    (csv_path,) = tables.write_table(table, "table_s1", tmp_path / "tables")
    assert csv_path.name == "table_s1.csv"
    reread = pl.read_csv(csv_path)
    assert reread.columns == tables.COLUMNS
    assert reread.rows() == table.rows()


def test_main_table_1_latex_preserves_cells_and_escapes_tex(tmp_path):
    frame = pl.DataFrame(
        {
            "section": ["Sex", "Sex"],
            "characteristic": ["male", "female"],
            tables.MAYO_COLUMN: ["1 (50%)", "1 (50%)"],
            "TranSORCE": ["2 (100%)", "0 (0%)"],
            "Vasudev cohort": ["3 & 4", ""],
        }
    )
    path = tables_latex.write_table_1(frame, tmp_path / "table_1.tex")
    text = path.read_text()
    assert text.count(r"\multicolumn{4}{@{}l}{\textbf{Sex}}") == 1
    assert r"male & 1 (50\%) & 2 (100\%) & 3 \& 4 \\" in text
    assert r"female & 1 (50\%) & 0 (0\%) &  \\" in text


def test_main_table_2_latex_has_labelled_intervals_and_no_secondary_horizon(tmp_path):
    frame = pl.DataFrame(
        {
            "cohort": ["transorce"],
            "model": ["Leibovich Plus"],
            "horizon_years": [5.0],
            "observed_incidence": ["0.271 (0.230\u20130.314)"],
            "oe_ratio": ["0.690 (0.590\u20130.789)"],
            "calibration_slope": ["1.022 (0.759\u20131.328)"],
            "ici": ["0.125 (0.101\u20130.167)"],
            "auc": ["0.743 (0.690\u20130.798)"],
            "brier": ["0.181 (0.164\u20130.198)"],
            "oe_other_cause": ["0.478 (0.200\u20130.798)"],
            "other_cause_status": ["descriptive"],
        }
    )
    path = tables_latex.write_table_2(
        frame, tmp_path / "table_2.tex", 5.0, {"transorce": "TranSORCE"}
    )
    text = path.read_text()
    assert (
        "TranSORCE & Leibovich Plus & Observed incidence & 0.271 (0.230\u20130.314) " + r"\\"
    ) in text
    assert ("Other-cause O/E (descriptive) & 0.478 (0.200\u20130.798)") in text
    assert "oe_ratio" not in text
    with pytest.raises(ValueError, match="primary horizon"):
        tables_latex.write_table_2(
            pl.concat([frame, frame.with_columns(horizon_years=pl.lit(3.0))]),
            tmp_path / "invalid.tex",
            5.0,
            {"transorce": "TranSORCE"},
        )


def _table2_summary() -> pl.DataFrame:
    """The report stage's Table 2 source, one cohort and one model."""
    rows = [
        ("aj_incidence", "observed", "", 0.20, 0.15, 0.26),
        ("oe_ratio", "Leibovich Plus", "pt3a", 0.9, 0.7, 1.1),
        ("calibration_slope", "Leibovich Plus", "pt3a", 1.1, 0.9, 1.3),
        ("ici", "Leibovich Plus", "pt3a", 0.031, 0.02, 0.045),
        ("auc", "Leibovich Plus", "pt3a", 0.7, 0.6, 0.8),
        ("brier", "Leibovich Plus", "pt3a", 0.12, 0.1, 0.15),
    ]
    return pl.DataFrame(
        [
            {
                "label": label,
                "series": series,
                "stratum": "transorce",
                "bound": bound,
                "horizon": 5.0,
                "estimate": estimate,
                "lo": lo,
                "hi": hi,
                "n": 438,
                "events": 60,
                "method": "bootstrap",
            }
            for label, series, bound, estimate, lo, hi in rows
        ]
    )


def test_table_2_forms_cells_with_intervals():
    table = tables.table_2(_table2_summary(), set())
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["cohort"] == "transorce"
    assert row["model"] == "Leibovich Plus"
    assert row["horizon_years"] == 5.0
    assert row["observed_incidence"] == "0.200 (0.150\u20130.260)"
    assert row["oe_ratio"] == "0.900 (0.700\u20131.100)"
    assert row["ici"] == "0.031 (0.020\u20130.045)"
    assert row["auc"].startswith("0.700 (")
    assert row["brier"].endswith("0.150)")


def test_table_2_handles_a_missing_interval():
    summary = _table2_summary().with_columns(
        pl.when(pl.col("label") == "auc").then(pl.lit(None)).otherwise(pl.col("lo")).alias("lo")
    )
    table = tables.table_2(summary, set())
    assert table.row(0, named=True)["auc"] == "0.700"


def synthetic_analysis(tmp_path, **overrides) -> params.AnalysisParams:
    """Analysis settings loaded from a synthetic params.yaml, like the real loader."""
    analysis = {
        "horizon_years": 5,
        "secondary_horizon_years": 3,
        "score_agreement_min": 0.9,
        "seed": 1,
        "bootstrap_resamples": 10,
        "imputations": 2,
        "bootstrap_imputations": 2,
        "missing_sex": "exclude",
        "fallback_surgery_year": 2006,
        "min_at_risk": 50,
        "subgroups": {
            "age_cut_points": [60, 70],
            "leibovich_groups": {"low": [0, 2], "intermediate": [3, 5], "high": [6, 11]},
        },
    }
    analysis.update(overrides)
    path = tmp_path / "params-analysis.yaml"
    path.write_text(yaml.safe_dump({"analysis": analysis}))
    return params.load_analysis(path=path)


def _dataset(name: str, rows: list[dict] | None = None) -> lv_datasets.Dataset:
    def harmonise(ctx):
        raise AssertionError

    def mapping_rows(params, analysis):
        return rows

    return lv_datasets.Dataset(
        name, name.upper(), {}, harmonise, mapping_rows=mapping_rows if rows else None
    )


def _row(section: str, item: str) -> dict:
    blank = dict.fromkeys(("source", "raw_coding", "rule", "missing"), "")
    return {"section": section, "item": item, **blank}


def test_table_s2_orders_rows_by_section_then_item_and_skips_datasets_without_the_hook(tmp_path):
    a = _dataset(
        "a", [_row("Timing", "time"), _row("Predictors", "grade"), _row("Predictors", "t_stage")]
    )
    b = _dataset("b", [_row("Predictors", "t_stage"), _row("Predictors", "grade")])
    c = _dataset("c")
    with lv_datasets.override([a, b, c]):
        table = tables.table_s2(synthetic_analysis(tmp_path), {"a": {}, "b": {}, "c": {}})
    assert list(zip(table["item"], table["cohort"], strict=True)) == [
        ("grade", "A"),
        ("grade", "B"),
        ("t_stage", "A"),
        ("t_stage", "B"),
        ("time", "A"),
    ]


def _sensitivity_rows() -> pl.DataFrame:
    """A miniature results/sensitivity table: one variant, both bounds, both horizons."""
    rows = []
    for horizon in (5.0, 3.0):
        rows.append(
            {
                "variant": "complete_cases",
                "cohort": "transorce",
                "horizon_years": horizon,
                "bound": "",
                "model": "observed",
                "metric": "aj_incidence",
                "estimate": 0.20,
                "lower": 0.15,
                "upper": 0.26,
                "method": "aalen-johansen se",
                "n": 400,
                "events": 90,
            }
        )
        for bound, estimate in (("pt3a", 0.9), ("pt3b", 1.1)):
            rows.append(
                {
                    "variant": "complete_cases",
                    "cohort": "transorce",
                    "horizon_years": horizon,
                    "bound": bound,
                    "model": "Leibovich Plus",
                    "metric": "oe_ratio",
                    "estimate": estimate,
                    "lower": estimate - 0.2,
                    "upper": estimate + 0.2,
                    "method": "bootstrap",
                    "n": 400,
                    "events": 90,
                }
            )
    return pl.DataFrame(rows)


def test_table_s3_keeps_the_primary_horizon_and_labels_variants():
    table = tables.table_s3(_sensitivity_rows(), 5.0)
    assert table.height == 2
    row = table.row(0, named=True)
    assert row["variant"] == "Complete cases"
    assert row["cohort"] == "transorce"
    assert row["bound"] == "pt3a"
    assert row["model"] == "Leibovich Plus"
    assert row["observed_incidence"] == "0.200 (0.150\u20130.260)"
    assert row["oe_ratio"] == "0.900 (0.700\u20131.100)"
    assert row["n"] == 400
    assert row["events"] == 90


def _o_e_only_rows(variant: str, cohort_name: str) -> pl.DataFrame:
    """Observed and O/E rows for one model and bound, as an O/E-only variant writes them."""
    rows = [
        ("aj_incidence", "observed", "", 0.25, 0.20, 0.30),
        ("oe_ratio", "Leibovich Plus", "pt3a", 0.80, 0.60, 1.00),
    ]
    if variant == "country_life_table":
        rows.append(("oe_other_cause", "Leibovich Plus", "pt3a", 1.10, 0.90, 1.30))
    return pl.DataFrame(
        [
            {
                "variant": variant,
                "cohort": cohort_name,
                "horizon_years": 5.0,
                "bound": bound,
                "model": model,
                "metric": metric,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "method": "bootstrap",
                "n": 100,
                "events": 20,
            }
            for metric, model, bound, estimate, lower, upper in rows
        ]
    )


def test_table_s3_marks_measures_an_o_e_only_variant_cannot_estimate():
    rows = pl.concat(
        [
            _sensitivity_rows(),
            _o_e_only_rows("surgery_clock", "transorce"),
            _o_e_only_rows("country_life_table", "vasudev"),
        ]
    )
    table = tables.table_s3(rows, 5.0)
    clock = table.filter(pl.col("variant") == "Surgery clock with entry at randomisation").row(
        0, named=True
    )
    assert clock["oe_ratio"] == "0.800 (0.600\u20131.000)"
    for measure in ("calibration_slope", "auc", "brier"):
        assert clock[measure] == "not estimable (a)"
    who = table.filter(pl.col("variant") == "Country-matched WHO life table").row(0, named=True)
    assert who["auc"] == "not estimable (b)"
    assert who["oe_other_cause"] == "1.100 (0.900\u20131.300)"
    assert clock["oe_other_cause"] == ""
    # A variant that was scored in full has blanks, not the not estimable marker.
    complete = table.filter(pl.col("variant") == "Complete cases").row(0, named=True)
    assert complete["auc"] == ""
    # Rows follow the register's order, not the file order.
    assert table["variant"].to_list() == [
        "Complete cases",
        "Complete cases",
        "Country-matched WHO life table",
        "Surgery clock with entry at randomisation",
    ]

    notes = tables.table_s3_footnotes(table)
    assert [note[:3] for note in notes] == ["(a)", "(b)"]
    assert "delayed entry" in notes[0]
    assert tables.table_s3_footnotes(tables.table_s3(_sensitivity_rows(), 5.0)) == []


def test_write_table_puts_footnotes_in_a_notes_file(tmp_path):
    table = pl.DataFrame({"a": ["x"]})
    paths = tables.write_table(table, "t", tmp_path, ["(a) Reason."])
    assert [path.name for path in paths] == ["t.csv", "t_notes.csv"]
    assert paths[0].read_text() == "a\nx\n"
    assert paths[1].read_text() == "note\n(a) Reason.\n"


def test_table_s3_passes_an_unknown_variant_name_through():
    rows = _sensitivity_rows().with_columns(pl.lit("a_new_variant").alias("variant"))
    table = tables.table_s3(rows, 5.0)
    assert table["variant"].to_list() == ["a_new_variant"] * 2


def _table1_summary() -> pl.DataFrame:
    """A report-stage Table 1 source in miniature, two cohorts."""
    return pl.DataFrame(
        [
            {
                "key": "n_patients",
                "section": "",
                "label": "Patients, n",
                "stratum": stratum,
                "value": value,
                "n": int(value),
            }
            for stratum, value in (("TranSORCE", "439"), ("Vasudev cohort", "766"))
        ]
        + [
            {
                "key": "sex_male",
                "section": "Sex",
                "label": "male",
                "stratum": stratum,
                "value": value,
                "n": 100,
            }
            for stratum, value in (("TranSORCE", "280 (63.8%)"), ("Vasudev cohort", "450 (58.7%)"))
        ]
    )


def test_table_1_pivots_cohorts_beside_the_mayo_column():
    table = tables.table_1(_table1_summary())
    assert table.columns == [
        "section",
        "characteristic",
        tables.MAYO_COLUMN,
        "TranSORCE",
        "Vasudev cohort",
    ]
    patients = table.filter(pl.col("characteristic") == "Patients, n").row(0, named=True)
    assert patients[tables.MAYO_COLUMN] == "1671"
    assert patients["TranSORCE"] == "439"
    assert patients["Vasudev cohort"] == "766"
    male = table.filter(pl.col("characteristic") == "male").row(0, named=True)
    assert male[tables.MAYO_COLUMN] == "1061 (63.5%)"
    assert male["section"] == "Sex"


def test_table_1_rejects_a_summary_key_without_a_mayo_value():
    summary = _table1_summary().vstack(
        pl.DataFrame(
            [
                {
                    "key": "new_key",
                    "section": "Sex",
                    "label": "other",
                    "stratum": "TranSORCE",
                    "value": "1",
                    "n": 1,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="new_key"):
        tables.table_1(summary)


def test_mayo_column_matches_leibovich_2003_table_1():
    # Worked example: the published counts for the 1671-patient development
    # cohort, including the checks that each breakdown sums to the total.
    mayo = tables.MAYO_2003_TABLE_1
    assert mayo["n_patients"] == "1671"
    assert mayo["age_median_range"] == "65 (24\u201389)"
    assert mayo["sex_male"] == "1061 (63.5%)"
    assert mayo["t_stage_pT1a"] == "384 (23.0%)"
    assert mayo["t_stage_pT3b"] == "343 (20.5%)"
    assert mayo["node_positive"] == "66 (4.0%)"
    assert mayo["node_negative"] == "1605 (96.0%)"
    assert mayo["size_ge_10cm"] == "359 (21.5%)"
    assert mayo["grade_2"] == "786 (47.0%)"
    assert mayo["necrosis"] == "439 (26.3%)"
    assert mayo["nephrectomy_radical"] == "1671 (100%)"
    assert mayo["followup_median_range"] == "5.4 (0\u201331)"
    assert mayo["events_interest"] == "479"
    counts = [int(mayo[f"t_stage_{s}"].split(" ")[0]) for s in lv_datasets.T_POINTS]
    assert sum(counts) == 1671
    grades = [int(mayo[f"grade_{g}"].split(" ")[0]) for g in lv_datasets.GRADE_POINTS]
    assert sum(grades) == 1671


def test_table_2_adds_the_other_cause_column_and_marks_low_death_cohort_descriptive():
    summary = pl.concat(
        [
            _table2_summary(),
            pl.DataFrame(
                {
                    "label": ["oe_other_cause"],
                    "series": ["Leibovich Plus"],
                    "stratum": ["transorce"],
                    "bound": ["pt3a"],
                    "horizon": [5.0],
                    "estimate": [1.4],
                    "lo": [0.8],
                    "hi": [2.2],
                    "n": [438],
                    "events": [14],
                    "method": ["bootstrap"],
                }
            ),
        ]
    )
    row = tables.table_2(summary, {"transorce"}).row(0, named=True)
    assert row["oe_other_cause"] == "1.400 (0.800\u20132.200)"
    assert row["other_cause_status"] == "descriptive"
    plain = tables.table_2(_table2_summary(), set()).row(0, named=True)
    assert (plain["oe_other_cause"], plain["other_cause_status"]) == ("", "")


def test_table_s6_keeps_flagged_groups_with_their_counts():
    summary = pl.DataFrame(
        {
            "cohort": ["transorce", "transorce"],
            "horizon_years": [5.0, 5.0],
            "group": ["Overall", "70+"],
            "n": [438, 20],
            "events": [14, 3],
            "observed": [0.03, 0.1],
            "observed_lower": [0.02, 0.0],
            "observed_upper": [0.05, 0.3],
            "mean_predicted": [0.02, 0.2],
            "oe_ratio": [1.5, 0.5],
            "lower": [0.9, 0.1],
            "upper": [2.4, 1.2],
            "small": [False, True],
            "status": ["descriptive", "descriptive"],
        }
    )
    table = tables.table_s6(summary)
    assert table["flag"].to_list() == ["", "few events"]
    assert table["deaths"].to_list() == [14, 3]
    assert table["oe_other_cause"][0] == "1.500 (0.900\u20132.400)"
    assert table["observed_incidence"][1] == "0.100 (0.000\u20130.300)"
