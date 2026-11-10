"""End-to-end smoke test: every stage on a synthetic raw-data folder.

The folder holds a handful of patients per cohort in the real raw formats
plus a synthetic life table, and LP_DATA_DIR points at it. The cohort, impute
and predict stages always run; score, sensitivity, report and table run only
when the R bridge is healthy. Each declared stage output is checked for existence, and the
Table 2 source must carry both cohorts. Everything written is aggregate or
goes to the interim directory; no patient-level row reaches a result file.
"""

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from leibovich_validation import cli, raw_vasudev, rbridge, sensitivity, tables
from tests import synthetic

COHORTS = ("transorce", "vasudev")

VASUDEV_PATIENTS = [
    # censored, alive at 6 years
    synthetic.vasudev_patient("L1"),
    # relapse at 2 years
    synthetic.vasudev_patient(
        "L2",
        RELAPSE="Yes",
        DFS_STATUS="1:Recurred/Progressed",
        DFS_MONTHS=24,
        LEIBOVICH_SCORE=4,
    ),
    # other-cause death at 4 years: the competing event
    synthetic.vasudev_patient("L3", OS_STATUS="1:DECEASED", RCC_RELATED_DEATH="No", RCC_MONTHS=48),
    # RCC death without a recorded relapse: relapse at 2.5 years
    synthetic.vasudev_patient(
        "L4",
        OS_STATUS="1:DECEASED",
        RCC_RELATED_DEATH="Yes",
        RCC_STATUS="1:DECEASED",
        RCC_MONTHS=30,
        LEIBOVICH_SCORE=3,
    ),
    # relapsed but coded disease-free: an event at 1.7 years, flagged
    synthetic.vasudev_patient(
        "L5", RELAPSE="Yes", DFS_STATUS="0:DiseaseFree", DFS_MONTHS=20, LEIBOVICH_SCORE=1
    ),
    # DFS time beyond the survival time: censored, capped at 5 years
    synthetic.vasudev_patient("L6", DFS_MONTHS=90, RCC_MONTHS=60, LEIBOVICH_SCORE=2),
    # excluded: overall stage IV
    synthetic.vasudev_patient("L7", STAGE="IV", LEIBOVICH_SCORE=None),
    # excluded: recorded M1
    synthetic.vasudev_patient("L8", STAGE="III", LEIBOVICH_SCORE=None),
]

# Predictors vary a little so the predicted risks are not all identical.
VASUDEV_SAMPLES = [
    synthetic.vasudev_sample("L1"),
    synthetic.vasudev_sample("L2", STAGE_T="TII", STAGE_T_CATEGORY="TII", FFPE_GRADE=3),
    synthetic.vasudev_sample("L3"),
    synthetic.vasudev_sample("L4", FFPE_GRADE=4),
    synthetic.vasudev_sample("L5", TUMOR_SIZE=120),
    synthetic.vasudev_sample("L6", STAGE_N="NI"),
    synthetic.vasudev_sample("L7"),
    synthetic.vasudev_sample("L8", STAGE_M="MI"),
]

TRANSORCE_PATIENTS = [
    # censored at about 7 years
    synthetic.transorce_patient("001"),
    # metastasis at about 1 year
    synthetic.transorce_patient(
        "002",
        dfs_status=1,
        lung_metastasis="Yes",
        dfs_date=dt.datetime(2009, 1, 15),
        isup_grade=3,
        leibovich_score=4,
    ),
    # metastasis at about 2 years
    synthetic.transorce_patient(
        "003",
        dfs_status=1,
        bone_metastasis="Yes",
        dfs_date=dt.datetime(2010, 1, 15),
        t_stage="pT1a",
        isup_grade=1,
        leibovich_score=0,
    ),
    # death without metastasis at about 3 years: the competing event
    synthetic.transorce_patient(
        "004",
        dfs_date=dt.datetime(2010, 6, 1),
        os_status=1,
        os_date=dt.datetime(2011, 3, 1),
    ),
    # censored
    synthetic.transorce_patient("005", age_at_randomisation=72.0),
]


def _data_dir(root: Path) -> Path:
    """The synthetic raw-data folder, laid out like $LP_DATA_DIR."""
    vasudev_dir = root / "vasudev"
    transorce_dir = root / "transorce"
    vasudev_dir.mkdir(parents=True)
    transorce_dir.mkdir(parents=True)
    synthetic.write_cbioportal(
        vasudev_dir / "data_clinical_patient.txt",
        raw_vasudev.PATIENT_COLUMNS,
        VASUDEV_PATIENTS,
    )
    synthetic.write_cbioportal(
        vasudev_dir / "data_clinical_sample.txt",
        raw_vasudev.SAMPLE_COLUMNS,
        VASUDEV_SAMPLES,
    )
    synthetic.write_transorce_workbook(transorce_dir / "clinical.xlsx", TRANSORCE_PATIENTS)
    synthetic.write_life_table(root / "lifetables" / "ons-2020-based-qx.xlsx")
    # The fixture Vasudev cohort patients are all United Kingdom, so the WHO variant
    # needs only the GBR pair.
    for sex, gho in (("male", "SEX_MLE"), ("female", "SEX_FMLE")):
        synthetic.write_who_nmx(
            root / "lifetables" / "who" / f"nmx-GBR-{sex}.json",
            synthetic.who_nmx_records(0.01, range(2000, 2024), sex=gho),
        )
    return root


def _invoke(runner: CliRunner, *args: str) -> None:
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output


def _run_pipeline(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    """Point the CLI at the synthetic data dir and run cohort, impute and predict.

    The synthetic patients record every predictor, so the impute stage finds
    nothing to draw and never touches R: it writes one identical completed
    dataset, which keeps this path runnable without the R bridge.
    """
    monkeypatch.setenv("LP_DATA_DIR", str(_data_dir(tmp_path / "lpdata")))
    interim = tmp_path / "interim"
    cohort_out = tmp_path / "results" / "cohort"
    score_dir = tmp_path / "results" / "scores"
    impute_out = tmp_path / "results" / "imputation"
    monkeypatch.setattr(cli, "INTERIM_DIR", interim)
    monkeypatch.setattr(cli, "COHORT_DIR", cohort_out)
    monkeypatch.setattr(cli, "SCORE_DIR", score_dir)
    monkeypatch.setattr(cli, "IMPUTE_DIR", impute_out)
    runner = CliRunner()
    for name in COHORTS:
        _invoke(runner, "cohort", "--name", name)
        _invoke(runner, "impute", "--name", name, "--profile", "dev")
        _invoke(runner, "predict", "--name", name)
    return interim, cohort_out, score_dir, impute_out


def test_cohort_impute_and_predict_stages_write_their_declared_outputs(tmp_path, monkeypatch):
    interim, cohort_out, _, impute_out = _run_pipeline(tmp_path, monkeypatch)
    for name in COHORTS:
        assert (interim / f"{name}_patients.parquet").is_file()
        assert (interim / f"{name}_imputed.parquet").is_file()
        assert (interim / f"{name}_predictions.parquet").is_file()
        assert (interim / f"{name}_predictions_cc.parquet").is_file()
        assert (impute_out / f"{name}_missingness.csv").is_file()
        for suffix in ("flow", "score_agreement", "prediction_summary"):
            assert (cohort_out / f"{name}_{suffix}.csv").is_file()
    # The harmonised table keeps the surgery clock; the imputed and predicted
    # tables are on the randomisation clock with the old entry as the offset.
    transorce = pl.read_parquet(interim / "transorce_patients.parquet")
    assert (transorce["entry_time"] > 0).all()
    for stage in ("imputed", "predictions"):
        frame = pl.read_parquet(interim / f"transorce_{stage}.parquet")
        assert (frame["entry_time"] == 0).all()
    reorigined = pl.read_parquet(interim / "transorce_imputed.parquet")
    assert reorigined["prediction_offset"].to_list() == pytest.approx(
        transorce["entry_time"].to_list()
    )
    assert reorigined["time"].to_list() == pytest.approx(
        (transorce["time"] - transorce["entry_time"]).to_list()
    )
    vasudev = pl.read_parquet(interim / "vasudev_patients.parquet")
    assert set(vasudev["cohort"].unique()) == {"vasudev"}
    assert (vasudev["entry_time"] == 0).all()
    # Complete patients give one completed dataset and nothing to impute.
    imputed = pl.read_parquet(interim / "vasudev_imputed.parquet")
    assert set(imputed["imputation"].unique()) == {1}
    assert imputed.height == vasudev.height
    missingness = pl.read_csv(impute_out / "vasudev_missingness.csv")
    assert (missingness["n_imputed"] == 0).all()
    # Unrecorded nodes score node-negative: recorded missing, not imputed.
    recorded = dict(zip(missingness["predictor"], missingness["n_missing"], strict=True))
    assert recorded["node_positive"] == 5
    assert recorded["any_predictor"] == 0
    # The flow counts include the four field anomalies.
    flow = dict(pl.read_csv(cohort_out / "vasudev_flow.csv").iter_rows())
    assert flow["analysis_set"] == 6
    assert flow["relapsed_but_coded_disease_free"] == 1
    assert flow["dfs_time_capped_at_survival_time"] == 1


def test_score_report_and_table_stages_write_their_declared_outputs(tmp_path, monkeypatch):
    try:
        checks = rbridge.doctor()
    except Exception as exc:
        pytest.skip(f"R bridge cannot start: {exc}")
    if any(not check.ok for check in checks):
        pytest.skip("R environment not ready")
    _, _, score_dir, impute_out = _run_pipeline(tmp_path, monkeypatch)
    runner = CliRunner()
    for name in COHORTS:
        _invoke(runner, "score", "--name", name, "--profile", "dev")
        for suffix in ("performance", "contrasts", "calibration"):
            assert (score_dir / f"{name}_{suffix}.csv").is_file()
            assert (score_dir / f"{name}_{suffix}_cc.csv").is_file()

    sensitivity_dir = tmp_path / "results" / "sensitivity"
    monkeypatch.setattr(cli, "SENSITIVITY_DIR", sensitivity_dir)
    for name in COHORTS:
        _invoke(runner, "sensitivity", "--kind", "definitions", "--name", name, "--profile", "dev")
        definitions = sensitivity_dir / f"{name}_definitions.csv"
        assert definitions.is_file()
        frame = pl.read_csv(definitions)
        assert frame.columns == [
            "variant",
            "cohort",
            "horizon_years",
            "bound",
            "model",
            "metric",
            "estimate",
            "lower",
            "upper",
            "method",
            "n",
            "events",
        ]
        # The copied variants reproduce the score-stage rows under their tag.
        assert set(frame["variant"]) >= {"complete_cases", "pt3b_for_unsplit_pt3"}
        own = "transorce_surgery_clock" if name == "transorce" else "exclude_inconsistent_vasudev"
        assert own in set(frame["variant"])
        if name == "transorce":
            clock = frame.filter(pl.col("variant") == own)
            assert set(clock["metric"]) == {"aj_incidence", "oe_ratio"}

    # The WHO life-table analysis writes its file to the same directory, in the
    # same schema, so the table stage reads all three sensitivity files.
    sensitivity_out = sensitivity_dir / "life_tables.csv"
    _invoke(
        runner,
        "sensitivity",
        "--kind",
        "life-tables",
        "--profile",
        "dev",
        "--out",
        str(sensitivity_out),
    )
    life = pl.read_csv(sensitivity_out)
    assert life.columns == sensitivity.SENSITIVITY_COLUMNS
    assert set(life["variant"]) == {"who_country_life_table_vasudev"}
    assert set(life["cohort"]) == {"vasudev"}
    assert set(life["metric"]) == {"aj_incidence", "oe_ratio", "oe_other_cause"}

    summary_dir = tmp_path / "results" / "summary"
    _invoke(runner, "report", "--out-dir", str(summary_dir), "--score-dir", str(score_dir))
    for filename in (
        "model_behaviour.csv",
        "table2.csv",
        "other_cause.csv",
        "calibration_5y.csv",
        "flow.csv",
        "table1.csv",
        "subgroups.csv",
    ):
        assert (summary_dir / filename).is_file()
    table2 = pl.read_csv(summary_dir / "table2.csv")
    assert set(table2["stratum"]) == {"transorce", "vasudev"}
    calibration = pl.read_csv(summary_dir / "calibration_5y.csv")
    assert set(calibration["stratum"]) == {"TranSORCE", "Vasudev cohort"}

    tables_dir = tmp_path / "results" / "tables"
    _invoke(
        runner,
        "table",
        "--out-dir",
        str(tables_dir),
        "--summary-dir",
        str(summary_dir),
        "--imputation-dir",
        str(impute_out),
        "--sensitivity-dir",
        str(sensitivity_dir),
    )
    manuscript = pl.read_csv(tables_dir / "table_2.csv")
    assert set(manuscript["cohort"]) == {"transorce", "vasudev"}
    assert set(manuscript["horizon_years"]) == {5.0}
    # One row per cohort and model: the pT3b bound is a sensitivity analysis.
    assert "bound" not in manuscript.columns
    assert manuscript.height == manuscript.select("cohort", "model").n_unique()
    for name in ("table_1.tex", "table_2.tex"):
        assert (tables_dir / name).is_file()
    # Table S2 is generated from the loaders' mapping definitions, not data.
    s2 = pl.read_csv(tables_dir / "table_s2.csv")
    assert {"t_stage", "event of interest", "competing event"} <= set(s2["item"])
    details = pl.read_csv(tables_dir / "table_s2_other_details.csv")
    assert {"other_details", "class", "flagged", "laterality"} <= set(details.columns)
    characteristics = pl.read_csv(tables_dir / "table_1.csv")
    assert tables.MAYO_COLUMN in characteristics.columns
    assert (tables_dir / "table_s4.csv").is_file()
    # The 3-year companion of Table 2 and Table S3 come from the same stage.
    for name in ("table_2_3y", "table_s3", "table_s6"):
        assert (tables_dir / f"{name}.csv").is_file()
    three_year = pl.read_csv(tables_dir / "table_2_3y.csv")
    assert set(three_year["horizon_years"]) == {3.0}
    s6 = pl.read_csv(tables_dir / "table_s6.csv")
    assert set(s6["group"]) == {"Overall", "<60", "60\u201369", "70+"}
    assert set(s6["cohort"]) == {"transorce", "vasudev"}
    assert set(s6.filter(pl.col("cohort") == "transorce")["status"]) == {"descriptive"}
    assert "oe_other_cause" in manuscript.columns
    s3 = pl.read_csv(tables_dir / "table_s3.csv")
    assert {"variant", "cohort", "bound", "model", "oe_ratio", "auc"} <= set(s3.columns)
    assert "not estimable (a)" in s3["auc"].to_list()
    assert "(a) " in (tables_dir / "table_s3_notes.csv").read_text()
