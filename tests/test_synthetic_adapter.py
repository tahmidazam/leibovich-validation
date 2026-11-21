"""The app runs a dataset it has never seen, registered only here through the override."""

import lv_datasets
import polars as pl
from lv_datasets import Context, Dataset
from typer.testing import CliRunner

from leibovich_validation import cli, params
from tests import synthetic

FLOW_STEPS = {
    "patient_rows": ("main", "Patients in the table"),
    "excluded_metastatic": ("excluded", "Metastatic at surgery"),
    "analysis_set": ("main", "Analysis set"),
}

RAW = """\
id,age,sex,stage,node,size_mm,grade,necrosis,years,outcome,score,metastatic
a,60,male,pT1a,N,45,2,N,6.0,censored,0,N
b,55,female,pT3a,N,120,3,Y,2.0,metastasis,7,N
c,70,male,pT2,Y,80,2,N,4.0,competing,5,N
d,65,female,pT1b,N,30,1,N,5.5,censored,2,N
e,50,male,pT4,N,90,4,N,1.0,metastasis,7,Y
"""


def harmonise(ctx: Context) -> tuple[pl.DataFrame, dict[str, int]]:
    raw = pl.read_csv(ctx.data_dir / "patients.csv")
    frame = raw.select(
        pl.col("id").alias("patient_id"),
        pl.col("age").cast(pl.Float64).alias("age_at_surgery"),
        "sex",
        pl.col("stage").alias("t_stage"),
        (pl.col("node") == "Y").alias("node_positive"),
        (pl.col("size_mm") >= 100).alias("size_ge_10cm"),
        "grade",
        (pl.col("necrosis") == "Y").alias("necrosis"),
        pl.lit(None, pl.String).alias("nephrectomy_type"),
        pl.lit(None, pl.String).alias("arm"),
        pl.lit(None, pl.Int64).alias("operation_year"),
        pl.lit(0.0).alias("entry_time"),
        pl.col("years").alias("time"),
        pl.col("outcome").alias("event"),
        pl.lit(False).alias("estimated_operation_date"),
        pl.col("score").alias("leibovich_score_recorded"),
        pl.lit(False).alias("dfs_inconsistent"),
        pl.lit([], pl.List(pl.String)).alias("missing_predictors"),
        pl.when(pl.col("metastatic") == "Y").then(pl.lit("excluded_metastatic")).alias("excluded"),
    )
    return frame, {"patient_rows": raw.height}


TOY = Dataset(name="toy", label="Toy dataset", flow_steps=FLOW_STEPS, harmonise=harmonise)


def test_a_registered_synthetic_dataset_runs_cohort_impute_and_predict(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "toy").mkdir(parents=True)
    (data / "toy" / "patients.csv").write_text(RAW)
    synthetic.write_life_table(data / "lifetables" / "ons-2020-based-qx.xlsx")
    monkeypatch.setenv("LV_DATA_DIR", str(data))
    monkeypatch.setattr(params, "load_dataset", lambda name: {})
    for directory in ("INTERIM_DIR", "COHORT_DIR", "IMPUTE_DIR"):
        monkeypatch.setattr(cli, directory, tmp_path / directory)
    runner = CliRunner()
    with lv_datasets.override([TOY]):
        for command in (["cohort"], ["impute", "--profile", "dev"], ["predict"]):
            result = runner.invoke(cli.app, [*command, "--name", "toy"])
            assert result.exit_code == 0, result.output
    flow = dict(pl.read_csv(tmp_path / "COHORT_DIR" / "toy_flow.csv").iter_rows())
    assert flow == {"patient_rows": 5, "excluded_metastatic": 1, "analysis_set": 4}
    patients = pl.read_parquet(tmp_path / "INTERIM_DIR" / "toy_patients.parquet")
    assert patients["patient_id"].to_list() == ["a", "b", "c", "d"]
    predictions = pl.read_parquet(tmp_path / "INTERIM_DIR" / "toy_predictions.parquet")
    assert set(predictions["patient_id"]) == {"a", "b", "c", "d"}
