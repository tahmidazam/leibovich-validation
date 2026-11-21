# Writing an adapter

An adapter is a uv workspace package under `packages/` that describes one dataset. See `GLOSSARY.md` for the terms and `docs/adr/0001-dataset-adapters-as-entry-point-packages.md` for the rules.

## Register

Depend on `lv-datasets` only, and register the `Dataset` in the entry-point group:

```toml
[project]
name = "lv-toy"
dependencies = ["lv-datasets", "polars>=2.1.0"]

[project.entry-points."leibovich_validation.datasets"]
toy = "lv_toy:dataset"

[tool.uv.sources]
lv-datasets = { workspace = true }
```

Copy the ruff block of `packages/vasudev/pyproject.toml`, which bans importing the app, and add the adapter's package to the app's `banned-api` list in the root `pyproject.toml`. Add a `datasets.toy` section to `params.yaml` for anything the adapter reads from its `Context` (an empty `{}` is fine), and put the raw files under `$LV_DATA_DIR/toy/`.

## Describe the dataset

`harmonise` reads the raw files and returns every raw patient in the harmonised columns (`lv_datasets.HARMONISED_COLUMNS`, minus `cohort` and `leibovich_score_computed`, which `finalise` derives) plus an `excluded` column. It also returns the counts of any notes. Declare each exclusion step, note and `analysis_set` in `flow_steps`.

This is the whole adapter the end-to-end test registers (`tests/test_synthetic_adapter.py`), reading a `patients.csv`:

```python
FLOW_STEPS = {
    "patient_rows": ("main", "Patients in the table"),
    "excluded_metastatic": ("excluded", "Metastatic at surgery"),
    "analysis_set": ("main", "Analysis set"),
}


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


dataset = Dataset(name="toy", label="Toy dataset", flow_steps=FLOW_STEPS, harmonise=harmonise)
```

Rules the schema enforces (`validate_patient_table`): unique `patient_id`, `event` in `lv_datasets.EVENT_TYPES`, `sex` in `SEXES`, `t_stage` in `T_STAGES`, `grade` 1 to 4, and `0 <= entry_time < time`. Leave a predictor null where it is not recorded; the app imputes it. Add other columns you want carried through (for example `country`) after the harmonised ones.

## Optional hooks

`audit(ctx, horizons)` returns aggregate audit tables by output name. `mapping_rows(params, analysis)` returns the dataset's rows for Table S2, and `mapping_tables()` further mapping tables. Without them the dataset has no audit tables and no S2 section.

## Check it

After `uv sync --all-packages`, `uv run leibovich-validation datasets` lists the dataset and `uv run leibovich-validation cohort --name toy` builds its patient table.
