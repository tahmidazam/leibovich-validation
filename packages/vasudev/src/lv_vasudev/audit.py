import polars as pl
from lv_datasets import Context, score_of
from lv_datasets.audit import followup, missing_counts, score_agreement, value_counts

from lv_vasudev import raw
from lv_vasudev.harmonise import predictors

# Predictor columns whose non-null values a mapping rule must explain. The covered values
# are the raw reader's declared categories, which already carry the values the mappings
# work with (unsplit "TI", the "pT3a - 4" spelling); the reader rejects anything outside.
_DECLARED = raw.PATIENT_CATEGORIES | raw.SAMPLE_CATEGORIES
COVERAGE: dict[str, frozenset[str]] = {
    column: _DECLARED[column]
    for column in (
        "STAGE_T",
        "STAGE_N",
        "STAGE_M",
        "NECROSIS",
        "STAGE",
        "SEX",
        "NEPHRECTOMY_TYPE",
        "RELAPSE",
        "OS_STATUS",
        "RCC_RELATED_DEATH",
        "DFS_STATUS",
        "RCC_STATUS",
    )
}


def run(ctx: Context, horizons: list[float]) -> dict[str, pl.DataFrame]:
    patient = raw.read_patient(ctx.data_dir / "data_clinical_patient.txt")
    sample = raw.read_sample(ctx.data_dir / "data_clinical_sample.txt")
    return tables(patient, sample, horizons)


def tables(
    patient: pl.DataFrame, sample: pl.DataFrame, horizons: list[float]
) -> dict[str, pl.DataFrame]:
    outputs: dict[str, pl.DataFrame] = {}
    outputs["vasudev_categorical_values"] = pl.concat(
        [
            value_counts(patient, list(raw.PATIENT_CATEGORIES), COVERAGE, "patient"),
            value_counts(sample, list(raw.SAMPLE_CATEGORIES), COVERAGE, "sample"),
        ]
    )

    outputs["vasudev_missing"] = pl.concat(
        [
            missing_counts(
                patient,
                [raw.MONTHS_TO_YEARS.get(c, c) for c in raw.PATIENT_COLUMNS],
                "patient",
            ),
            missing_counts(sample, list(raw.SAMPLE_COLUMNS), "sample"),
        ]
    )

    rule_counts = patient.select(
        (pl.col("RELAPSE") == "Yes").sum().alias("relapse_yes"),
        ((pl.col("RELAPSE") == "Yes") & (pl.col("DFS_STATUS") == "1:Recurred/Progressed"))
        .sum()
        .alias("relapse_at_dfs_time"),
        ((pl.col("RCC_RELATED_DEATH") == "Yes") & (pl.col("RELAPSE") == "No"))
        .sum()
        .alias("rcc_death_without_relapse"),
        ((pl.col("RELAPSE") == "Yes") & (pl.col("DFS_STATUS") == "0:DiseaseFree"))
        .sum()
        .alias("relapsed_but_coded_disease_free"),
        (pl.col("DFS_YEARS") > pl.col("RCC_YEARS")).sum().alias("dfs_time_later_than_survival"),
        (pl.col("OS_STATUS") == "1:DECEASED").sum().alias("deceased"),
        (pl.col("RCC_STATUS") == "1:DECEASED").sum().alias("rcc_deceased"),
        ((pl.col("OS_STATUS") == "1:DECEASED") & (pl.col("RCC_STATUS") == "0:LIVING"))
        .sum()
        .alias("other_cause_deceased"),
        (pl.col("STAGE") == "IV").sum().alias("overall_stage_iv"),
    )
    outputs["vasudev_event_rules"] = rule_counts.unpivot(variable_name="rule", value_name="n")

    unsplit = sample.group_by("STAGE_T").agg(pl.len().alias("n")).sort("STAGE_T")
    outputs["vasudev_unsplit_stage"] = pl.DataFrame(
        [
            {
                "stage_t": value,
                "n": n,
                "interpretation": {
                    "TI": "unsplit pT1, treated as missing",
                    "TIII": "unsplit pT3, pT3a coefficient in main analysis",
                }[str(value)],
            }
            for value, n in unsplit.iter_rows()
            if str(value) in {"TI", "TIII"}
        ],
        schema={"stage_t": pl.String, "n": pl.UInt32, "interpretation": pl.String},
    )

    joined = patient.join(sample, on="PATIENT_ID", how="left")
    computed = [
        score_of(predictors(*row))
        for row in joined.select(
            "STAGE_T", "STAGE_N", "TUMOR_SIZE", "FFPE_GRADE", "NECROSIS"
        ).iter_rows()
    ]
    outputs["vasudev_score_agreement"] = score_agreement(
        computed, joined["LEIBOVICH_SCORE"].to_list()
    )

    usable = patient.filter(pl.col("DFS_YEARS").is_not_null())
    outputs["vasudev_followup"] = followup(
        usable["DFS_YEARS"].to_list(),
        (usable["DFS_STATUS"] == "0:DiseaseFree").fill_null(False).to_list(),
        horizons,
        "vasudev",
    )

    outputs["vasudev_counts"] = pl.DataFrame(
        [
            {"measure": "patients", "value": patient.height},
            {"measure": "samples", "value": sample.height},
            {"measure": "unique_patient_ids", "value": patient["PATIENT_ID"].n_unique()},
            {"measure": "unique_sample_patient_ids", "value": sample["PATIENT_ID"].n_unique()},
            {"measure": "missing_sex", "value": patient["SEX"].null_count()},
        ]
    )
    return outputs
