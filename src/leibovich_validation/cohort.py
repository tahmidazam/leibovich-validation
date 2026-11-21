import polars as pl


def reorigin_at_randomisation(patients: pl.DataFrame) -> pl.DataFrame:
    """The patient table on the randomisation clock.

    Follow-up becomes `time - entry_time` and entry becomes zero. The
    surgery-to-randomisation interval is kept as `prediction_offset`. Age stays the age
    at surgery. Cohorts without delayed entry pass through with a zero offset.
    """
    if (patients["entry_time"] >= patients["time"]).any():
        raise ValueError("follow-up must end after entry in every row")
    return patients.with_columns(
        (pl.col("time") - pl.col("entry_time")).alias("time"),
        pl.col("entry_time").alias("prediction_offset"),
        pl.lit(0.0).alias("entry_time"),
    )


def surgery_clock(patients: pl.DataFrame) -> pl.DataFrame:
    """The patient table on the surgery clock, for the surgery-clock sensitivity analysis.

    Follow-up stays timed from surgery with delayed entry at randomisation, so
    predictions are unconditional risks and `prediction_offset` is zero.
    """
    if (patients["entry_time"] >= patients["time"]).any():
        raise ValueError("follow-up must end after entry in every row")
    return patients.with_columns(pl.lit(0.0).alias("prediction_offset"))
