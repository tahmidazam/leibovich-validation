import lv_datasets
import polars as pl
import pytest
from lv_datasets import Context, SchemaError

from tests.test_synthetic_adapter import RAW, TOY, harmonise


def _harmonised(tmp_path) -> tuple[pl.DataFrame, dict[str, int]]:
    (tmp_path / "patients.csv").write_text(RAW)
    return harmonise(Context(tmp_path, {}))


def test_a_dataset_with_no_recorded_score_has_an_empty_agreement_table(tmp_path):
    frame, notes = _harmonised(tmp_path)
    frame = frame.with_columns(pl.lit(None, pl.Int64).alias("leibovich_score_recorded"))
    result = lv_datasets.finalise(TOY, frame, notes, 0.95)
    assert result.score_agreement.rows() == [
        ("total_compared", 0),
        ("agree", 0),
        ("disagrees_unexplained", 0),
    ]


def test_a_declared_step_without_a_note_raises_a_schema_error(tmp_path):
    frame, _ = _harmonised(tmp_path)
    with pytest.raises(SchemaError, match="no count"):
        lv_datasets.finalise(TOY, frame, {}, 0.95)


def test_a_flow_that_does_not_add_up_to_the_raw_rows_raises_a_schema_error(tmp_path):
    frame, notes = _harmonised(tmp_path)
    with pytest.raises(SchemaError, match="does not reconcile"):
        lv_datasets.finalise(TOY, frame, {"patient_rows": 6}, 0.95)
    # A row removed under a step that is not an exclusion step leaves the flow short.
    removed_as_note = frame.with_columns(
        excluded=pl.when(pl.col("patient_id") == "a")
        .then(pl.lit("patient_rows"))
        .otherwise(pl.col("excluded"))
    )
    with pytest.raises(SchemaError, match="does not reconcile"):
        lv_datasets.finalise(TOY, removed_as_note, notes, 0.95)
