import polars as pl
import pytest

from leibovich_validation import cohort


def test_reorigin_at_randomisation_moves_time_zero_and_keeps_age_at_surgery():
    # Randomised 0.3 years after surgery, event 5.2 years after surgery: from
    # randomisation the event is at 4.9 years. The old entry is the prediction
    # offset, and age stays the age at surgery.
    patients = pl.DataFrame(
        {"age_at_surgery": [54.7, 61.2], "entry_time": [0.3, 0.0], "time": [5.2, 3.0]}
    )
    out = cohort.reorigin_at_randomisation(patients)
    assert out["time"].to_list() == pytest.approx([4.9, 3.0])
    assert out["entry_time"].to_list() == [0.0, 0.0]
    assert out["prediction_offset"].to_list() == [0.3, 0.0]
    assert out["age_at_surgery"].to_list() == [54.7, 61.2]


def test_reorigin_at_randomisation_rejects_exit_before_entry():
    patients = pl.DataFrame({"age_at_surgery": [60.0], "entry_time": [0.3], "time": [0.3]})
    with pytest.raises(ValueError, match="after entry"):
        cohort.reorigin_at_randomisation(patients)
