from pathlib import Path

import lv_datasets
import polars as pl
import pytest

from leibovich_validation import audit


def test_write_outputs_has_no_patient_rows(tmp_path: Path):
    outputs = {"counts": pl.DataFrame({"measure": ["n"], "value": [6]})}
    written = audit.write_outputs(outputs, tmp_path)
    assert [path.name for path in written] == ["counts.csv"]
    assert pl.read_csv(written[0]).rows() == [("n", 6)]


def test_assert_aggregate_rejects_patient_rows():
    with pytest.raises(ValueError, match="identifiers"):
        audit.assert_aggregate(pl.DataFrame({"patient_id": ["T1"]}), "bad")


def test_run_collects_hooks_and_skips_datasets_without_one(tmp_path: Path):
    def harmonise(ctx):
        raise AssertionError

    with_hook = lv_datasets.Dataset(
        "a", "A", {}, harmonise, audit=lambda ctx, horizons: {"a_counts": pl.DataFrame({"n": [1]})}
    )
    without_hook = lv_datasets.Dataset("b", "B", {}, harmonise)
    params_path = tmp_path / "params.yaml"
    params_path.write_text("datasets:\n  a: {}\n  b: {}\n")
    with lv_datasets.override([with_hook, without_hook]):
        written = audit.run(tmp_path, tmp_path / "out", params_path, [3.0])
    assert [path.name for path in written] == ["a_counts.csv"]
