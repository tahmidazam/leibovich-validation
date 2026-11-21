from pathlib import Path

import lv_datasets
import polars as pl

from leibovich_validation import atomic, params

_ID_LIKE = {
    "patient_id",
    "PATIENT_ID",
    "SAMPLE_ID",
    "STUDY_ID",
    "trial_identifier",
    "icaird_ref",
    "image_tag",
    "patient_id_number",
}
_MAX_ROWS = 100_000


def assert_aggregate(frame: pl.DataFrame, name: str) -> None:
    offending = _ID_LIKE & set(frame.columns)
    if offending:
        raise ValueError(
            f"{name}: audit output would carry patient identifiers {sorted(offending)}"
        )
    if frame.height > _MAX_ROWS:
        raise ValueError(f"{name}: {frame.height} rows is not an aggregate table")


def write_outputs(outputs: dict[str, pl.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in outputs.items():
        assert_aggregate(frame, name)
        path = out_dir / f"{name}.csv"
        atomic.save(frame, path)
        written.append(path)
    return written


def run(data_dir: Path, out_dir: Path, params_path: Path, horizons: list[float]) -> list[Path]:
    """Write every registered dataset's audit tables; datasets without the hook add none."""
    outputs: dict[str, pl.DataFrame] = {}
    for name in lv_datasets.names():
        if audit := lv_datasets.get(name).audit:
            ctx = lv_datasets.Context(data_dir / name, params.load_dataset(name, params_path))
            outputs |= audit(ctx, horizons)
    return write_outputs(outputs, out_dir)
