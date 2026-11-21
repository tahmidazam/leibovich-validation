from collections.abc import Callable
from pathlib import Path

import polars as pl


def write(path: Path, writer: Callable[[Path], object]) -> Path:
    """Have `writer` fill a temporary file, then rename it to `path`.

    An interrupt leaves `path` as it was and no temporary file behind.
    """
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save(frame: pl.DataFrame, path: Path) -> Path:
    """Write `frame` to `path` as Parquet or CSV by suffix, creating folders, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return write(path, frame.write_parquet if path.suffix == ".parquet" else frame.write_csv)
