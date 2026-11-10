"""Checksum manifest of the raw data, so pipeline stages can depend on its exact contents."""

import hashlib
import json
from pathlib import Path

from leibovich_validation import progress

_CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build(data_dir: Path) -> dict[str, dict[str, int | str]]:
    """Map each file's path relative to `data_dir` to its size and SHA-256."""
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    entries: dict[str, dict[str, int | str]] = {}
    with progress.bar() as bar:
        task = bar.add_task("Hashing files", total=len(files))
        for p in files:
            entries[p.relative_to(data_dir).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            bar.advance(task)
    return entries


def write(data_dir: Path, out: Path) -> int:
    entries = build(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2) + "\n")
    return len(entries)
