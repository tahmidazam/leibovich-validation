from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points

from lv_datasets.dataset import Dataset

GROUP = "leibovich_validation.datasets"

_override: dict[str, Dataset] | None = None


def names() -> list[str]:
    """Registered dataset names, sorted."""
    if _override is not None:
        return sorted(_override)
    return sorted(entry.name for entry in entry_points(group=GROUP))


def get(name: str) -> Dataset:
    """The registered dataset called `name`, loaded on first use."""
    if name not in names():
        raise ValueError(f"unknown dataset {name!r}; registered: {', '.join(names())}")
    if _override is not None:
        return _override[name]
    return entry_points(group=GROUP)[name].load()


@contextmanager
def override(datasets: Iterable[Dataset]) -> Iterator[None]:
    """Replace the entry-point registry with `datasets`, for tests."""
    global _override
    previous, _override = _override, {dataset.name: dataset for dataset in datasets}
    try:
        yield
    finally:
        _override = previous
