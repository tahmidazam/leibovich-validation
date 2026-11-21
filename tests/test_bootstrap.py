import os
import time

import polars as pl
import pytest

from leibovich_validation import bootstrap


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_interrupt_cancels_pending_work_and_stops_the_workers():
    pids: set[int] = set()

    def slow(_resample, _seed):
        time.sleep(1)
        return {"pid": float(os.getpid())}

    def interrupt_after_first_batch(batch):
        pids.update(int(values["pid"]) for values, _ in batch)
        raise KeyboardInterrupt

    frame = pl.DataFrame({"x": range(5)})
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        list(bootstrap._outcomes(slow, frame, list(range(400)), 2, interrupt_after_first_batch))
    assert time.monotonic() - start < 30  # the 400 resamples would take minutes
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.1)
    assert pids
    assert not any(_alive(pid) for pid in pids)
