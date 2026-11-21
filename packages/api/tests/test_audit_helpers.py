from lv_datasets.audit import reverse_km_median


def test_reverse_km_median_worked_example():
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    censored = [True] * 8
    # Survival of censoring: 7/8, 6/7*7/8=0.75, then 0.625, then 0.5 at t=4.
    assert reverse_km_median(times, censored) == 4.0
