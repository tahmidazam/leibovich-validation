import dataclasses
import math

import polars as pl
import pytest

from leibovich_validation import leibovich, leibovich_plus, lifetables, params
from leibovich_validation.config import get_settings
from leibovich_validation.leibovich import PREPRINT_PROFILES, Tumour
from tests.synthetic import constant_rates

# Stimpson et al. Table 1, men: Leibovich Plus metastasis and other-cause death risks (percent)
# at 5 and 10 years. The 85-year-old low-risk row is left out because its other-cause columns
# repeat the metastasis columns, which looks like a copying error in the preprint.
TABLE_1 = {
    ("low", 55): ((2.94, 4.22), (2.96, 6.81)),
    ("low", 65): ((2.92, 4.19), (6.85, 15.45)),
    ("low", 75): ((2.91, 4.17), (16.57, 38.44)),
    ("intermediate", 55): ((25.09, 33.32), (2.92, 6.59)),
    ("intermediate", 65): ((24.37, 31.49), (6.61, 14.24)),
    ("intermediate", 75): ((23.15, 28.74), (15.26, 32.49)),
    ("intermediate", 85): ((21.53, 26.70), (38.40, 60.37)),
    ("high", 55): ((77.60, 84.59), (2.91, 6.55)),
    ("high", 65): ((74.77, 78.17), (6.55, 13.82)),
    ("high", 75): ((68.71, 65.37), (14.62, 28.91)),
    ("high", 85): ((56.58, 52.06), (32.57, 46.04)),
}


def test_score_matches_leibovich_table_4():
    worst = Tumour("pT4", node_positive=True, size_ge_10cm=True, grade=4, necrosis=True)
    assert leibovich.score(worst) == 11
    # The preprint labels its intermediate profile (pT2, grade 3) as score 3, but Table 4 gives
    # 3 points for pT2 plus 1 for grade 3.
    assert [leibovich.score(t) for t in PREPRINT_PROFILES.values()] == [0, 4, 8]


def test_linear_predictor_sums_table_3_coefficients():
    assert leibovich.linear_predictor(PREPRINT_PROFILES["high"]) == pytest.approx(
        2.014 + 0.924 + 0.393 + 0.640
    )


def test_recovered_baseline_reproduces_published_leibovich_risks():
    for name, tumour in PREPRINT_PROFILES.items():
        for years, published in leibovich.PREPRINT_LEIBOVICH_RISK[name].items():
            assert 100 * leibovich.risk(tumour, years) == pytest.approx(published, abs=0.15)


def test_risk_at_the_baseline_knots_is_unchanged_by_interpolation():
    for tumour in PREPRINT_PROFILES.values():
        multiplier = math.exp(leibovich.linear_predictor(tumour))
        for years in (1, 5, 10):
            expected = 1 - math.exp(-leibovich.BASELINE_CUMULATIVE_HAZARD[years] * multiplier)
            assert leibovich.risk(tumour, years) == pytest.approx(expected)


def test_risk_interpolates_between_baseline_knots():
    # The low profile has linear predictor 0, so its risk is 1 - exp(-H(t)). Year 3 sits
    # halfway between the year-1 and year-5 cumulative hazards.
    halfway = (
        leibovich.BASELINE_CUMULATIVE_HAZARD[1] + leibovich.BASELINE_CUMULATIVE_HAZARD[5]
    ) / 2
    assert leibovich.risk(PREPRINT_PROFILES["low"], 3) == pytest.approx(1 - math.exp(-halfway))


def test_risk_rejects_times_outside_the_domain():
    with pytest.raises(ValueError, match="domain"):
        leibovich.risk(PREPRINT_PROFILES["low"], 12)


def test_combine_worked_example():
    # High-risk man aged 55 at 10 years: R_D = 0.9047 and an other-cause risk of 7%.
    metastasis, other = leibovich_plus.combine(0.9047, 0.07)
    overall = 1 - 0.0953 * 0.93
    assert metastasis == pytest.approx(overall * 0.9047 / 0.9747)
    assert 100 * metastasis == pytest.approx(84.59, abs=0.01)
    assert 100 * other == pytest.approx(6.55, abs=0.01)


def test_combine_can_fall_with_longer_follow_up():
    # High-risk man aged 75: published Leibovich Plus risk is 68.71% at 5 years, 65.37% at 10.
    five = leibovich_plus.combine(0.7991, 0.17)[0]
    ten = leibovich_plus.combine(0.9047, 0.40)[0]
    assert ten < five
    assert math.isclose(100 * ten, 65.37, abs_tol=0.01)


def _life_table():
    try:
        data_dir = get_settings().data_dir
    except Exception:
        pytest.skip("LP_DATA_DIR is not set")
    p = params.load_model()
    path = data_dir / p.life_table
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    return lifetables.load_period_qx(path, "male"), p


@pytest.mark.parametrize(("profile", "age"), list(TABLE_1))
def test_reproduces_preprint_table_1(profile, age):
    rates, p = _life_table()
    tumour = PREPRINT_PROFILES[profile]
    for i, years in enumerate((5, 10)):
        r_p = lifetables.other_cause_risk(
            rates, age, years, p.start_year, p.round_other_cause_to_percent
        )
        metastasis, other = leibovich_plus.combine(leibovich.risk(tumour, years), r_p)
        assert 100 * metastasis == pytest.approx(TABLE_1[(profile, age)][0][i], abs=0.02)
        assert 100 * other == pytest.approx(TABLE_1[(profile, age)][1][i], abs=0.02)


def test_unknown_t_stage_and_grade_are_rejected():
    bad_stage = dataclasses.replace(PREPRINT_PROFILES["low"], t_stage="pT5")
    with pytest.raises(ValueError, match="T stage"):
        leibovich.linear_predictor(bad_stage)
    bad_grade = dataclasses.replace(PREPRINT_PROFILES["low"], grade=9)
    with pytest.raises(ValueError, match="grade"):
        leibovich.linear_predictor(bad_grade)


def test_published_risks_at_integer_times_match_the_deployed_model():
    rates = constant_rates(0.07)
    tumour = PREPRINT_PROFILES["high"]
    for years in (1, 5, 10):
        metastasis, other = leibovich_plus.published_risks(rates, tumour, 70, years, 2024, False)
        expected = leibovich_plus.combine(
            leibovich.risk(tumour, years),
            lifetables.other_cause_risk(rates, 70, years, 2024, False),
        )
        assert metastasis == pytest.approx(expected[0])
        assert other == pytest.approx(expected[1])


def test_published_risks_reject_times_outside_the_domain_and_bad_entry():
    rates = constant_rates(0.02)
    tumour = PREPRINT_PROFILES["low"]
    with pytest.raises(ValueError, match="domain"):
        leibovich_plus.published_risks(rates, tumour, 60, 11, 2024, True)
    with pytest.raises(ValueError, match="entry"):
        leibovich_plus.published_risks(rates, tumour, 60, 2, 2024, True, entry=3)


def test_published_risks_reject_ages_and_years_outside_the_life_table():
    rates = constant_rates(0.02, max_age=100, first_year=2024, last_year=2030)
    tumour = PREPRINT_PROFILES["low"]
    with pytest.raises(ValueError, match="projection"):
        leibovich_plus.published_risks(rates, tumour, 60, 10, 2024, True)
    with pytest.raises(ValueError, match="below the life table"):
        leibovich_plus.published_risks(rates.filter(pl.col("age") >= 40), tumour, 30, 5, 2024, True)


def test_ninety_year_old_followed_ten_years_stays_on_the_table():
    # The table's top age is an open band: a person older than it takes its qx.
    rates = constant_rates(0.05, max_age=95)
    metastasis, other = leibovich_plus.published_risks(
        rates, PREPRINT_PROFILES["high"], 90, 10, 2024, True
    )
    assert 0 < metastasis < 1
    assert 0 < other < 1


def test_published_risks_with_entry_condition_each_cumulative_risk():
    # Constant qx = 0.1 and no rounding: other-cause survival over (1, 5] is 0.9 ** 4,
    # and the disease survival is S(5) / S(1) for the Leibovich model.
    rates = constant_rates(0.1)
    tumour = PREPRINT_PROFILES["high"]
    disease = 1 - (1 - leibovich.risk(tumour, 5)) / (1 - leibovich.risk(tumour, 1))
    other = 1 - 0.9**4
    metastasis, other_out = leibovich_plus.published_risks(
        rates, tumour, 60, 5, 2024, False, entry=1
    )
    expected = leibovich_plus.combine(disease, other)
    assert metastasis == pytest.approx(expected[0])
    assert other_out == pytest.approx(expected[1])
