import math

import openpyxl
import pytest

from leibovich_validation import lifetables
from tests import synthetic

# A tiny qx table in the ONS layout: a title row, then a header of "Age" plus one column per
# calendar year. Values are deaths per 100,000, so 50000 is qx = 0.5.
QX_PER_100K = {
    80: {2024: 50000, 2025: 40000, 2026: 30000},
    81: {2024: 60000, 2025: 65400, 2026: 50000},
    82: {2024: 70000, 2025: 80000, 2026: 90000},
}


@pytest.fixture()
def rates(tmp_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "males period qx"
    sheet.append(["Period mortality rates"])
    sheet.append(["Age", "Year 2024", "Year 2025", "Year 2026"])
    for age, by_year in QX_PER_100K.items():
        sheet.append([age, *by_year.values()])
    path = tmp_path / "qx.xlsx"
    workbook.save(path)
    return lifetables.load_period_qx(path, "male")


def test_load_period_qx_returns_rates_by_age_year_and_sex(rates):
    assert rates.columns == ["age", "year", "qx"]
    assert rates.height == 9
    lookup = {(row["age"], row["year"]): row["qx"] for row in rates.iter_rows(named=True)}
    assert lookup[80, 2024] == pytest.approx(0.5)
    assert lookup[81, 2025] == pytest.approx(0.654)
    assert lookup[82, 2026] == pytest.approx(0.9)


def test_other_cause_risk_follows_one_person_through_calendar_years(rates):
    # A man aged 80 in 2024 faces qx(80, 2024) = 0.5 then qx(81, 2025) = 0.654.
    risk = lifetables.other_cause_risk(rates, 80, 2, 2024, round_to_percent=False)
    assert risk == pytest.approx(0.827)


def test_other_cause_risk_rounds_to_a_whole_percent(rates):
    risk = lifetables.other_cause_risk(rates, 80, 2, 2024, round_to_percent=True)
    assert risk == pytest.approx(0.83)


def test_other_cause_risk_needs_every_calendar_year(rates):
    with pytest.raises(ValueError, match="projection"):
        lifetables.other_cause_risk(rates, 80, 2, 2030, round_to_percent=False)


def test_other_cause_risk_rejects_ages_below_the_table(rates):
    with pytest.raises(ValueError, match="below the life table"):
        lifetables.other_cause_risk(rates, 79, 1, 2024, round_to_percent=False)


def test_other_cause_risk_at_a_fractional_year(rates):
    # Constant hazard within the year: the partial third year leaves (1 - 0.9) ** 0.5.
    risk = lifetables.other_cause_risk(rates, 80, 2.5, 2024, round_to_percent=False)
    survival = 0.5 * (1 - 0.654) * 0.1**0.5
    assert risk == pytest.approx(1 - survival)


def test_other_cause_risk_clips_ages_above_the_top_of_the_table(rates):
    # An 81-year-old followed three years is 83 in the last year, above the table's top
    # age of 82, so the open top band's qx applies.
    risk = lifetables.other_cause_risk(rates, 81, 3, 2024, round_to_percent=False)
    assert risk == pytest.approx(1 - (1 - 0.6) * (1 - 0.8) * (1 - 0.9))


# A tiny WHO GHO response: three bands at a different nMx per band, two
# calendar years at a different nMx per year.
WHO_RECORDS = [
    {"Dim1": "SEX_MLE", "Dim2": band, "TimeDim": year, "NumericValue": mx}
    for band, mx in (
        ("AGEGROUP_YEARS60-64", 0.02),
        ("AGEGROUP_YEARS65-69", 0.03),
        ("AGEGROUP_YEARS85PLUS", 0.20),
    )
    for year, mx in ((2005, mx), (2006, mx * 2))
]


@pytest.fixture()
def who_rates(tmp_path):
    path = tmp_path / "nmx-TST-male.json"
    synthetic.write_who_nmx(path, WHO_RECORDS)
    return lifetables.load_who_qx(path)


def test_load_who_qx_reads_the_infant_band_as_age_zero(tmp_path):
    # The GHO codes the infant band "00-01" next to "01-04"; read as ages 0 to 1
    # it would give age 1 two rates.
    path = tmp_path / "nmx-TST-male.json"
    records = [
        {"Dim1": "SEX_MLE", "Dim2": band, "TimeDim": 2005, "NumericValue": mx}
        for band, mx in (("AGEGROUP_YEARS00-01", 0.004), ("AGEGROUP_YEARS01-04", 0.0002))
    ]
    synthetic.write_who_nmx(path, records)
    rates = lifetables.load_who_qx(path)
    lookup = {row["age"]: row["qx"] for row in rates.iter_rows(named=True)}
    assert sorted(lookup) == [0, 1, 2, 3, 4]
    assert lookup[0] == pytest.approx(1 - math.exp(-0.004))
    assert lookup[1] == pytest.approx(1 - math.exp(-0.0002))


def test_load_who_qx_expands_bands_to_single_years(who_rates):
    assert who_rates.columns == ["age", "year", "qx"]
    # 5 + 5 + 26 single ages across two calendar years.
    assert who_rates.height == (5 + 5 + 26) * 2
    lookup = {(row["age"], row["year"]): row["qx"] for row in who_rates.iter_rows(named=True)}
    # Constant hazard inside the band: every single year takes 1 - exp(-nMx),
    # and the 2006 records carry the doubled rate.
    assert lookup[60, 2005] == pytest.approx(1 - math.exp(-0.02))
    assert lookup[64, 2005] == pytest.approx(1 - math.exp(-0.02))
    assert lookup[64, 2006] == pytest.approx(1 - math.exp(-0.04))
    assert lookup[65, 2005] == pytest.approx(1 - math.exp(-0.03))
    # The open 85+ band's finite rate applies at every age to the table's top.
    assert lookup[85, 2005] == pytest.approx(1 - math.exp(-0.2))
    assert lookup[110, 2005] == pytest.approx(1 - math.exp(-0.2))


def test_who_qx_feeds_other_cause_risk_across_bands_and_years(who_rates):
    # Worked example: a 64-year-old in 2005 faces the 60-64 rate in 2005 and
    # the doubled 65-69 rate in 2006.
    risk = lifetables.other_cause_risk(who_rates, 64, 2, 2005, round_to_percent=False)
    expected = 1 - (1 - (1 - math.exp(-0.02))) * (1 - (1 - math.exp(-0.06)))
    assert risk == pytest.approx(expected)


def test_load_who_qx_rejects_a_paginated_response(tmp_path):
    path = tmp_path / "nmx-TST-male.json"
    synthetic.write_who_nmx(path, synthetic.who_nmx_records(0.01, range(2000, 2002)))
    path.write_text(
        path.read_text().replace(
            '"value"', '"@odata.nextLink": "https://ghoapi.example/next", "value"'
        )
    )
    with pytest.raises(ValueError, match="paginated"):
        lifetables.load_who_qx(path)


def test_load_who_qx_rejects_mixed_sexes(tmp_path):
    path = tmp_path / "nmx-TST-mixed.json"
    synthetic.write_who_nmx(
        path,
        [
            *synthetic.who_nmx_records(0.01, range(2000, 2001), sex="SEX_MLE"),
            *synthetic.who_nmx_records(0.01, range(2000, 2001), sex="SEX_FMLE"),
        ],
    )
    with pytest.raises(ValueError, match="one sex"):
        lifetables.load_who_qx(path)


def test_load_who_qx_rejects_an_unknown_band(tmp_path):
    path = tmp_path / "nmx-TST-male.json"
    synthetic.write_who_nmx(
        path,
        [
            {
                "Dim1": "SEX_MLE",
                "Dim2": "AGEGROUP_DECADES60-69",
                "TimeDim": 2005,
                "NumericValue": 0.02,
            }
        ],
    )
    with pytest.raises(ValueError, match="age-band"):
        lifetables.load_who_qx(path)


def test_load_who_rates_by_sex_reads_the_country_pair(tmp_path):
    for sex, gho in (("male", "SEX_MLE"), ("female", "SEX_FMLE")):
        synthetic.write_who_nmx(
            tmp_path / "lifetables" / "who" / f"nmx-CZE-{sex}.json",
            synthetic.who_nmx_records(0.01, range(2000, 2024), sex=gho),
        )
    rates = lifetables.load_who_rates_by_sex(tmp_path, "Czech Republic")
    assert set(rates) == {"male", "female"}
    assert rates["male"]["qx"][0] == pytest.approx(1 - math.exp(-0.01))


def test_load_who_rates_by_sex_names_the_missing_input(tmp_path):
    with pytest.raises(ValueError, match="ISO3"):
        lifetables.load_who_rates_by_sex(tmp_path, "Atlantis")
    with pytest.raises(ValueError, match=r"download.sh who"):
        lifetables.load_who_rates_by_sex(tmp_path, "Serbia")
