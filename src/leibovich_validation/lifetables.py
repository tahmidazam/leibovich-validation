import json
import math
import re
from pathlib import Path
from typing import Literal

import openpyxl
import polars as pl

Sex = Literal["male", "female"]

# Country names (the cBioPortal COUNTRY values) to the ISO3 codes
# the WHO Global Health Observatory API is keyed by.
COUNTRY_TO_ISO3 = {
    "Czech Republic": "CZE",
    "Romania": "ROU",
    "Russia": "RUS",
    "Serbia": "SRB",
    "United Kingdom": "GBR",
}

# The GHO OData indicator for the life-table death rate: "nMx - age-specific
# death rate between ages x and x+n" by country, sex and calendar year
# (https://www.who.int/data/gho/data/indicators/indicator-details/GHO/gho-ghe-life-tables-by-country).
WHO_NMX_INDICATOR = "LIFE_0000000029"

# The WHO tables are abridged: 0, 1-4, then 5-year bands to 80-84 and an open
# 85+ band. The open band's rate is written out to this age so the expanded
# table has a top edge like the ONS one.
WHO_TOP_AGE = 110

_AGE_BAND = re.compile(r"AGEGROUP_YEARS(\d+)-(\d+)")
_AGE_BAND_OPEN = re.compile(r"AGEGROUP_YEARS(\d+)PLUS")
_AGE_BAND_INFANT = "AGEGROUP_YEARS00-01"


def load_period_qx(path: Path, sex: Sex) -> pl.DataFrame:
    """Probability of dying before the next birthday, by age and calendar year.

    Returns columns age, year and qx. The ONS sheets give deaths per 100,000; qx is a
    probability.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook[f"{sex}s period qx"].iter_rows(values_only=True)
    header = next(row for row in rows if str(row[0]).lower().startswith(("age", "exact age")))
    years = [int(str(cell).removeprefix("Year ")) for cell in header[1:]]
    records = [
        (int(row[0]), year, float(value) / 100_000)
        for row in rows
        if isinstance(row[0], int | float)
        for year, value in zip(years, row[1:], strict=False)
        if value is not None
    ]
    return pl.DataFrame(records, schema=["age", "year", "qx"], orient="row")


def qx_path(rates: pl.DataFrame, age: int, years: int, start_year: int) -> list[float]:
    """One mortality probability per follow-up year, following one person through calendar time.

    Ages above the table's top age take the top age's rate, since the ONS table ends in
    an open age band. Ages below the youngest age and years outside the projection raise
    a ValueError naming the range covered.
    """
    if years < 0:
        msg = f"follow-up years cannot be negative, got {years}"
        raise ValueError(msg)
    if years == 0:
        return []
    ages, years_in_table = rates["age"].to_list(), rates["year"].to_list()
    min_age, max_age = min(ages), max(ages)
    if age < min_age:
        msg = f"age {age} is below the life table, which starts at age {min_age}"
        raise ValueError(msg)
    first_year, last_year = min(years_in_table), max(years_in_table)
    needed_last = start_year + years - 1
    if start_year < first_year or needed_last > last_year:
        msg = (
            f"calendar years {start_year} to {needed_last} are outside the life-table "
            f"projection, which covers {first_year} to {last_year}"
        )
        raise ValueError(msg)
    follow_up = pl.DataFrame(
        {
            "age": [min(age + k, max_age) for k in range(years)],
            "year": [start_year + k for k in range(years)],
        }
    )
    joined = follow_up.join(rates, on=["age", "year"], how="left")
    missing = joined.filter(pl.col("qx").is_null())
    if missing.height:
        msg = f"no qx for (age, year): {missing.rows()}"
        raise ValueError(msg)
    return joined["qx"].to_list()


def other_cause_risk(
    rates: pl.DataFrame, age: int, years: float, start_year: int, round_to_percent: bool
) -> float:
    """Probability of death by `years`, following one person through the calendar years.

    Stimpson et al. appear to have rounded this risk to a whole percent; their Table 1
    reproduces only with rounding on. At a fractional time the final year's hazard is
    constant, so the fraction applies as the survival factor (1 - qx) ** fraction.
    """
    if years == 0:
        return 0.0
    whole = math.floor(years)
    fraction = years - whole
    qx = qx_path(rates, age, whole + (1 if fraction else 0), start_year)
    factors = [1 - q for q in qx[:whole]]
    if fraction:
        factors.append((1 - qx[whole]) ** fraction)
    risk = 1 - math.prod(factors)
    return round(risk * 100) / 100 if round_to_percent else risk


def _who_age_band(dim: str) -> tuple[int, int]:
    """The first and last single ages of a GHO age-band code.

    Closed bands look like "AGEGROUP_YEARS60-64"; the open band is
    "AGEGROUP_YEARS85PLUS" and runs to WHO_TOP_AGE. The infant band is coded
    "AGEGROUP_YEARS00-01" but covers age 0 alone, since "AGEGROUP_YEARS01-04" starts at 1.
    """
    if dim == _AGE_BAND_INFANT:
        return 0, 0
    if match := _AGE_BAND_OPEN.fullmatch(dim):
        return int(match.group(1)), WHO_TOP_AGE
    if match := _AGE_BAND.fullmatch(dim):
        return int(match.group(1)), int(match.group(2))
    msg = f"unknown WHO age-band code {dim!r}"
    raise ValueError(msg)


def load_who_qx(path: Path) -> pl.DataFrame:
    """Single-year qx from a saved WHO GHO nMx response, in the ONS frame's shape.

    The response is the raw OData JSON of `WHO_NMX_INDICATOR` for one country and sex:
    one record per abridged age band and year, with the rate in `NumericValue`. Each
    band expands to single ages under a constant force of mortality, q = 1 - exp(-nMx).
    The open 85+ band has a finite nMx, so the same formula applies from 85 up. That is
    why nMx is used rather than nqx, which is exactly 1 in the open band.
    """
    payload = json.loads(path.read_text())
    if payload.get("@odata.nextLink"):
        msg = f"{path}: the WHO response was paginated; re-download the full series"
        raise ValueError(msg)
    rows = payload.get("value")
    if not isinstance(rows, list) or not rows:
        msg = f"{path}: the WHO response holds no records"
        raise ValueError(msg)
    sexes = {row.get("Dim1") for row in rows}
    if len(sexes) != 1:
        msg = f"{path}: expected one sex per file, found {sorted(map(str, sexes))}"
        raise ValueError(msg)
    records: list[tuple[int, int, float]] = []
    for row in rows:
        try:
            dim = str(row["Dim2"])
            year = int(row["TimeDim"])
            mx = float(row["NumericValue"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"{path}: cannot read WHO record {row}"
            raise ValueError(msg) from exc
        lo, hi = _who_age_band(dim)
        if mx < 0:
            msg = f"{path}: negative death rate in WHO record {row}"
            raise ValueError(msg)
        qx = 1 - math.exp(-mx)
        records.extend((age, year, qx) for age in range(lo, hi + 1))
    frame = pl.DataFrame(records, schema=["age", "year", "qx"], orient="row")
    duplicated = frame.group_by("age", "year").len().filter(pl.col("len") > 1)
    if duplicated.height:
        msg = f"{path}: duplicated (age, year) cells in the WHO response"
        raise ValueError(msg)
    return frame.sort("age", "year")


def load_who_rates_by_sex(data_dir: Path, country: str) -> dict[str, pl.DataFrame]:
    """The male and female WHO qx frames for one country.

    `country` is a COUNTRY value from the cBioPortal patient table. Files are keyed by
    the ISO3 code it maps to; a missing file names the download target to run.
    """
    try:
        iso = COUNTRY_TO_ISO3[country]
    except KeyError:
        msg = f"no ISO3 code for the country {country!r}"
        raise ValueError(msg) from None
    rates = {}
    for sex in ("male", "female"):
        path = data_dir / "lifetables" / "who" / f"nmx-{iso}-{sex}.json"
        if not path.is_file():
            msg = f"{path} not found; run `scripts/download.sh who {iso}` first"
            raise ValueError(msg)
        rates[sex] = load_who_qx(path)
    return rates
