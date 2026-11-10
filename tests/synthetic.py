"""Synthetic raw-data builders shared by the cohort and pipeline tests.

Each builder writes a file in the real raw format: the Vasudev cohort cBioPortal
clinical tables, the TranSORCE ccRCC workbook and an ONS-shaped life-table
workbook, so tests exercise the real readers and never a mock of them.
"""

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path

import openpyxl
import polars as pl

from leibovich_validation import raw_transorce, raw_vasudev

# cBioPortal clinical tables start with comment lines before the TSV header.
CBIOPORTAL_PREAMBLE = "#desc\n#desc2\n#STRING\tSTRING\n#1\t1\n"


def write_cbioportal(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """One cBioPortal table: preamble lines, the column header, one line per row."""
    header = "\t".join(columns)
    lines = [CBIOPORTAL_PREAMBLE + header]
    for row in rows:
        lines.append("\t".join("NA" if row.get(c) is None else str(row[c]) for c in columns))
    path.write_text("\n".join(lines) + "\n")


def vasudev_patient(patient_id: str, **overrides) -> dict:
    """A baseline Vasudev cohort patient row: alive, disease-free, every field recorded."""
    row = {
        "PATIENT_ID": patient_id,
        "COHORT_CATEGORY": "C2",
        "COHORT": "Validation Cohort",
        "AGE": 60,
        "SEX": "Male",
        "BMI": 25.0,
        "BMI_CATEGORIES": "25-29.9",
        "TOBACCO_STATUS": "Never Smoker",
        "COUNTRY": "United Kingdom",
        "STAGE": "I",
        "LEIBOVICH_SCORE": 0,
        "LEIBOVICH_RISK_GROUP": "Low",
        "RHABDOID_OR_SARCOMATOID": "Not Present",
        "LATERALITY": "Left Kidney",
        "NEPHRECTOMY_TYPE": "Radical",
        "RELAPSE": "No",
        "OS_STATUS": "0:LIVING",
        "RCC_RELATED_DEATH": None,
        "DFS_STATUS": "0:DiseaseFree",
        "DFS_MONTHS": 72,
        "RCC_STATUS": "0:LIVING",
        "RCC_MONTHS": 72,
    }
    row.update(overrides)
    return row


def vasudev_sample(patient_id: str, **overrides) -> dict:
    """A baseline Vasudev cohort sample row: pT1a, grade 2, 45 mm, node status unknown."""
    row = {
        "PATIENT_ID": patient_id,
        "SAMPLE_ID": f"{patient_id}-T",
        "TUMOR_SIZE": 45,
        "NECROSIS": "Absent",
        "FFPE_GRADE": 2,
        "STAGE_T": "TIa",
        "STAGE_T_CATEGORY": "TI",
        "STAGE_N": None,
        "STAGE_M": None,
        "ONCOTREE_CODE": "CCRCC",
        "SOMATIC_STATUS": "Matched",
        "VHL_MUTATION": "Mutated",
        "GENOMIC_GROUP": "VHL+1",
        "TMB_NONSYNONYMOUS": 0,
    }
    row.update(overrides)
    return row


def write_vasudev_tables(
    directory: Path, patients: list[dict], samples: list[dict]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write and re-read the two cBioPortal tables, returning typed frames."""
    patient_path = directory / "data_clinical_patient.txt"
    sample_path = directory / "data_clinical_sample.txt"
    write_cbioportal(patient_path, raw_vasudev.PATIENT_COLUMNS, patients)
    write_cbioportal(sample_path, raw_vasudev.SAMPLE_COLUMNS, samples)
    return raw_vasudev.read_patient(patient_path), raw_vasudev.read_sample(sample_path)


def transorce_patient(identifier: str, **overrides) -> dict:
    """A baseline TranSORCE workbook row: censored, every field recorded, dates valid."""
    row = {
        "image_tag": f"{identifier}.svs",
        "icaird_ref": f"iCAIRD-{identifier}",
        "trial_identifier": identifier,
        "patient_id": int(identifier),
        "arm": "A",
        "age_at_randomisation": 60.0,
        "diagnosis_to_randomisation": 60.0,
        "randomisation_date": dt.datetime(2008, 3, 1),
        "operation_date": dt.datetime(2008, 1, 15),  # 46-day lag, inside the window
        "sex": "M",
        "who_ps": "0",
        "histology": "Conventional/ Clear Cell",
        "t_stage": "pT2",
        "node_status": "pNx/ pN0",
        "tumour_size_category": "<10",
        "isup_grade": 2,
        "necrosis": "No",
        "leibovich_score": 3,  # pT2 alone; agrees with the computed score
        "leibovich_risk_group": "0",
        "max_diameter_mm": 60.0,
        "dfs_status": 0,
        "dfs_date": dt.datetime(2015, 1, 1),
        "os_status": 0,
        "os_date": dt.datetime(2015, 6, 1),
        "local_recurrence": "No",
        "contralateral_kidney_metastasis": "No",
        "nodal_metastasis": "No",
        "lung_metastasis": "No",
        "bone_metastasis": "No",
        "liver_metastasis": "No",
        "other_metastasis": "No",
        "other_details": None,
        "method_of_assessment": None,
        "other_method_details": None,
    }
    row.update(overrides)
    return row


def write_transorce_workbook(path: Path, rows: list[dict]) -> None:
    """The TranSORCE clinical.xlsx with the ccRCC sheet's real headers."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = raw_transorce.CCLRCC_SHEET
    sheet.append(list(raw_transorce.CLINICAL_SCHEMA))
    canonical = [canonical for canonical, _ in raw_transorce.CLINICAL_SCHEMA.values()]
    for row in rows:
        sheet.append([row.get(c) for c in canonical])
    book.save(path)


def write_life_table(
    path: Path, qx_per_100k: float = 500.0, years: Iterable[int] = range(2024, 2040)
) -> None:
    """An ONS-shaped qx workbook with a constant rate at every age and year.

    Covers ages 0-110 and calendar years 2024-2039 by default, enough for the
    longest horizon the models and the behaviour summary reach. `years`
    widens the range for the calendar-matched sensitivity variant.
    """
    book = openpyxl.Workbook()
    years = list(years)
    for index, name in enumerate(("males period qx", "females period qx")):
        sheet = book.active if index == 0 else book.create_sheet()
        sheet.title = name
        sheet.append(["Exact age", *[f"Year {year}" for year in years]])
        for age in range(111):
            sheet.append([age, *([qx_per_100k] * len(years))])
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)


def constant_rates(
    qx: float, max_age: int = 110, first_year: int = 2020, last_year: int = 2050
) -> pl.DataFrame:
    """A synthetic life table with the same qx at every age and calendar year.

    The default span covers the horizons the models and the behaviour
    summary reach, so the other-cause algebra can be checked by hand.
    """
    records = [
        (age, year, qx) for age in range(max_age + 1) for year in range(first_year, last_year + 1)
    ]
    return pl.DataFrame(records, schema=["age", "year", "qx"], orient="row")


# The 19 abridged age bands of the WHO GHO life tables, in order, with the
# codes the GHO API returns.
WHO_AGE_BANDS = (
    "AGEGROUP_YEARS00-01",
    "AGEGROUP_YEARS01-04",
    *(f"AGEGROUP_YEARS{lo:02d}-{lo + 4:02d}" for lo in range(5, 85, 5)),
    "AGEGROUP_YEARS85PLUS",
)


def who_nmx_records(
    nmx: float, years: range, sex: str = "SEX_MLE", country: str = "TST"
) -> list[dict]:
    """One WHO "value" record per age band and calendar year, at a constant nMx."""
    return [
        {"Dim1": sex, "Dim2": band, "TimeDim": year, "NumericValue": nmx, "SpatialDim": country}
        for band in WHO_AGE_BANDS
        for year in years
    ]


def write_who_nmx(path: Path, records: list[dict]) -> None:
    """A saved WHO GHO nMx response in the real OData envelope."""
    payload = {
        "@odata.context": "https://ghoapi.azureedge.net/api/$metadata#LIFE_0000000029",
        "value": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
