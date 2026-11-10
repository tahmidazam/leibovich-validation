"""Raw readers for the Vasudev cohort cBioPortal tables (study ccrcc_sjuh_2023).

A cBioPortal clinical table is tab-separated, with preamble lines beginning "#", then the
header row, then one row per record. "NA", "NA/Unknown" and empty strings are missing. The
two follow-up columns are stored in months and are returned in years as ``DFS_YEARS`` and
``RCC_YEARS``. An unexpected column, value or category raises ``SchemaError``.

Source: https://www.cbioportal.org/study/summary?id=ccrcc_sjuh_2023
"""

from pathlib import Path

import polars as pl


class SchemaError(ValueError):
    """A raw file does not match the declared schema."""


NULL_TOKENS = frozenset({"", "NA", "NA/Unknown"})

PATIENT_COLUMNS = (
    "PATIENT_ID",
    "COHORT_CATEGORY",
    "COHORT",
    "AGE",
    "SEX",
    "BMI",
    "BMI_CATEGORIES",
    "TOBACCO_STATUS",
    "COUNTRY",
    "STAGE",
    "LEIBOVICH_SCORE",
    "LEIBOVICH_RISK_GROUP",
    "RHABDOID_OR_SARCOMATOID",
    "LATERALITY",
    "NEPHRECTOMY_TYPE",
    "RELAPSE",
    "OS_STATUS",
    "RCC_RELATED_DEATH",
    "DFS_STATUS",
    "DFS_MONTHS",
    "RCC_STATUS",
    "RCC_MONTHS",
)

SAMPLE_COLUMNS = (
    "PATIENT_ID",
    "SAMPLE_ID",
    "TUMOR_SIZE",
    "NECROSIS",
    "FFPE_GRADE",
    "STAGE_T",
    "STAGE_T_CATEGORY",
    "STAGE_N",
    "STAGE_M",
    "ONCOTREE_CODE",
    "SOMATIC_STATUS",
    "VHL_MUTATION",
    "GENOMIC_GROUP",
    "TMB_NONSYNONYMOUS",
)

PATIENT_CATEGORIES: dict[str, frozenset[str]] = {
    "COHORT_CATEGORY": frozenset({"C1", "C2", "C3"}),
    "COHORT": frozenset({"Discovery Cohort", "Validation Cohort"}),
    "SEX": frozenset({"Male", "Female"}),
    "BMI_CATEGORIES": frozenset({"<24.9", "25-29.9", "30-34.9", ">35"}),
    "TOBACCO_STATUS": frozenset({"Current Smoker", "Ex-Smoker", "Never Smoker"}),
    "COUNTRY": frozenset({"Czech Republic", "United Kingdom", "Russia", "Romania", "Serbia"}),
    "STAGE": frozenset({"I", "II", "III", "IV"}),
    "LEIBOVICH_RISK_GROUP": frozenset({"Low", "Intermediate", "High"}),
    "RHABDOID_OR_SARCOMATOID": frozenset({"Not Present", "Present", "No Consensus"}),
    "LATERALITY": frozenset({"Left Kidney", "Right Kidney", "Bilateral"}),
    "NEPHRECTOMY_TYPE": frozenset({"Radical", "Partial"}),
    "RELAPSE": frozenset({"Yes", "No"}),
    "OS_STATUS": frozenset({"0:LIVING", "1:DECEASED"}),
    "RCC_RELATED_DEATH": frozenset({"Yes", "No"}),
    "DFS_STATUS": frozenset({"0:DiseaseFree", "1:Recurred/Progressed"}),
    "RCC_STATUS": frozenset({"0:LIVING", "1:DECEASED"}),
}

SAMPLE_CATEGORIES: dict[str, frozenset[str]] = {
    "NECROSIS": frozenset({"Absent", "Present"}),
    "STAGE_T": frozenset({"TIa", "TIb", "TI", "TII", "TIII", "TIV"}),
    "STAGE_T_CATEGORY": frozenset({"TI", "TII", "TIII", "TIV"}),
    "STAGE_N": frozenset({"NI"}),
    "STAGE_M": frozenset({"M0", "MI"}),
    "ONCOTREE_CODE": frozenset({"CCRCC"}),
    "SOMATIC_STATUS": frozenset({"Matched"}),
    "VHL_MUTATION": frozenset({"Mutated", "Not Mutated"}),
    "GENOMIC_GROUP": frozenset({"VHL+0", "VHL+1", "VHL+2", "VHL+>=3"}),
}

PATIENT_NUMERIC: dict[str, pl.DataType] = {
    "AGE": pl.Int64(),
    "BMI": pl.Float64(),
    "LEIBOVICH_SCORE": pl.Int64(),
    "DFS_MONTHS": pl.Float64(),
    "RCC_MONTHS": pl.Float64(),
}

SAMPLE_NUMERIC: dict[str, pl.DataType] = {
    "TUMOR_SIZE": pl.Float64(),
    "FFPE_GRADE": pl.Int64(),
    "TMB_NONSYNONYMOUS": pl.Float64(),
}

MONTHS_TO_YEARS = {"DFS_MONTHS": "DFS_YEARS", "RCC_MONTHS": "RCC_YEARS"}


def _typed_value(text: str, column: str, dtype: pl.DataType) -> int | float:
    try:
        if dtype == pl.Int64:
            return int(text)
        return float(text)
    except ValueError:
        raise SchemaError(f"{column}: cannot parse {text!r} as {dtype}") from None


def _read_cbioportal(
    path: Path,
    columns: tuple[str, ...],
    categories: dict[str, frozenset[str]],
    numeric: dict[str, pl.DataType],
) -> pl.DataFrame:
    lines = path.read_text().splitlines()
    try:
        header_at = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    except StopIteration:
        raise SchemaError(f"{path}: no header row found") from None
    header = lines[header_at].split("\t")
    if header != list(columns):
        raise SchemaError(
            f"{path}: unexpected columns\n  expected: {list(columns)}\n  found:    {header}"
        )
    parsed: dict[str, list[int | float | str | None]] = {c: [] for c in columns}
    for lineno, line in enumerate(lines[header_at + 1 :], start=header_at + 2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(columns):
            raise SchemaError(
                f"{path}:{lineno}: expected {len(columns)} fields, found {len(fields)}"
            )
        for column, text in zip(columns, fields, strict=True):
            value: int | float | str | None = None if text in NULL_TOKENS else text
            if value is not None and column in categories and value not in categories[column]:
                raise SchemaError(f"{path}:{lineno}: {column} has unexpected value {value!r}")
            if value is not None and column in numeric:
                value = _typed_value(value, column, numeric[column])
            parsed[column].append(value)
    frame = pl.DataFrame(parsed)
    for months, years in MONTHS_TO_YEARS.items():
        if months in frame.columns:
            frame = frame.with_columns((pl.col(months) / 12.0).alias(years)).drop(months)
    return frame


def read_patient(path: Path) -> pl.DataFrame:
    """Read ``data_clinical_patient.txt`` into a typed frame, one row per patient."""
    return _read_cbioportal(path, PATIENT_COLUMNS, PATIENT_CATEGORIES, PATIENT_NUMERIC)


def read_sample(path: Path) -> pl.DataFrame:
    """Read ``data_clinical_sample.txt`` into a typed frame, one row per sample."""
    return _read_cbioportal(path, SAMPLE_COLUMNS, SAMPLE_CATEGORIES, SAMPLE_NUMERIC)
