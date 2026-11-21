from pathlib import Path

import polars as pl
import pytest
from lv_datasets import SchemaError

from lv_vasudev import raw as raw_vasudev

CBIOPORTAL_PREAMBLE = "#desc\n#desc2\n#STRING\tSTRING\n#1\t1\n"


def _patient_file(rows: list[list[str]]) -> str:
    header = "\t".join(raw_vasudev.PATIENT_COLUMNS)
    lines = [CBIOPORTAL_PREAMBLE + header] + ["\t".join(r) for r in rows]
    return "\n".join(lines) + "\n"


PATIENT_ROW = [
    "K1",
    "C1",
    "Discovery Cohort",
    "60",
    "Male",
    "25.0",
    "25-29.9",
    "Never Smoker",
    "Czech Republic",
    "I",
    "0",
    "Low",
    "Not Present",
    "Left Kidney",
    "Partial",
    "No",
    "0:LIVING",
    "NA",
    "0:DiseaseFree",
    "72",
    "0:LIVING",
    "72.5",
]


def _sample_file(rows: list[list[str]]) -> str:
    header = "\t".join(raw_vasudev.SAMPLE_COLUMNS)
    lines = [CBIOPORTAL_PREAMBLE + header] + ["\t".join(r) for r in rows]
    return "\n".join(lines) + "\n"


SAMPLE_ROW = [
    "K1",
    "K1",
    "45",
    "Absent",
    "2",
    "TIa",
    "TI",
    "NA/Unknown",
    "NA/Unknown",
    "CCRCC",
    "Matched",
    "Mutated",
    "VHL+1",
    "0",
]


def test_patient_reader_decodes_and_types(tmp_path: Path):
    row = list(PATIENT_ROW)
    row[4] = "NA"  # SEX missing
    path = tmp_path / "patient.txt"
    path.write_text(_patient_file([row]))
    frame = raw_vasudev.read_patient(path)
    assert frame.height == 1
    assert frame["AGE"].dtype == pl.Int64
    assert frame["SEX"][0] is None
    assert frame["RCC_RELATED_DEATH"][0] is None
    # Months are returned in years.
    assert frame["DFS_YEARS"][0] == pytest.approx(6.0)
    assert frame["RCC_YEARS"][0] == pytest.approx(72.5 / 12)
    assert "DFS_MONTHS" not in frame.columns


def test_sample_reader_types(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(_sample_file([SAMPLE_ROW]))
    frame = raw_vasudev.read_sample(path)
    assert frame["STAGE_N"][0] is None
    assert frame["FFPE_GRADE"][0] == 2
    assert frame["TUMOR_SIZE"].dtype == pl.Float64


def test_patient_reader_rejects_unexpected_column(tmp_path: Path):
    path = tmp_path / "patient.txt"
    text = _patient_file([PATIENT_ROW]).replace("BMI_CATEGORIES", "BMI_GROUP")
    path.write_text(text)
    with pytest.raises(SchemaError, match="unexpected columns"):
        raw_vasudev.read_patient(path)


def test_patient_reader_rejects_unexpected_category(tmp_path: Path):
    row = list(PATIENT_ROW)
    row[4] = "Other"  # not a known SEX
    path = tmp_path / "patient.txt"
    path.write_text(_patient_file([row]))
    with pytest.raises(SchemaError, match="unexpected value"):
        raw_vasudev.read_patient(path)


def test_patient_reader_rejects_short_row(tmp_path: Path):
    path = tmp_path / "patient.txt"
    path.write_text(_patient_file([PATIENT_ROW[:-3]]))
    with pytest.raises(SchemaError, match="fields"):
        raw_vasudev.read_patient(path)
