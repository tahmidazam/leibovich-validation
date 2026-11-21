import datetime as dt
from pathlib import Path

import openpyxl
import polars as pl
import pytest
from lv_datasets import SchemaError

from lv_transorce import raw as raw_transorce


def _workbook(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = raw_transorce.CCLRCC_SHEET
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    book.save(path)


def _transorce_row() -> list[object]:
    return [
        "iCAIRD-1100-TS1_1.svs",
        "iCAIRD-1100",
        "003001",
        3001,
        "A",
        74.5,
        90,
        dt.datetime(2008, 2, 20),  # datetime cell
        39_462,  # Excel serial for 2008-01-15
        "M",
        "0",
        "Conventional/ Clear Cell",
        "pT3a - 4",
        "pNx/ pN0",
        "<10",
        3,
        "Yes",
        6,
        "0",
        55,
        0,
        dt.datetime(2013, 1, 15),
        0,
        ".a",  # OS date missing
        "No",
        "No",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "Brain",
        "CT Scan",
        "NA",  # Other Method: Details missing
    ]


def test_transorce_reader_decodes_codes_dates_and_types(tmp_path: Path):
    path = tmp_path / "clinical.xlsx"
    headers = list(raw_transorce.CLINICAL_SCHEMA)
    _workbook(path, headers, [_transorce_row()])
    frame = raw_transorce.read_clinical(path)
    assert frame["patient_id"][0] == 3001
    assert frame["operation_date"][0] == dt.date(2008, 1, 15)
    assert frame["randomisation_date"][0] == dt.date(2008, 2, 20)
    assert frame["os_date"][0] is None
    assert frame["other_method_details"][0] is None
    assert frame["isup_grade"].dtype == pl.Int64


def test_transorce_reader_rejects_unknown_missing_code(tmp_path: Path):
    path = tmp_path / "clinical.xlsx"
    headers = list(raw_transorce.CLINICAL_SCHEMA)
    row = _transorce_row()
    row[12] = ".z"  # unknown code in T stage
    _workbook(path, headers, [row])
    with pytest.raises(SchemaError, match="missing-value code"):
        raw_transorce.read_clinical(path)


def test_transorce_reader_rejects_unexpected_column(tmp_path: Path):
    path = tmp_path / "clinical.xlsx"
    headers = list(raw_transorce.CLINICAL_SCHEMA)
    headers[0] = "Patient"
    _workbook(path, headers, [])
    with pytest.raises(SchemaError, match="unexpected columns"):
        raw_transorce.read_clinical(path)


def test_transorce_reader_rejects_unexpected_category(tmp_path: Path):
    path = tmp_path / "clinical.xlsx"
    headers = list(raw_transorce.CLINICAL_SCHEMA)
    row = _transorce_row()
    row[12] = "pT5"  # not a known T stage
    _workbook(path, headers, [row])
    with pytest.raises(SchemaError, match="unexpected value"):
        raw_transorce.read_clinical(path)
