"""Raw readers for the TranSORCE spreadsheets (BioImage Archive S-BIAD2453).

Missing values use SPSS-style codes ".a", ".b" and ".c", which decode to null; any other
dot-letter code raises. Excel serial dates become ``datetime.date``. Each workbook is
checked against a declared schema: an unexpected column, category or code raises
``SchemaError``.

The declared schema was reconciled against the real workbook: the ``ccRCC-Transorce``
sheet carries every clinical field including the recurrence-site flags and the free-text
"Other: Details". ``analysis.xlsx`` holds image-analysis morphology, which the validation
does not use. Downstream code uses the canonical names, never the raw headers.

Source: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2453
"""

import datetime as dt
import re
from pathlib import Path

import openpyxl
import polars as pl

from leibovich_validation.raw_vasudev import SchemaError

KNOWN_MISSING_CODES = frozenset({".a", ".b", ".c"})
_MISSING_CODE = re.compile(r"^\.[a-zA-Z]$")
NULL_TOKENS = frozenset({"", "NA"})

# Column kinds: "str" free text, "int", "num", "date", "cat" (categorical, checked
# against CATEGORIES under the canonical name). The workbook holds a ccRCC sheet and a
# pRCC sheet; only ccRCC is eligible, and it is read by name.
CCLRCC_SHEET = "ccRCC-Transorce"

CLINICAL_SCHEMA: dict[str, tuple[str, str]] = {
    "Image Tag": ("image_tag", "str"),
    "iCAIRD ref": ("icaird_ref", "str"),
    "Patient Trial Identifier number": ("trial_identifier", "str"),
    "Patient ID": ("patient_id", "int"),
    "Treatment Arm": ("arm", "cat"),
    "Age at Randomisation": ("age_at_randomisation", "num"),
    "Time from Diagnosis to Randomisation": ("diagnosis_to_randomisation", "num"),
    # The raw header is " Randomisation Date" with a leading space; headers are stripped.
    "Randomisation Date": ("randomisation_date", "date"),
    "Date of Operation": ("operation_date", "date"),
    "Gender": ("sex", "cat"),
    "WHO Performance Status": ("who_ps", "cat"),
    "Histologyat Diagnosis": ("histology", "cat"),
    "T stage": ("t_stage", "cat"),
    "Regional Lymph Node Status": ("node_status", "cat"),
    "Tumour Size": ("tumour_size_category", "cat"),
    "ISUP Nuclear Grade": ("isup_grade", "int"),
    "Tumour Necrosis": ("necrosis", "cat"),
    "Raw Leibovich Score": ("leibovich_score", "int"),
    "Leibovich Group": ("leibovich_risk_group", "cat"),
    "Maximum Diameter of Primary Tumour": ("max_diameter_mm", "num"),
    "DFS 0/1": ("dfs_status", "int"),
    "DFS Date": ("dfs_date", "date"),
    "OS 0/1": ("os_status", "int"),
    "OS Date": ("os_date", "date"),
    "Local Recurrence": ("local_recurrence", "cat"),
    "Contrlateral Kidney Metastasis": ("contralateral_kidney_metastasis", "cat"),
    "Nodal Metastasis": ("nodal_metastasis", "cat"),
    "Lung Metastasis": ("lung_metastasis", "cat"),
    "Bone Metastasis": ("bone_metastasis", "cat"),
    "Liver Metastasis": ("liver_metastasis", "cat"),
    "Other Metastasis": ("other_metastasis", "cat"),
    "Other: Details": ("other_details", "str"),
    "Method of Assessment": ("method_of_assessment", "cat"),
    "Other Method: Details": ("other_method_details", "str"),
}

_YES_NO = frozenset({"Yes", "No"})
CATEGORIES: dict[str, frozenset[str]] = {
    "arm": frozenset({"A", "B", "C"}),
    "sex": frozenset({"M", "F"}),
    "who_ps": frozenset({"0", "1", "2"}),
    "histology": frozenset({"Conventional/ Clear Cell"}),
    # "pT3a - 4" is how T3a tumours are written in this workbook.
    "t_stage": frozenset({"pT1a", "pT1b", "pT2", "pT3a - 4"}),
    "node_status": frozenset({"pNx/ pN0", "pN1/ pN2"}),
    "tumour_size_category": frozenset({"<10", ">10"}),
    "necrosis": _YES_NO,
    "leibovich_risk_group": frozenset({"0", "1"}),
    # Local recurrence is recorded as a site, not a flag.
    "local_recurrence": frozenset({"No", "Renal bed", "Local node", "Remnant kidney"}),
    "contralateral_kidney_metastasis": _YES_NO,
    "nodal_metastasis": _YES_NO,
    "lung_metastasis": _YES_NO,
    "bone_metastasis": _YES_NO,
    "liver_metastasis": _YES_NO,
    "other_metastasis": _YES_NO,
    "method_of_assessment": frozenset({"CT Scan", "Chest X-Ray", "Other"}),
}

_EXCEL_EPOCH = dt.date(1899, 12, 30)


def _decode(value: object, canonical: str, header: str) -> str | None:
    """Return the cell as a string, with missing codes decoded to null."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in NULL_TOKENS:
            return None
        if _MISSING_CODE.match(text):
            if text not in KNOWN_MISSING_CODES:
                raise SchemaError(f"{canonical} ({header}): unknown missing-value code {text!r}")
            return None
        return text
    return str(value)


def _to_date(value: object, canonical: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, int | float):
        return _EXCEL_EPOCH + dt.timedelta(days=int(value))
    raise SchemaError(f"{canonical}: cannot parse {value!r} as a date")


def _convert(text: str | None, canonical: str, kind: str) -> object:
    if text is None:
        return None
    if kind == "str":
        return text
    if kind == "cat":
        if text not in CATEGORIES[canonical]:
            raise SchemaError(f"{canonical}: unexpected value {text!r}")
        return text
    if kind == "int":
        try:
            return int(float(text))
        except ValueError:
            raise SchemaError(f"{canonical}: cannot parse {text!r} as an integer") from None
    if kind == "num":
        try:
            return float(text)
        except ValueError:
            raise SchemaError(f"{canonical}: cannot parse {text!r} as a number") from None
    return text


def read_sheet(path: Path, schema: dict[str, tuple[str, str]], sheet: str) -> pl.DataFrame:
    """Read the named sheet of `path` into a typed frame with canonical column names."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in workbook.sheetnames:
        raise SchemaError(f"{path}: no sheet {sheet!r}; found {workbook.sheetnames}")
    rows = list(workbook[sheet].iter_rows(values_only=True))
    if not rows:
        raise SchemaError(f"{path}: empty workbook")
    headers = ["" if h is None else str(h).strip() for h in rows[0]]
    expected = list(schema)
    if sorted(headers) != sorted(expected):
        raise SchemaError(
            f"{path}: unexpected columns\n  expected: {expected}\n  found:    {headers}"
        )
    parsed: dict[str, list] = {canonical: [] for canonical, _ in schema.values()}
    for lineno, row in enumerate(rows[1:], start=2):
        if all(cell is None or str(cell).strip() == "" for cell in row):
            continue
        if len(row) != len(headers):
            raise SchemaError(f"{path}:{lineno}: expected {len(headers)} cells, found {len(row)}")
        for header, cell in zip(headers, row, strict=True):
            canonical, kind = schema[header]
            if kind == "date":
                value: object = (
                    None
                    if cell is None or _decode(cell, canonical, header) is None
                    else _to_date(cell, canonical)
                )
            else:
                value = _convert(_decode(cell, canonical, header), canonical, kind)
            parsed[canonical].append(value)
    return pl.DataFrame(parsed)


def read_clinical(path: Path) -> pl.DataFrame:
    """Read the ccRCC sheet of ``clinical.xlsx`` (SORCE Clinical Data) into a typed frame."""
    return read_sheet(path, CLINICAL_SCHEMA, CCLRCC_SHEET)
