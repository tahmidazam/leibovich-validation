"""Editable LaTeX bodies for the two main manuscript tables."""

from pathlib import Path

import polars as pl

from leibovich_validation.tables import MAYO_COLUMN

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_METRICS = (
    ("observed_incidence", "Observed incidence"),
    ("oe_ratio", "Event O/E"),
    ("calibration_slope", "Calibration slope"),
    ("ici", "Integrated calibration index"),
    ("auc", "Time-dependent AUC"),
    ("brier", "Brier score"),
    ("oe_other_cause", "Other-cause O/E"),
)


def _cell(value: object) -> str:
    return "".join(_ESCAPES.get(char, char) for char in str("" if value is None else value))


def _header(columns: tuple[str, ...], widths: tuple[str, ...]) -> list[str]:
    spec = "".join(rf">{{\raggedright\arraybackslash}}p{{{width}}}" for width in widths)
    heading = " & ".join(r"\textbf{" + _cell(column) + "}" for column in columns) + r" \\"
    return [
        r"\begingroup",
        r"\small",
        r"\setstretch{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}",
        r"\hline",
        heading,
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        heading,
        r"\hline",
        r"\endhead",
    ]


def _finish(lines: list[str], path: Path) -> Path:
    if lines[-1] != r"\hline":
        lines.append(r"\hline")
    lines.extend((r"\end{longtable}", r"\endgroup"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_table_1(frame: pl.DataFrame, path: Path) -> Path:
    """Typeset the cohort characteristics with section headings and original cells."""
    columns = ("Characteristic", "Mayo cohort", "TranSORCE", "Vasudev cohort")
    lines = _header(columns, ("5.4cm", "3.2cm", "3.2cm", "3.2cm"))
    section = None
    for row in frame.iter_rows(named=True):
        if row["section"] and row["section"] != section:
            section = row["section"]
            lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + _cell(section) + r"}} \\")
        lines.append(
            " & ".join(
                _cell(row[key])
                for key in ("characteristic", MAYO_COLUMN, "TranSORCE", "Vasudev cohort")
            )
            + r" \\"
        )
    return _finish(lines, path)


def write_table_2(frame: pl.DataFrame, path: Path, horizon: float) -> Path:
    """Keep interval estimates legible by placing each measure on its own row."""
    if set(frame["horizon_years"].to_list()) != {horizon}:
        msg = "Table 2 must contain only the primary horizon"
        raise ValueError(msg)
    lines = _header(
        ("Cohort", "Model", "Measure", "Estimate (95% CI)"),
        ("2.6cm", "2.8cm", "5.0cm", "4.4cm"),
    )
    for row in frame.iter_rows(named=True):
        cohort = {"vasudev": "Vasudev cohort", "transorce": "TranSORCE"}[row["cohort"]]
        metrics = [(key, label) for key, label in _METRICS if row[key]]
        for index, (key, label) in enumerate(metrics):
            value = row[key]
            if key == "oe_other_cause" and row["other_cause_status"] == "descriptive":
                label += " (descriptive)"
            lines.append(
                " & ".join(
                    (
                        _cell(cohort) if index == 0 else "",
                        _cell(row["model"]) if index == 0 else "",
                        _cell(label),
                        _cell(value),
                    )
                )
                + (r" \\" if index == len(metrics) - 1 else r" \\*")
            )
        lines.append(r"\hline")
    return _finish(lines, path)
