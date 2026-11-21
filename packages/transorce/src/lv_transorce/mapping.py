from typing import Any

import polars as pl
from lv_datasets import S2_SECTIONS, SIZE_THRESHOLD_MM
from lv_datasets.mapping import (
    INCOMPLETE,
    PT3_BOUND,
    arrows,
    quoted,
    row,
    score_rule,
    sex_missing,
)

from lv_transorce import raw
from lv_transorce.harmonise import (
    DISTANT_FLAGS,
    NODE_POSITIVE,
    OTHER_DETAILS_MAPPING,
    SEX,
    T_STAGE,
    YES,
    load_other_details_mapping,
)

PREDICTORS, DESIGN, TIMING, EVENT, SCORE = S2_SECTIONS

# The reviewed mapping's repository path, as quoted in Table S2.
MAPPING_REF = f"packages/transorce/src/lv_transorce/{OTHER_DETAILS_MAPPING.name}"

_HEADERS = {name: header for header, (name, _kind) in raw.CLINICAL_SCHEMA.items()}


def _source(*canonical: str) -> str:
    return "; ".join(_HEADERS[name] for name in canonical)


def mapping_rows(params: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, str]]:
    categories = raw.CATEGORIES
    window = " to ".join(str(d) for d in params["operation_window"])
    lag = " to ".join(str(d) for d in params["randomisation_lag_days"]) + " days"
    *first, last = (name.removesuffix("_metastasis") for name in DISTANT_FLAGS)
    distant_sites = f"{', '.join(first)} or {last}"
    return [
        row(
            PREDICTORS,
            "t_stage",
            _source("t_stage"),
            quoted(categories["t_stage"]),
            arrows(T_STAGE) + PT3_BOUND,
            f"missing or unrecorded stage {INCOMPLETE}",
        ),
        row(
            PREDICTORS,
            "node_positive",
            _source("node_status"),
            quoted(categories["node_status"]),
            f'"{NODE_POSITIVE}" counts as node-positive; the combined pNx/pN0 category '
            "counts as node-negative because pNx scores as pN0 ",
            "missing counts as node-negative",
        ),
        row(
            PREDICTORS,
            "size_ge_10cm",
            _source("max_diameter_mm", "tumour_size_category"),
            f"millimetres; category {quoted(categories['tumour_size_category'])}",
            f"a diameter of {SIZE_THRESHOLD_MM} mm or more counts as 10 cm or greater; "
            "the recorded category applies when the diameter is missing ",
            f"diameter and category both missing {INCOMPLETE}",
        ),
        row(
            PREDICTORS,
            "grade",
            _source("isup_grade"),
            "integer grade",
            "ISUP grade stands in for Fuhrman grade",
            f"missing grade {INCOMPLETE}",
        ),
        row(
            PREDICTORS,
            "necrosis",
            _source("necrosis"),
            quoted(categories["necrosis"]),
            f'"{YES}" counts as necrosis',
            "missing counts as absent",
        ),
        row(DESIGN, "patient_id", _source("trial_identifier"), "text", "kept as text", "required"),
        row(
            DESIGN,
            "eligibility",
            _source("operation_date", "randomisation_date"),
            "Excel dates",
            "every ccRCC workbook row is eligible by the trial record; an operation "
            f"outside {window} or an operation-to-randomisation lag outside {lag} is "
            "excluded and counted (audit date checks)",
            "a missing operation date is estimated; both dates missing stops the build",
        ),
        row(
            DESIGN,
            "sex",
            _source("sex"),
            quoted(categories["sex"]),
            arrows(SEX),
            sex_missing(analysis),
        ),
        row(
            DESIGN,
            "age_at_surgery",
            _source("age_at_randomisation"),
            "years",
            "age at randomisation minus the operation-to-randomisation lag",
            "a missing age stops the build",
        ),
        row(
            DESIGN,
            "arm",
            _source("arm"),
            quoted(categories["arm"]),
            "recorded as the randomised arm; no analysis uses it",
            "stays null",
        ),
        row(
            DESIGN,
            "nephrectomy_type",
            "not recorded",
            "",
            "the workbook records no operation type; null",
            "always null",
        ),
        row(
            TIMING,
            "operation_year",
            _source("operation_date"),
            "Excel date",
            "the operation's calendar year",
            "a missing operation date is estimated as randomisation minus the "
            "cohort-median lag and flagged",
        ),
        row(
            TIMING,
            "entry_time",
            _source("operation_date", "randomisation_date"),
            "Excel dates",
            "the operation-to-randomisation lag in years; the patient enters the risk "
            "set at randomisation (left truncation)",
            "estimated along with the operation date",
        ),
        row(
            TIMING,
            "time",
            _source("dfs_date", "os_date", "operation_date"),
            "Excel dates",
            "event or censoring date minus the operation date, in years",
            "a missing DFS status or date stops the build",
        ),
        row(
            EVENT,
            "event of interest",
            _source("dfs_status", "dfs_date", *DISTANT_FLAGS, "other_metastasis", "other_details"),
            '0/1 status; "No"/"Yes" flags; free text',
            f"the first recorded distant site: any of the {distant_sites} flags, or an "
            "Other: Details string classed distant_metastasis by the approved mapping in "
            f"{MAPPING_REF}; the contralateral-kidney flag and second primaries are not "
            "the event",
            "a missing DFS status or date stops the build; an unmapped Other: Details "
            "string stops the build",
        ),
        row(
            EVENT,
            "competing event",
            _source("os_status", "os_date"),
            "0/1 status; Excel date",
            "death without distant metastasis: a recorded death with no DFS event, or a "
            "DFS event that is the death itself",
            "a recorded death with no date stops the build",
        ),
        row(
            EVENT,
            "censoring",
            _source("dfs_status", "dfs_date", "local_recurrence", "other_details"),
            '0/1 status; "No"/"Yes"; free text',
            "a DFS event with no distant site, a local-only recurrence or a second "
            "primary censors the patient at that event; otherwise the last "
            "disease-free date censors",
        ),
        row(
            SCORE,
            "leibovich_score_recorded",
            _source("leibovich_score"),
            "integer",
            score_rule(analysis),
            "missing scores are not compared",
        ),
    ]


def mapping_tables() -> dict[str, pl.DataFrame]:
    """The "Other: Details" mapping as a publication table.

    One row per distinct free-text string in the approved mapping, with its class and
    the flagged and laterality marks.
    """
    rows = [
        {
            "other_details": text,
            "class": entry["class"],
            "flagged": "yes" if entry.get("flagged") else "",
            "laterality": "yes" if entry.get("laterality") else "",
        }
        for text, entry in load_other_details_mapping().items()
    ]
    return {"table_s2_other_details": pl.DataFrame(rows)}
