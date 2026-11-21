"""Cell formatting shared by the adapters' `mapping_rows` hooks."""

from collections.abc import Iterable, Mapping
from typing import Any

INCOMPLETE = "leaves the predictor set incomplete, imputed in the main analysis"
PT3_BOUND = "; the pT3b coefficient bound runs alongside in the principal analysis"


def sex_missing(analysis: Mapping[str, Any]) -> str:
    return (
        f'missing-sex rule from params.yaml: "{analysis["missing_sex"]}" (the life-table '
        "risk needs sex)"
    )


def score_rule(analysis: Mapping[str, Any]) -> str:
    return (
        "checked against the computed score; agreement below "
        f"{analysis['score_agreement_min']:.0%} stops the dataset build"
    )


def row(
    section: str, item: str, source: str, raw_coding: str, rule: str, missing: str = ""
) -> dict[str, str]:
    return {
        "section": section,
        "item": item,
        "source": source,
        "raw_coding": raw_coding,
        "rule": rule,
        "missing": missing,
    }


def quoted(values: Iterable[str]) -> str:
    """A raw-coding cell: each allowed value in quotes, comma separated."""
    return ", ".join(f'"{value}"' for value in sorted(values, key=str.lower))


def arrows(mapping: Mapping[str, str]) -> str:
    """A rule cell for a code-to-harmonised map, rendered raw -> harmonised."""
    return ", ".join(f"{raw} -> {mapped}" for raw, mapped in mapping.items())
