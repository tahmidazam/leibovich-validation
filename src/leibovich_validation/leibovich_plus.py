import polars as pl

from leibovich_validation import leibovich, lifetables


def combine(disease_risk: float, other_risk: float) -> tuple[float, float]:
    """Return (metastasis risk, other-cause death risk) adjusted for each other.

    The overall risk 1 - (1 - R_D)(1 - R_P) is split in proportion to the two cumulative
    risks. Because the split uses cumulative risks rather than hazards, the adjusted
    metastasis risk can fall as follow-up lengthens.
    """
    overall = 1 - (1 - disease_risk) * (1 - other_risk)
    total = disease_risk + other_risk
    if total == 0:
        return 0.0, 0.0
    return overall * disease_risk / total, overall * other_risk / total


def published_risks(
    rates: pl.DataFrame,
    tumour: leibovich.Tumour,
    age: int,
    years: float,
    start_year: int,
    round_to_percent: bool,
    entry: float = 0.0,
) -> tuple[float, float]:
    """Leibovich Plus as published, at any time inside the model's domain.

    Disease risk uses the piecewise-constant baseline hazard; other-cause risk
    accumulates yearly qx plus the partial final year. With `entry` above zero, each
    cumulative risk is first conditioned on survival to `entry`, then the published
    split applies to the conditional risks.
    """
    if not 0 <= entry <= years:
        msg = f"entry must lie in [0, {years:g}] years; got {entry}"
        raise ValueError(msg)
    disease = leibovich.risk(tumour, years)
    other = lifetables.other_cause_risk(rates, age, years, start_year, round_to_percent)
    if entry:
        disease = 1 - (1 - disease) / (1 - leibovich.risk(tumour, entry))
        entry_other = lifetables.other_cause_risk(rates, age, entry, start_year, round_to_percent)
        other = 1 - (1 - other) / (1 - entry_other)
    return combine(disease, other)
