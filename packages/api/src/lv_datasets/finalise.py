from dataclasses import dataclass

import polars as pl

from lv_datasets.dataset import (
    HARMONISED_COLUMNS,
    PREDICTOR_COLUMNS,
    Dataset,
    SchemaError,
    leibovich_score,
    validate_patient_table,
)


@dataclass(frozen=True)
class Finalised:
    """The retained patient table plus the aggregate outputs."""

    patients: pl.DataFrame
    flow: pl.DataFrame
    score_agreement: pl.DataFrame


def _score_agreement(frame: pl.DataFrame, minimum: float) -> pl.DataFrame:
    """Computed-versus-recorded score agreement over every raw row, counted by rule.

    A dataset with no recorded score has nothing to compare and passes with zero counts.
    """
    compared = frame.filter(
        pl.col("leibovich_score_computed").is_not_null()
        & pl.col("leibovich_score_recorded").is_not_null()
    )
    agree = compared.filter(
        pl.col("leibovich_score_computed") == pl.col("leibovich_score_recorded")
    ).height
    explained = (
        dict(compared["score_explained_by"].drop_nulls().value_counts(sort=False).iter_rows())
        if "score_explained_by" in compared.columns
        else {}
    )
    rows = [("total_compared", compared.height), ("agree", agree)]
    rows += [(f"disagrees_as_{rule}", n) for rule, n in explained.items()]
    rows.append(("disagrees_unexplained", compared.height - agree - sum(explained.values())))
    if compared.height and agree / compared.height < minimum:
        raise SchemaError(
            f"computed-versus-recorded Leibovich score agreement is "
            f"{agree / compared.height:.1%}, below the {minimum:.0%} stop threshold; "
            "check the predictor mappings before scoring"
        )
    return pl.DataFrame(rows, schema=["rule", "n"], orient="row")


def finalise(
    dataset: Dataset,
    frame: pl.DataFrame,
    notes: dict[str, int],
    score_agreement_min: float,
) -> Finalised:
    """Derive the flow, computed score, score agreement and patient table of one dataset.

    `frame` and `notes` are the output of `dataset.harmonise`. Raises SchemaError when
    the patient table breaks the schema, a flow step is undeclared or has no count, the
    flow does not reconcile with the raw row count, or score agreement
    falls below `score_agreement_min`.
    """
    frame = frame.with_columns(
        pl.lit(dataset.name).alias("cohort"),
        leibovich_score(*(pl.col(name) for name in PREDICTOR_COLUMNS)).alias(
            "leibovich_score_computed"
        ),
    )
    patients = frame.filter(pl.col("excluded").is_null())
    undeclared = (set(notes) | set(frame["excluded"].drop_nulls())) - set(dataset.flow_steps)
    if undeclared:
        raise SchemaError(
            f"flow steps missing from {dataset.name} flow_steps: {sorted(undeclared)}"
        )
    counts = {
        step: frame.filter(pl.col("excluded") == step).height
        for step, (kind, _) in dataset.flow_steps.items()
        if kind == "excluded"
    }
    counts |= notes | {"analysis_set": patients.height}
    absent = [step for step in dataset.flow_steps if step not in counts]
    if absent:
        raise SchemaError(f"{dataset.name} declares flow steps with no count: {absent}")
    rows_in = next(step for step, (kind, _) in dataset.flow_steps.items() if kind == "main")
    removed = sum(n for step, n in counts.items() if dataset.flow_steps[step][0] == "excluded")
    if counts[rows_in] != frame.height or removed + patients.height != frame.height:
        raise SchemaError(
            f"{dataset.name} flow does not reconcile: {counts[rows_in]} rows in, {removed} "
            f"excluded and {patients.height} analysed, from {frame.height} raw rows"
        )
    flow = pl.DataFrame(
        [(step, counts[step]) for step in dataset.flow_steps], schema=["step", "n"], orient="row"
    )
    internal = {*HARMONISED_COLUMNS, "excluded", "score_explained_by"}
    patients = patients.select(
        *HARMONISED_COLUMNS, *(c for c in frame.columns if c not in internal)
    )
    validate_patient_table(patients)
    return Finalised(patients, flow, _score_agreement(frame, score_agreement_min))
