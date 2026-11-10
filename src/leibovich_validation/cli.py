from dataclasses import replace
from pathlib import Path
from typing import cast

import polars as pl
import typer

from leibovich_validation import (
    audit,
    cohort,
    export,
    impute,
    lifetables,
    manifest,
    params,
    predict,
    raw_transorce,
    raw_vasudev,
    rbridge,
    report,
    score,
    sensitivity,
    tables,
    tables_latex,
)
from leibovich_validation.config import REPO_ROOT, get_settings

app = typer.Typer(no_args_is_help=True)

INTERIM_DIR = REPO_ROOT / "data" / "interim"
SCORE_DIR = REPO_ROOT / "results" / "scores"
SENSITIVITY_DIR = REPO_ROOT / "results" / "sensitivity"
COHORT_DIR = REPO_ROOT / "results" / "cohort"
IMPUTE_DIR = REPO_ROOT / "results" / "imputation"

# Display names used for the `stratum` column role in summary tables.
COHORT_LABELS = {"transorce": "TranSORCE", "vasudev": "Vasudev cohort"}

OTHER_DETAILS_MAPPING = REPO_ROOT / "docs" / "transorce-other-details-mapping.yaml"


def _data_dir() -> Path:
    data_dir = get_settings().data_dir
    if not data_dir.is_dir():
        raise typer.BadParameter(f"LP_DATA_DIR does not exist: {data_dir}")
    return data_dir


def _cohort_label(name: str) -> str:
    try:
        return COHORT_LABELS[name]
    except KeyError:
        raise typer.BadParameter(f"unknown cohort {name!r}") from None


@app.command("manifest")
def manifest_cmd(out: Path = REPO_ROOT / "data" / "manifest.json") -> None:
    """Write a checksum manifest of everything under LP_DATA_DIR."""
    count = manifest.write(_data_dir(), out)
    typer.echo(f"{count} files -> {out}")


def _surgery_clock_patients(name: str) -> pl.DataFrame:
    """The cohort's patients timed from surgery with delayed entry (surgery-clock analysis)."""
    return cohort.surgery_clock(pl.read_parquet(INTERIM_DIR / f"{name}_patients.parquet"))


def _scoring_patients(name: str) -> pl.DataFrame:
    """The cohort's patients on the randomisation clock, as the later stages use them."""
    return cohort.reorigin_at_randomisation(
        pl.read_parquet(INTERIM_DIR / f"{name}_patients.parquet")
    )


@app.command("doctor")
def doctor_cmd() -> None:
    """Check that rpy2 can reach R and the pinned R packages load."""
    failures = 0
    for check in rbridge.doctor():
        typer.echo(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
        if not check.ok:
            failures += 1
            typer.echo(f"     fix: {check.fix}")
    if failures:
        raise typer.Exit(code=1)
    typer.echo("all checks passed")


@app.command("cohort")
def cohort_cmd(name: str = "transorce") -> None:
    """Build the harmonised patient table and flow counts for one cohort."""
    _cohort_label(name)
    settings = params.load_analysis()
    data_dir = _data_dir()
    if name == "vasudev":
        vasudev_dir = data_dir / "vasudev"
        result = cohort.harmonise_vasudev(
            raw_vasudev.read_patient(vasudev_dir / "data_clinical_patient.txt"),
            raw_vasudev.read_sample(vasudev_dir / "data_clinical_sample.txt"),
            score_agreement_min=settings.score_agreement_min,
        )
    else:
        audit_settings = params.load_audit()
        clinical = raw_transorce.read_clinical(data_dir / "transorce" / "clinical.xlsx")
        mapping = cohort.load_other_details_mapping(OTHER_DETAILS_MAPPING)
        result = cohort.harmonise_transorce(
            clinical,
            mapping,
            operation_window=audit_settings.operation_window,
            lag_days=audit_settings.randomisation_lag_days,
            score_agreement_min=settings.score_agreement_min,
        )
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    patient_path = INTERIM_DIR / f"{name}_patients.parquet"
    result.patients.write_parquet(patient_path)
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    flow_path = COHORT_DIR / f"{name}_flow.csv"
    agreement_path = COHORT_DIR / f"{name}_score_agreement.csv"
    result.flow.write_csv(flow_path)
    result.score_agreement.write_csv(agreement_path)
    typer.echo(f"{result.patients.height} patients -> {patient_path}")
    for step, n in result.flow.iter_rows():
        typer.echo(f"  {step}: {n}")


@app.command("impute")
def impute_cmd(name: str = "transorce", profile: str = "main") -> None:
    """Impute missing Leibovich predictors by MICE for one cohort."""
    _cohort_label(name)
    settings = params.load_analysis(profile)
    patients = _scoring_patients(name)
    imputed = impute.impute_frame(patients, settings.imputations, settings.seed)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    imputed_path = INTERIM_DIR / f"{name}_imputed.parquet"
    imputed.write_parquet(imputed_path)
    IMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = IMPUTE_DIR / f"{name}_missingness.csv"
    impute.missingness_summary(patients).write_csv(summary_path)
    n_imputations = imputed["imputation"].n_unique()
    typer.echo(f"{imputed.height} rows in {n_imputations} completed datasets -> {imputed_path}")
    typer.echo(f"missingness counts -> {summary_path}")


@app.command("predict")
def predict_cmd(name: str = "transorce", complete_case: bool = False) -> None:
    """Write per-patient risks per completed dataset, plus the complete-case file.

    The main file has one row per patient per imputation; the `_cc` file keeps
    the complete-case predictions for the sensitivity analysis. With
    --complete-case only the `_cc` file is written, for runs without the
    imputed table.
    """
    _cohort_label(name)
    settings = params.load_analysis()
    model = params.load_model()
    data_dir = _data_dir()
    patients = _scoring_patients(name)
    rates_by_sex = {
        sex: lifetables.load_period_qx(data_dir / model.life_table, sex)
        for sex in ("male", "female")
    }
    horizons = (float(settings.horizon_years), float(settings.secondary_horizon_years))
    if not complete_case:
        imputed = pl.read_parquet(INTERIM_DIR / f"{name}_imputed.parquet")
        predictions, summary = predict.predict_frame(
            imputed,
            rates_by_sex,
            horizons,
            start_year=model.start_year,
            round_to_percent=model.round_other_cause_to_percent,
        )
        out_path = INTERIM_DIR / f"{name}_predictions.parquet"
        predictions.write_parquet(out_path)
        COHORT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = COHORT_DIR / f"{name}_prediction_summary.csv"
        summary.write_csv(summary_path)
        typer.echo(f"{predictions.height} rows -> {out_path}")
    cc_predictions, _ = predict.predict_frame(
        patients,
        rates_by_sex,
        horizons,
        start_year=model.start_year,
        round_to_percent=model.round_other_cause_to_percent,
    )
    cc_path = INTERIM_DIR / f"{name}_predictions_cc.parquet"
    cc_predictions.write_parquet(cc_path)
    typer.echo(f"{cc_predictions.height} complete-case rows -> {cc_path}")


@app.command("score")
def score_cmd(name: str = "transorce", profile: str = "main") -> None:
    """Score predictions against observed outcomes."""
    label = _cohort_label(name)
    settings = params.load_analysis(profile)
    reporting = params.load_reporting()
    model = params.load_model()
    data_dir = _data_dir()
    rates_by_sex = {
        sex: lifetables.load_period_qx(data_dir / model.life_table, sex)
        for sex in ("male", "female")
    }
    patients = _scoring_patients(name)
    predictions = pl.read_parquet(INTERIM_DIR / f"{name}_predictions.parquet")
    horizons = (float(settings.horizon_years), float(settings.secondary_horizon_years))
    outputs = score.score_cohort_imputed(
        patients,
        predictions,
        label,
        horizons,
        resamples=settings.bootstrap_resamples,
        seed=settings.seed,
        resample_imputations=settings.bootstrap_imputations,
        rates_by_sex=rates_by_sex,
        start_year=model.start_year,
        round_to_percent=model.round_other_cause_to_percent,
        n_bins=reporting.calibration_bins,
        rug_points=reporting.calibration_rug_points,
        min_at_risk=settings.min_at_risk,
        bandwidth=reporting.calibration_bandwidth,
        minimum_support=reporting.minimum_at_risk,
        age_cut_points=settings.age_cut_points,
        leibovich_groups=settings.leibovich_groups,
    )
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for frame, suffix in (
        (outputs.performance, "performance"),
        (outputs.contrasts, "contrasts"),
        (outputs.calibration, "calibration"),
        (outputs.other_cause, "other_cause"),
        (outputs.subgroups, "subgroups"),
    ):
        path = SCORE_DIR / f"{name}_{suffix}.csv"
        frame.write_csv(path)
        typer.echo(f"{frame.height} rows -> {path}")
    # Complete-case results stay separate: they are the sensitivity analysis.
    cc_predictions = pl.read_parquet(INTERIM_DIR / f"{name}_predictions_cc.parquet")
    cc_outputs = score.score_cohort(
        cc_predictions,
        label,
        horizons,
        resamples=settings.bootstrap_resamples,
        seed=settings.seed,
        n_bins=reporting.calibration_bins,
        rug_points=reporting.calibration_rug_points,
        min_at_risk=settings.min_at_risk,
        bandwidth=reporting.calibration_bandwidth,
        minimum_support=reporting.minimum_at_risk,
        age_cut_points=settings.age_cut_points,
    )
    for frame, suffix in (
        (cc_outputs.performance, "performance"),
        (cc_outputs.contrasts, "contrasts"),
        (cc_outputs.calibration, "calibration"),
    ):
        path = SCORE_DIR / f"{name}_{suffix}_cc.csv"
        frame.write_csv(path)
        typer.echo(f"{frame.height} rows -> {path}")


@app.command("sensitivity")
def sensitivity_cmd(
    kind: str = "definitions",
    name: str | None = None,
    profile: str = "main",
    out: Path | None = None,
) -> None:
    """Run one family of sensitivity analyses.

    --kind definitions runs the variants switched on in analysis.sensitivity
    for one cohort (--name transorce or vasudev) and writes
    results/sensitivity/{name}_definitions.csv unless `out` overrides it.
    --kind life-tables runs the country-matched WHO life table on the Vasudev cohort
    and writes `out` (results/sensitivity/life_tables.csv by default). Both
    feed Table S3.
    """
    if kind not in ("definitions", "life-tables"):
        raise typer.BadParameter(f"unknown sensitivity family {kind!r}")
    if kind == "definitions" and name is None:
        raise typer.BadParameter("--kind definitions needs --name")
    if kind == "life-tables" and name is not None:
        raise typer.BadParameter("--name is only used with --kind definitions")
    analysis = params.load_analysis(profile)
    analysis = replace(analysis, bootstrap_resamples=analysis.sensitivity_resamples)
    switches = params.load_sensitivity()
    model = params.load_model()
    data_dir = _data_dir()
    if kind == "definitions":
        assert name is not None  # checked above
        label = _cohort_label(name)
        reporting = params.load_reporting()
        rates_by_sex = {
            sex: lifetables.load_period_qx(data_dir / model.life_table, sex)
            for sex in ("male", "female")
        }
        ctx = sensitivity.RunContext(
            cohort=name,
            label=label,
            patients=_scoring_patients(name),
            surgery_clock_patients=_surgery_clock_patients(name),
            performance_path=SCORE_DIR / f"{name}_performance.csv",
            performance_cc_path=SCORE_DIR / f"{name}_performance_cc.csv",
            analysis=analysis,
            model=model,
            rates_by_sex=rates_by_sex,
            horizons=(float(analysis.horizon_years), float(analysis.secondary_horizon_years)),
            n_bins=reporting.calibration_bins,
            rug_points=reporting.calibration_rug_points,
            bandwidth=reporting.calibration_bandwidth,
            minimum_support=reporting.minimum_at_risk,
        )
        rows = sensitivity.definition_sensitivity(ctx, switches)
        path = out or SENSITIVITY_DIR / f"{name}_definitions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.write_csv(path)
        for variant in rows["variant"].unique(maintain_order=True):
            n_rows = rows.filter(pl.col("variant") == variant).height
            typer.echo(f"  {variant}: {n_rows} rows")
        typer.echo(f"{rows.height} rows -> {path}")
        return
    if not switches.who_country_life_table_vasudev:
        typer.echo("the country-matched WHO life-table analysis is off in params.yaml")
        raise typer.Exit(code=0)
    table = sensitivity.life_table_variants(
        _scoring_patients("vasudev"),
        pl.read_parquet(INTERIM_DIR / "vasudev_imputed.parquet"),
        switches=switches,
        data_dir=data_dir,
        model=model,
        horizons=(float(analysis.horizon_years), float(analysis.secondary_horizon_years)),
        vasudev_who_year=analysis.vasudev_who_year,
        resamples=analysis.bootstrap_resamples,
        seed=analysis.seed,
        resample_imputations=analysis.bootstrap_imputations,
    )
    path = out or SENSITIVITY_DIR / "life_tables.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
    typer.echo(f"{table.height} rows -> {path}")


@app.command("report")
def report_cmd(
    out_dir: Path = REPO_ROOT / "results" / "summary",
    score_dir: Path = SCORE_DIR,
) -> None:
    """Write the tidy summary CSVs that feed the figure and table stages."""
    model = params.load_model()
    analysis = params.load_analysis()
    reporting = params.load_reporting()
    # The behaviour figure shows men, matching the preprint's worked example. The sex
    # enters the predictions only through the life table.
    rates = lifetables.load_period_qx(
        _data_dir() / model.life_table, cast("lifetables.Sex", reporting.behaviour_sex)
    )
    table = report.model_behaviour(
        rates,
        ages=reporting.behaviour_ages,
        times=report.time_grid(reporting.behaviour_time_step),
        start_year=model.start_year,
        round_to_percent=model.round_other_cause_to_percent,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_behaviour.csv"
    table.write_csv(path)
    typer.echo(f"{table.height} rows -> {path}")
    for written in report.write_performance_summaries(
        score_dir, out_dir, float(analysis.horizon_years), reporting.minimum_group_n
    ):
        typer.echo(f"-> {written}")
    for written in report.write_other_cause_by_age(
        score_dir,
        out_dir,
        COHORT_LABELS,
        float(analysis.horizon_years),
        reporting.minimum_group_n,
    ):
        typer.echo(f"-> {written}")
    for written in report.write_cohort_summaries(
        INTERIM_DIR, COHORT_DIR, out_dir, COHORT_LABELS, analysis.leibovich_groups
    ):
        typer.echo(f"-> {written}")
    for written in report.write_subgroup_summaries(
        INTERIM_DIR,
        score_dir,
        out_dir,
        COHORT_LABELS,
        horizon=float(analysis.horizon_years),
        minimum_group_n=reporting.minimum_group_n,
        age_cut_points=analysis.age_cut_points,
        leibovich_groups=analysis.leibovich_groups,
    ):
        typer.echo(f"-> {written}")


@app.command("figure")
def figure_cmd(
    name: str,
    manuscript: Path = REPO_ROOT / "manuscript.yaml",
    summary_dir: Path = REPO_ROOT / "results" / "summary",
    out_dir: Path = REPO_ROOT / "results" / "figures",
) -> None:
    """Render one manuscript.yaml figure entry to a vector PDF and a 300 dpi PNG."""
    from leibovich_validation import figures

    reporting = params.load_reporting()
    paths = figures.build_figure(
        name, manuscript, summary_dir, out_dir, minimum_at_risk=reporting.minimum_at_risk
    )
    for path in paths:
        typer.echo(path)


def _principal_bound(summary: pl.DataFrame) -> pl.DataFrame:
    """Rows at the principal pT3 bound; observed rows carry no bound and are kept."""
    return summary.filter(pl.col("bound").fill_null("").is_in(["", predict.BOUNDS[0]]))


@app.command("table")
def table_cmd(
    out_dir: Path = REPO_ROOT / "results" / "tables",
    summary_dir: Path = REPO_ROOT / "results" / "summary",
    imputation_dir: Path = IMPUTE_DIR,
    sensitivity_dir: Path = SENSITIVITY_DIR,
) -> None:
    """Write the manuscript tables as CSV, with LaTeX for Tables 1 and 2."""
    analysis = params.load_analysis()
    audit_settings = params.load_audit()
    written = tables.write_table(tables.table_s1(params.load_model()), "table_s1", out_dir)
    written += tables.write_table(
        tables.table_s2(analysis, audit_settings, OTHER_DETAILS_MAPPING), "table_s2", out_dir
    )
    written += tables.write_table(
        tables.table_s2_other_details(OTHER_DETAILS_MAPPING),
        "table_s2_other_details",
        out_dir,
    )
    for summary_name, out_name, builder in (
        ("table1", "table_1", tables.table_1),
        ("table2", "table_2", tables.table_2),
    ):
        summary_path = summary_dir / f"{summary_name}.csv"
        if summary_path.is_file():
            source = pl.read_csv(summary_path)
            if out_name == "table_2":
                source = _principal_bound(source).filter(
                    pl.col("horizon") == float(analysis.horizon_years)
                )
            frame = builder(source)
            if out_name == "table_2":
                frame = frame.drop("bound")
            written += tables.write_table(frame, out_name, out_dir)
            latex_path = out_dir / f"{out_name}.tex"
            if out_name == "table_1":
                written.append(tables_latex.write_table_1(frame, latex_path))
            else:
                written.append(
                    tables_latex.write_table_2(frame, latex_path, float(analysis.horizon_years))
                )
    # The supplementary 3-year version of Table 2 comes from the same summary.
    table2_path = summary_dir / "table2.csv"
    if table2_path.is_file():
        secondary = _principal_bound(pl.read_csv(table2_path)).filter(
            pl.col("horizon") == float(analysis.secondary_horizon_years)
        )
        written += tables.write_table(
            tables.table_2(secondary).drop("bound"), "table_2_3y", out_dir
        )
    other_cause_path = summary_dir / "other_cause.csv"
    if other_cause_path.is_file():
        written += tables.write_table(
            tables.table_s6(pl.read_csv(other_cause_path)), "table_s6", out_dir
        )
    # Table S3 concatenates whatever results/sensitivity/ holds, so it works
    # before every sensitivity variant has run.
    sensitivity_files = sorted(sensitivity_dir.glob("*.csv"))
    if sensitivity_files:
        combined = pl.concat(
            [pl.read_csv(path) for path in sensitivity_files], how="diagonal_relaxed"
        )
        s3 = tables.table_s3(combined, float(analysis.horizon_years))
        written += tables.write_table(s3, "table_s3", out_dir, tables.table_s3_footnotes(s3))
    missingness_files = sorted(imputation_dir.glob("*_missingness.csv"))
    if missingness_files:
        missingness = pl.concat([pl.read_csv(path) for path in missingness_files])
        s4 = tables.table_s4(missingness, analysis, COHORT_LABELS)
        written += tables.write_table(s4, "table_s4", out_dir)
    for path in written:
        typer.echo(path)


@app.command("export")
def export_cmd(
    out_dir: Path = REPO_ROOT / "results" / "manuscript",
    manuscript: Path = REPO_ROOT / "manuscript.yaml",
    tables_dir: Path = REPO_ROOT / "results" / "tables",
    figures_dir: Path = REPO_ROOT / "results" / "figures",
) -> None:
    """Assemble the dated manuscript export from the items in manuscript.yaml."""
    target = export.export(out_dir, manuscript, tables_dir, figures_dir)
    typer.echo(f"export -> {target}")


@app.command("audit")
def audit_cmd(
    out: Path = REPO_ROOT / "results" / "audit",
    params_path: Path = REPO_ROOT / "params.yaml",
) -> None:
    """Run the blind audit and write aggregate CSVs to results/audit/."""
    settings = params.load_audit(params_path)
    written = audit.run(
        _data_dir(),
        out,
        operation_window=settings.operation_window,
        lag_days=settings.randomisation_lag_days,
        horizons=settings.followup_horizons_years,
    )
    typer.echo(f"{len(written)} files -> {out}")


@app.callback()
def main() -> None:
    """Pipeline commands for the Leibovich Plus validation."""
