import logging
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated, cast

import lv_datasets
import polars as pl
import typer
from tqdm.contrib.logging import logging_redirect_tqdm

from leibovich_validation import (
    atomic,
    audit,
    bootstrap,
    cohort,
    export,
    impute,
    interrupt,
    lifetables,
    manifest,
    params,
    predict,
    progress,
    rbridge,
    report,
    score,
    sensitivity,
    tables,
    tables_latex,
)
from leibovich_validation.config import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)

INTERIM_DIR = REPO_ROOT / "data" / "interim"
SCORE_DIR = REPO_ROOT / "results" / "scores"
SENSITIVITY_DIR = REPO_ROOT / "results" / "sensitivity"
COHORT_DIR = REPO_ROOT / "results" / "cohort"
IMPUTE_DIR = REPO_ROOT / "results" / "imputation"


def _data_dir() -> Path:
    data_dir = get_settings().data_dir
    if not data_dir.is_dir():
        raise typer.BadParameter(f"LV_DATA_DIR does not exist: {data_dir}")
    return data_dir


Name = Annotated[str, typer.Option(help="Registered dataset name.")]


def _dataset(name: str) -> lv_datasets.Dataset:
    try:
        return lv_datasets.get(name)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None


def _cohort_label(name: str) -> str:
    return _dataset(name).label


def _cohort_labels() -> dict[str, str]:
    """Display names by dataset name."""
    return {n: lv_datasets.get(n).label for n in lv_datasets.names()}


@app.command("datasets")
def datasets_cmd() -> None:
    """List the registered datasets."""
    for name in lv_datasets.names():
        typer.echo(f"{name}: {lv_datasets.get(name).label}")


@app.command("manifest")
def manifest_cmd(out: Path = REPO_ROOT / "data" / "manifest.json") -> None:
    """Write a checksum manifest of everything under LV_DATA_DIR."""
    data_dir = _data_dir()
    logger.info("manifest: hashing %s", data_dir)
    count = manifest.write(data_dir, out)
    logger.info("manifest: %d files -> %s", count, out)


def _surgery_clock_patients(name: str) -> pl.DataFrame:
    """The dataset's patients timed from surgery with delayed entry (surgery-clock analysis)."""
    return cohort.surgery_clock(pl.read_parquet(INTERIM_DIR / f"{name}_patients.parquet"))


def _scoring_patients(name: str) -> pl.DataFrame:
    """The dataset's patients on the randomisation clock, as the later stages use them."""
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
def cohort_cmd(name: Name) -> None:
    """Build the harmonised patient table and flow counts for one dataset."""
    dataset = _dataset(name)
    logger.info("cohort: dataset %s", name)
    frame, notes = dataset.harmonise(
        lv_datasets.Context(_data_dir() / name, params.load_dataset(name))
    )
    result = lv_datasets.finalise(dataset, frame, notes, params.load_analysis().score_agreement_min)
    patient_path = atomic.save(result.patients, INTERIM_DIR / f"{name}_patients.parquet")
    atomic.save(result.flow, COHORT_DIR / f"{name}_flow.csv")
    atomic.save(result.score_agreement, COHORT_DIR / f"{name}_score_agreement.csv")
    logger.info("cohort: %d patients -> %s", result.patients.height, patient_path)
    for step, n in result.flow.iter_rows():
        logger.info("cohort: flow %s: %d", step, n)


@app.command("impute")
def impute_cmd(name: Name, profile: str = "main") -> None:
    """Impute missing Leibovich predictors by MICE for one dataset."""
    _cohort_label(name)
    settings = params.load_analysis(profile)
    patients = _scoring_patients(name)
    logger.info(
        "impute: dataset %s, %d imputations, seed %d", name, settings.imputations, settings.seed
    )
    imputed = impute.impute_frame(patients, settings.imputations, settings.seed)
    imputed_path = atomic.save(imputed, INTERIM_DIR / f"{name}_imputed.parquet")
    summary_path = atomic.save(
        impute.missingness_summary(patients), IMPUTE_DIR / f"{name}_missingness.csv"
    )
    n_imputations = imputed["imputation"].n_unique()
    logger.info(
        "impute: %d rows in %d completed datasets -> %s",
        imputed.height,
        n_imputations,
        imputed_path,
    )
    logger.info("impute: missingness counts -> %s", summary_path)


@app.command("predict")
def predict_cmd(name: Name, complete_case: bool = False) -> None:
    """Write per-patient risks for each completed dataset, plus the complete-case file.

    The main file has one row per patient per imputation. With --complete-case only
    the `_cc` file is written, for runs without the imputed table.
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
    logger.info("predict: dataset %s, horizons %s", name, horizons)
    if not complete_case:
        imputed = pl.read_parquet(INTERIM_DIR / f"{name}_imputed.parquet")
        predictions, summary = predict.predict_frame(
            imputed,
            rates_by_sex,
            horizons,
            start_year=model.start_year,
            round_to_percent=model.round_other_cause_to_percent,
        )
        out_path = atomic.save(predictions, INTERIM_DIR / f"{name}_predictions.parquet")
        atomic.save(summary, COHORT_DIR / f"{name}_prediction_summary.csv")
        logger.info("predict: %d rows -> %s", predictions.height, out_path)
    cc_predictions, _ = predict.predict_frame(
        patients,
        rates_by_sex,
        horizons,
        start_year=model.start_year,
        round_to_percent=model.round_other_cause_to_percent,
    )
    cc_path = atomic.save(cc_predictions, INTERIM_DIR / f"{name}_predictions_cc.parquet")
    logger.info("predict: %d complete-case rows -> %s", cc_predictions.height, cc_path)


@app.command("score")
def score_cmd(name: Name, profile: str = "main") -> None:
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
    logger.info(
        "score: dataset %s, %d resamples, seed %d, workers %d, horizons %s",
        name,
        settings.bootstrap_resamples,
        settings.seed,
        bootstrap.default_workers(),
        horizons,
    )
    with progress.bar(total=2, desc=f"Score {name}", unit="analysis") as steps:
        steps.set_postfix(analysis="imputed", refresh=True)
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
        steps.update()
        steps.set_postfix(analysis="complete-case", refresh=True)
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
        steps.update()
    for frame, suffix in (
        (outputs.performance, "performance"),
        (outputs.contrasts, "contrasts"),
        (outputs.calibration, "calibration"),
        (outputs.other_cause, "other_cause"),
        (outputs.subgroups, "subgroups"),
        (cc_outputs.performance, "performance_cc"),
        (cc_outputs.contrasts, "contrasts_cc"),
        (cc_outputs.calibration, "calibration_cc"),
    ):
        path = atomic.save(frame, SCORE_DIR / f"{name}_{suffix}.csv")
        logger.info("score: %d rows -> %s", frame.height, path)


@app.command("sensitivity")
def sensitivity_cmd(
    kind: str = "definitions",
    name: str | None = None,
    profile: str = "main",
    out: Path | None = None,
) -> None:
    """Run one family of sensitivity analyses.

    --kind definitions runs the variants switched on in analysis.sensitivity for
    --name <dataset>, writing results/sensitivity/{name}_definitions.csv
    unless `out` is given. --kind life-tables runs the country-matched WHO life table
    on every dataset with a country column, writing `out` (default
    results/sensitivity/life_tables.csv).
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
        logger.info(
            "sensitivity: dataset %s, %d resamples, seed %d, workers %d",
            name,
            analysis.bootstrap_resamples,
            analysis.seed,
            bootstrap.default_workers(),
        )
        rows = sensitivity.definition_sensitivity(ctx, switches)
        for variant in rows["variant"].unique(maintain_order=True):
            n_rows = rows.filter(pl.col("variant") == variant).height
            logger.info("sensitivity: variant %s: %d rows", variant, n_rows)
        path = atomic.save(rows, out or SENSITIVITY_DIR / f"{name}_definitions.csv")
        logger.info("sensitivity: %d rows -> %s", rows.height, path)
        return
    if not switches.country_life_table:
        logger.info(
            "sensitivity: the country-matched WHO life-table analysis is off in params.yaml"
        )
        raise typer.Exit(code=0)
    logger.info(
        "sensitivity: country life tables, %d resamples, seed %d, workers %d",
        analysis.bootstrap_resamples,
        analysis.seed,
        bootstrap.default_workers(),
    )
    tables_by_dataset = []
    with progress.bar(lv_datasets.names(), desc="Life tables", unit="dataset") as datasets:
        for dataset in datasets:
            datasets.set_postfix(dataset=dataset, refresh=True)
            patients = _scoring_patients(dataset)
            if "country" not in patients.columns:
                continue
            tables_by_dataset.append(
                sensitivity.life_table_variants(
                    patients,
                    pl.read_parquet(INTERIM_DIR / f"{dataset}_imputed.parquet"),
                    switches=switches,
                    data_dir=data_dir,
                    model=model,
                    horizons=(
                        float(analysis.horizon_years),
                        float(analysis.secondary_horizon_years),
                    ),
                    fallback_surgery_year=analysis.fallback_surgery_year,
                    resamples=analysis.bootstrap_resamples,
                    seed=analysis.seed,
                    resample_imputations=analysis.bootstrap_imputations,
                )
            )
    table = (
        pl.concat(tables_by_dataset) if tables_by_dataset else sensitivity.empty_sensitivity_frame()
    )
    path = atomic.save(table, out or SENSITIVITY_DIR / "life_tables.csv")
    logger.info("sensitivity: %d rows -> %s", table.height, path)


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

    def write_model_behaviour() -> list[Path]:
        table = report.model_behaviour(
            rates,
            ages=reporting.behaviour_ages,
            times=report.time_grid(reporting.behaviour_time_step),
            start_year=model.start_year,
            round_to_percent=model.round_other_cause_to_percent,
        )
        return [atomic.save(table, out_dir / "model_behaviour.csv")]

    steps = {
        "model behaviour": write_model_behaviour,
        "performance summaries": lambda: report.write_performance_summaries(
            score_dir,
            out_dir,
            float(analysis.horizon_years),
            reporting.minimum_group_n,
            reporting.descriptive_below_deaths,
        ),
        "other cause by age": lambda: report.write_other_cause_by_age(
            score_dir,
            out_dir,
            _cohort_labels(),
            float(analysis.horizon_years),
            reporting.minimum_group_n,
        ),
        "cohort summaries": lambda: report.write_cohort_summaries(
            INTERIM_DIR, COHORT_DIR, out_dir, _cohort_labels(), analysis.leibovich_groups
        ),
        "subgroup summaries": lambda: report.write_subgroup_summaries(
            INTERIM_DIR,
            score_dir,
            out_dir,
            _cohort_labels(),
            horizon=float(analysis.horizon_years),
            minimum_group_n=reporting.minimum_group_n,
            age_cut_points=analysis.age_cut_points,
            leibovich_groups=analysis.leibovich_groups,
        ),
    }
    with progress.bar(steps.items(), desc="Report", unit="step") as bar:
        for step, write in bar:
            bar.set_postfix(step=step, refresh=True)
            for written in write():
                logger.info("report: %s -> %s", step, written)


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
        logger.info("figure: %s -> %s", name, path)


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
    other_cause_path = summary_dir / "other_cause.csv"
    other_cause = pl.read_csv(other_cause_path) if other_cause_path.is_file() else None
    descriptive = (
        set(other_cause.filter(pl.col("status") == "descriptive")["cohort"])
        if other_cause is not None
        else set()
    )
    written = tables.write_table(tables.table_s1(params.load_model()), "table_s1", out_dir)
    names = lv_datasets.names()
    written += tables.write_table(
        tables.table_s2(analysis, {name: params.load_dataset(name) for name in names}),
        "table_s2",
        out_dir,
    )
    for name, frame in tables.mapping_tables(names).items():
        written += tables.write_table(frame, name, out_dir)
    for summary_name, out_name, builder in (
        ("table1", "table_1", tables.table_1),
        ("table2", "table_2", lambda source: tables.table_2(source, descriptive)),
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
                    tables_latex.write_table_2(
                        frame, latex_path, float(analysis.horizon_years), _cohort_labels()
                    )
                )
    # The supplementary 3-year version of Table 2 comes from the same summary.
    table2_path = summary_dir / "table2.csv"
    if table2_path.is_file():
        secondary = _principal_bound(pl.read_csv(table2_path)).filter(
            pl.col("horizon") == float(analysis.secondary_horizon_years)
        )
        written += tables.write_table(
            tables.table_2(secondary, descriptive).drop("bound"), "table_2_3y", out_dir
        )
    if other_cause is not None:
        written += tables.write_table(tables.table_s6(other_cause), "table_s6", out_dir)
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
        s4 = tables.table_s4(missingness, analysis, _cohort_labels())
        written += tables.write_table(s4, "table_s4", out_dir)
    for path in written:
        logger.info("table: -> %s", path)


@app.command("export")
def export_cmd(
    out_dir: Path = REPO_ROOT / "results" / "manuscript",
    manuscript: Path = REPO_ROOT / "manuscript.yaml",
    tables_dir: Path = REPO_ROOT / "results" / "tables",
    figures_dir: Path = REPO_ROOT / "results" / "figures",
) -> None:
    """Assemble the dated manuscript export from the items in manuscript.yaml."""
    target = export.export(out_dir, manuscript, tables_dir, figures_dir)
    logger.info("export -> %s", target)


@app.command("audit")
def audit_cmd(
    out: Path = REPO_ROOT / "results" / "audit",
    params_path: Path = REPO_ROOT / "params.yaml",
) -> None:
    """Run the blind audit and write aggregate CSVs to results/audit/."""
    written = audit.run(
        _data_dir(),
        out,
        params_path,
        horizons=params.load_audit(params_path).followup_horizons_years,
    )
    logger.info("audit: %d files -> %s", len(written), out)


@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[
        str, typer.Option(envvar="LV_LOG_LEVEL", help="Log level for stderr.")
    ] = "INFO",
) -> None:
    """Pipeline commands for the Leibovich Plus validation."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("rpy2").setLevel(logging.WARNING)
    ctx.with_resource(logging_redirect_tqdm())


def run() -> None:
    """Console entry point: `app` with SIGINT and SIGTERM as a clean stop.

    The signal unwinds the running stage, which cancels its workers and leaves no
    partial output. The exit code is 128 plus the signal number.
    """
    started = time.perf_counter()
    interrupt.install()
    try:
        app()
    except interrupt.Interrupted as stop:
        logger.warning(
            "interrupted by %s after %.1f s; no output written for this run",
            signal.Signals(stop.signum).name,
            time.perf_counter() - started,
        )
        sys.exit(128 + stop.signum)
    except SystemExit as stop:
        if not stop.code:
            logger.info("done in %.1f s", time.perf_counter() - started)
        raise
