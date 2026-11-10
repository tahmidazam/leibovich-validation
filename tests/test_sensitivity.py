"""The five sensitivity analyses: definition variants, the WHO life table and the surgery clock.

The WHO worked examples use synthetic tables where the country changes the
rate, so the check is that each patient is scored against their own country.
The surgery-clock worked example is a delayed-entry Aalen-Johansen estimate
worked out by hand. The definition tests cover censoring the inconsistent
Vasudev cohort records and the copied score-stage rows.
"""

from dataclasses import fields
from pathlib import Path

import polars as pl
import pytest

from leibovich_validation import (
    cohort,
    leibovich,
    leibovich_plus,
    lifetables,
    params,
    predict,
    rbridge,
    sensitivity,
)
from tests.synthetic import constant_rates

TUMOUR = leibovich.Tumour("pT2", node_positive=False, size_ge_10cm=False, grade=3, necrosis=False)
START_YEAR = 2024


def _patients(cohort_name: str = "transorce", **overrides) -> pl.DataFrame:
    """A one-row harmonised table with complete predictors; overrides set any column."""
    row = {
        "patient_id": "p1",
        "cohort": cohort_name,
        "sex": "male",
        "arm": "A",
        "age_at_surgery": 60.0,
        "operation_year": 2008,
        "country": None,
        "entry_time": 0.0,
        "prediction_offset": 0.0,
        "time": 6.0,
        "event": "censored",
        "estimated_operation_date": False,
        "dfs_inconsistent": False,
        "t_stage": "pT2",
        "node_positive": False,
        "size_ge_10cm": False,
        "grade": 3,
        "necrosis": False,
    }
    row.update(overrides)
    return pl.DataFrame([row])


def _vasudev_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Eight censored Vasudev cohort patients, four per country, and their stacked imputed table."""
    patients = pl.concat(
        [
            _patients(
                "vasudev",
                patient_id=f"l{i}",
                operation_year=None,
                country="Czech Republic" if i < 4 else "Serbia",
            )
            for i in range(8)
        ]
    )
    imputed = patients.with_columns(pl.lit(1).alias("imputation")).select(
        "imputation", *patients.columns
    )
    return patients, imputed


def _context(
    tmp_path: Path, patients: pl.DataFrame, surgery_clock: pl.DataFrame | None = None
) -> sensitivity.RunContext:
    """A RunContext with a constant 2% life table; data paths need not exist."""
    rates = constant_rates(0.02)
    return sensitivity.RunContext(
        cohort=str(patients["cohort"][0]),
        label=str(patients["cohort"][0]),
        patients=patients,
        surgery_clock_patients=patients if surgery_clock is None else surgery_clock,
        performance_path=tmp_path / "performance.csv",
        performance_cc_path=tmp_path / "performance_cc.csv",
        analysis=params.AnalysisParams(
            horizon_years=5,
            secondary_horizon_years=3,
            score_agreement_min=0.9,
            seed=1,
            bootstrap_resamples=3,
            sensitivity_resamples=3,
            imputations=2,
            bootstrap_imputations=1,
            missing_sex="exclude",
            vasudev_who_year=2006,
            min_at_risk=50,
            age_cut_points=(60, 70),
            sensitivity={},
            leibovich_groups={"low": (0, 2)},
        ),
        model=params.ModelParams(
            life_table="lifetables/qx.xlsx",
            start_year=START_YEAR,
            round_other_cause_to_percent=False,
        ),
        rates_by_sex={"male": rates, "female": rates},
        horizons=(5.0, 3.0),
        n_bins=5,
        rug_points=25,
        bandwidth=0.05,
        minimum_support=20,
    )


def _switches(**on: bool) -> params.SensitivityParams:
    """A SensitivityParams with every switch off except the named ones."""
    return params.SensitivityParams(
        **{
            field.name: bool(on.get(field.name, False))
            for field in fields(params.SensitivityParams)
        }
    )


def _delayed_entry_patients() -> pl.DataFrame:
    """Two copies of four patients with delayed entry, on the surgery clock.

    Entry, exit and event, with surgery as time zero:
    A 0.1 to 1.0 metastasis, B 1.5 to 2.0 censored, C 0.1 to 3.0 metastasis,
    D 0.3 to 6.0 censored.
    """
    rows = [
        ("A", 0.1, 1.0, "metastasis"),
        ("B", 1.5, 2.0, "censored"),
        ("C", 0.1, 3.0, "metastasis"),
        ("D", 0.3, 6.0, "censored"),
    ]
    return pl.concat(
        [
            _patients(patient_id=f"{name}{copy}", entry_time=entry, time=time, event=event)
            for copy in (1, 2)
            for name, entry, time, event in rows
        ]
    )


def test_surgery_clock_rows_use_delayed_entry_incidence_and_unconditional_risks(tmp_path):
    # Worked example with two copies of each patient. B enters at 1.5, after the events at
    # 1.0, so the risk set at 1.0 is A, C and D (six patients, two events): S = 1 - 2/6 = 2/3.
    # At 3.0 the risk set is C and D (B left at 2.0; four patients, two events):
    # S = 2/3 * (1 - 2/4) = 1/3. The delayed-entry incidence is 1 - 1/3 = 2/3. Ignoring entry
    # would count B at risk at 1.0 and give 1 - 3/4 * 1/2 = 0.625.
    patients = _delayed_entry_patients()
    ctx = _context(tmp_path, patients)
    rows = sensitivity.definition_sensitivity(ctx, _switches(transorce_surgery_clock=True))

    assert set(rows["variant"]) == {"transorce_surgery_clock"}
    assert set(rows["metric"]) == {"aj_incidence", "oe_ratio"}
    five = rows.filter(pl.col("horizon_years") == 5.0)
    observed = five.filter(pl.col("metric") == "aj_incidence")
    assert observed["estimate"].to_list() == pytest.approx([2 / 3])
    assert observed["n"].to_list() == [8]
    assert observed["events"].to_list() == [4]

    # The predictions run from surgery: zero offset, so the unconditional risks over (0, 5].
    rates = constant_rates(0.02)
    expected = {
        predict.LEIBOVICH: leibovich.risk(TUMOUR, 5.0),
        predict.LEIBOVICH_PLUS: leibovich_plus.published_risks(
            rates, TUMOUR, 60, 5.0, START_YEAR, False
        )[0],
    }
    oe = five.filter((pl.col("metric") == "oe_ratio") & (pl.col("bound") == "pt3a"))
    for model, risk in expected.items():
        assert oe.filter(pl.col("model") == model)["estimate"].to_list() == pytest.approx(
            [(2 / 3) / risk]
        )


def test_surgery_clock_reports_no_score_measures(tmp_path):
    ctx = _context(tmp_path, _delayed_entry_patients())
    rows = sensitivity.definition_sensitivity(ctx, _switches(transorce_surgery_clock=True))
    assert not set(rows["metric"]) & {"calibration_slope", "auc", "brier"}
    assert rows.columns == sensitivity.SENSITIVITY_COLUMNS


def test_surgery_clock_is_transorce_only(tmp_path):
    ctx = _context(tmp_path, _patients("vasudev"))
    rows = sensitivity.definition_sensitivity(ctx, _switches(transorce_surgery_clock=True))
    assert rows.height == 0


def test_surgery_clock_table_keeps_the_clock_and_zeroes_the_offset():
    patients = _patients(entry_time=0.2, time=5.0, prediction_offset=None)
    out = cohort.surgery_clock(patients)
    assert out["time"].to_list() == [5.0]
    assert out["entry_time"].to_list() == [0.2]
    assert out["prediction_offset"].to_list() == [0.0]
    with pytest.raises(ValueError, match="after entry"):
        cohort.surgery_clock(_patients(entry_time=5.0, time=5.0))


def test_who_variant_uses_each_patients_country_table(tmp_path, monkeypatch):
    # A fixed observed incidence of 20% against the pooled mean predicted risk. Half the
    # patients live in each country, so the mean is the average of the two countries' risks.
    monkeypatch.setattr(
        rbridge,
        "observed_incidence",
        lambda outcomes, horizon, cause="metastasis": rbridge.AalenJohansen(0.2, 0.05),
    )
    low = constant_rates(0.01, first_year=2000)
    high = constant_rates(0.20, first_year=2000)
    tables = {
        "Czech Republic": {"male": low, "female": low},
        "Serbia": {"male": high, "female": high},
    }
    monkeypatch.setattr(
        lifetables, "load_who_rates_by_sex", lambda data_dir, country: tables[country]
    )
    patients, imputed = _vasudev_frames()
    table = sensitivity.life_table_variants(
        patients,
        imputed,
        switches=_switches(who_country_life_table_vasudev=True),
        data_dir=tmp_path,
        model=params.ModelParams("lifetables/qx.xlsx", START_YEAR, False),
        horizons=(5.0,),
        vasudev_who_year=2006,
        resamples=3,
        seed=1,
        resample_imputations=1,
    )
    assert table.columns == sensitivity.SENSITIVITY_COLUMNS
    assert set(table["variant"]) == {"who_country_life_table_vasudev"}
    assert set(table["metric"]) == {"aj_incidence", "oe_ratio", "oe_other_cause"}

    plus = [
        leibovich_plus.published_risks(rates, TUMOUR, 60, 5.0, 2006, False)[0]
        for rates in (low, high)
    ]
    expected = {
        predict.LEIBOVICH: 0.2 / leibovich.risk(TUMOUR, 5.0),
        predict.LEIBOVICH_PLUS: 0.2 / (sum(plus) / 2),
    }
    oe = table.filter((pl.col("metric") == "oe_ratio") & (pl.col("bound") == "pt3a"))
    for model, value in expected.items():
        assert oe.filter(pl.col("model") == model)["estimate"].to_list() == pytest.approx([value])
    # The two countries' risks differ, so using one table for everyone would change the answer.
    assert plus[0] != pytest.approx(plus[1])


def _who_other_cause_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Eight Vasudev cohort patients, four per country; one death without the event in each country.

    The deaths fall at 2.0, with every other patient censored at 6.0, so nobody leaves the
    risk set before the deaths and the incidence of death without the event at 5 years is 2/8.
    """
    patients = pl.concat(
        [
            _patients(
                "vasudev",
                patient_id=f"l{i}",
                operation_year=None,
                country="Czech Republic" if i < 4 else "Serbia",
                time=2.0 if i in (0, 4) else 6.0,
                event="competing" if i in (0, 4) else "censored",
            )
            for i in range(8)
        ]
    )
    imputed = patients.with_columns(pl.lit(1).alias("imputation")).select(
        "imputation", *patients.columns
    )
    return patients, imputed


def _who_variant(tmp_path, monkeypatch, patients, imputed, low, high) -> pl.DataFrame:
    tables = {
        "Czech Republic": {"male": low, "female": low},
        "Serbia": {"male": high, "female": high},
    }
    monkeypatch.setattr(
        lifetables, "load_who_rates_by_sex", lambda data_dir, country: tables[country]
    )
    return sensitivity.life_table_variants(
        patients,
        imputed,
        switches=_switches(who_country_life_table_vasudev=True),
        data_dir=tmp_path,
        model=params.ModelParams("lifetables/qx.xlsx", START_YEAR, False),
        horizons=(5.0,),
        vasudev_who_year=2006,
        resamples=3,
        seed=1,
        resample_imputations=1,
    )


def test_who_variant_other_cause_oe_is_incidence_over_mean_country_risk(tmp_path, monkeypatch):
    # Observed incidence of death without the event: 2 deaths among 8 at risk, so 0.25.
    # Expected: half the patients face each country's other-cause risk over 5 years.
    low = constant_rates(0.01, first_year=2000)
    high = constant_rates(0.20, first_year=2000)
    patients, imputed = _who_other_cause_frames()
    table = _who_variant(tmp_path, monkeypatch, patients, imputed, low, high)

    other = [
        leibovich_plus.published_risks(rates, TUMOUR, 60, 5.0, 2006, False)[1]
        for rates in (low, high)
    ]
    assert other[0] != pytest.approx(other[1])
    row = table.filter(pl.col("metric") == "oe_other_cause")
    assert row.height == 1
    assert row["model"].to_list() == [predict.LEIBOVICH_PLUS]
    assert row["bound"].to_list() == ["pt3a"]
    assert row["estimate"].to_list() == pytest.approx([0.25 / (sum(other) / 2)])
    assert row["events"].to_list() == [2]
    assert row["n"].to_list() == [8]
    assert row["lower"][0] is not None
    assert row["upper"][0] is not None


def test_who_variant_reports_no_score_measures(tmp_path, monkeypatch):
    low = constant_rates(0.01, first_year=2000)
    patients, imputed = _who_other_cause_frames()
    table = _who_variant(tmp_path, monkeypatch, patients, imputed, low, low)
    assert set(table["metric"]) == {"aj_incidence", "oe_ratio", "oe_other_cause"}
    assert not set(table["metric"]) & {"calibration_slope", "auc", "brier"}


def test_who_variant_writes_zero_oe_when_no_events(tmp_path, monkeypatch):
    table = constant_rates(0.01, first_year=2000)
    monkeypatch.setattr(
        lifetables,
        "load_who_rates_by_sex",
        lambda data_dir, country: {"male": table, "female": table},
    )
    patients, imputed = _vasudev_frames()
    rows = sensitivity.life_table_variants(
        patients,
        imputed,
        switches=_switches(who_country_life_table_vasudev=True),
        data_dir=tmp_path,
        model=params.ModelParams("lifetables/qx.xlsx", START_YEAR, False),
        horizons=(5.0, 3.0),
        vasudev_who_year=2006,
        resamples=3,
        seed=1,
        resample_imputations=1,
    )
    # Two horizons, each with an observed row and two models times two bounds of O/E rows.
    assert rows.height == 2 * (1 + 1 + len(predict.MODELS) * len(predict.BOUNDS))
    assert (rows["estimate"] == 0).all()
    assert (rows["n"] == 8).all()


def test_who_variant_needs_every_country_table(tmp_path, monkeypatch):
    def missing(data_dir, country):
        raise FileNotFoundError(country)

    monkeypatch.setattr(lifetables, "load_who_rates_by_sex", missing)
    patients, imputed = _vasudev_frames()
    with pytest.raises(FileNotFoundError):
        sensitivity.life_table_variants(
            patients,
            imputed,
            switches=_switches(who_country_life_table_vasudev=True),
            data_dir=tmp_path,
            model=params.ModelParams("lifetables/qx.xlsx", START_YEAR, False),
            horizons=(5.0,),
            vasudev_who_year=2006,
            resamples=1,
            seed=1,
            resample_imputations=1,
        )


def test_disabled_who_variant_writes_the_empty_schema(tmp_path):
    patients, imputed = _vasudev_frames()
    table = sensitivity.life_table_variants(
        patients,
        imputed,
        switches=_switches(),
        data_dir=tmp_path,
        model=params.ModelParams("lifetables/qx.xlsx", START_YEAR, False),
        horizons=(5.0,),
        vasudev_who_year=2006,
        resamples=1,
        seed=1,
        resample_imputations=1,
    )
    assert table.height == 0
    assert table.columns == sensitivity.SENSITIVITY_COLUMNS


def test_censor_inconsistent_flags_become_censored_at_the_same_time():
    patients = _patients("vasudev", dfs_inconsistent=True, event="metastasis", time=1.7).vstack(
        _patients("vasudev", patient_id="2", event="metastasis", time=2.0)
    )
    out = sensitivity.censor_inconsistent(patients)
    by_id = {row["patient_id"]: row for row in out.iter_rows(named=True)}
    assert by_id["p1"]["event"] == "censored"
    assert by_id["p1"]["time"] == 1.7
    assert by_id["2"]["event"] == "metastasis"


def test_censoring_variant_scores_the_censored_table_for_vasudev_only(tmp_path, monkeypatch):
    seen: list[pl.DataFrame] = []

    def fake_score(name: str, patients: pl.DataFrame, ctx: sensitivity.RunContext):
        seen.append(patients)
        return sensitivity._tagged(name, sensitivity.empty_sensitivity_frame().drop("variant"))

    monkeypatch.setattr(sensitivity, "_score_variant_patients", fake_score)
    vasudev = _patients("vasudev", dfs_inconsistent=True, event="metastasis", time=1.7)
    sensitivity.definition_sensitivity(
        _context(tmp_path, vasudev), _switches(exclude_inconsistent_vasudev=True)
    )
    assert len(seen) == 1
    assert seen[0]["event"].to_list() == ["censored"]
    sensitivity.definition_sensitivity(
        _context(tmp_path, _patients("transorce")), _switches(exclude_inconsistent_vasudev=True)
    )
    assert len(seen) == 1


def test_definition_sensitivity_with_every_switch_off_writes_the_schema(tmp_path):
    ctx = _context(tmp_path, _patients())
    rows = sensitivity.definition_sensitivity(ctx, _switches())
    assert rows.columns == sensitivity.SENSITIVITY_COLUMNS
    assert rows.height == 0


def test_definition_sensitivity_tolerates_missing_score_files(tmp_path):
    ctx = _context(tmp_path, _patients())
    # The complete-cases variant reuses a score file that is not there yet:
    # no rows, no crash, before the score stage has run.
    rows = sensitivity.definition_sensitivity(ctx, _switches(complete_cases=True))
    assert rows.height == 0
    assert rows.columns == sensitivity.SENSITIVITY_COLUMNS


def test_copied_rows_tags_and_bounds(tmp_path):
    ctx = _context(tmp_path, _patients())
    performance = pl.DataFrame(
        {
            "cohort": ["transorce"] * 2,
            "horizon_years": [5.0, 5.0],
            "bound": ["pt3a", "pt3b"],
            "model": ["Leibovich", "Leibovich"],
            "metric": ["oe_ratio", "oe_ratio"],
            "estimate": [0.9, 1.1],
            "lower": [0.7, 0.9],
            "upper": [1.1, 1.3],
            "method": ["bootstrap", "bootstrap"],
            "n": [439, 439],
            "events": [112, 112],
        }
    )
    performance.write_csv(tmp_path / "performance.csv")
    rows = sensitivity.definition_sensitivity(ctx, _switches(pt3b_for_unsplit_pt3=True))
    assert rows.height == 1
    row = rows.row(0, named=True)
    assert row["variant"] == "pt3b_for_unsplit_pt3"
    assert row["bound"] == "pt3b"
    assert row["estimate"] == 1.1


def test_the_register_holds_exactly_the_five_variants():
    assert set(sensitivity.VARIANT_LABELS) == {
        "pt3b_for_unsplit_pt3",
        "complete_cases",
        "who_country_life_table_vasudev",
        "transorce_surgery_clock",
        "exclude_inconsistent_vasudev",
    }
    assert {field.name for field in fields(params.SensitivityParams)} == set(
        sensitivity.VARIANT_LABELS
    )
