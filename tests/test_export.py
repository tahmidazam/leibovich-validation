"""The manuscript export: a dated folder of declared tables and figures plus an index."""

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from leibovich_validation import cli, export

MANUSCRIPT = {
    "style": {},
    "figure": {
        "calibration": {
            "label": "Figure 2",
            "placement": "main",
            "size": [183, 90],
            "mosaic": [["A"]],
            "panels": {"A": {"draw": "calibration", "table": "calibration_5y"}},
            "caption": "Calibration of predicted 5-year risk.",
        },
        "model_behaviour": {
            "label": "Figure S1",
            "placement": "supplement",
            "size": [183, 75],
            "mosaic": [["A"]],
            "panels": {"A": {"draw": "curve", "table": "model_behaviour"}},
            "caption": "Predicted risk over follow-up.",
        },
    },
    "table": {
        "table_2": {
            "label": "Table 2",
            "placement": "main",
            "latex": True,
            "caption": "5-year performance.",
        },
        "table_s1": {
            "label": "Table S1",
            "placement": "supplement",
            "caption": "Full model specification.",
        },
    },
}


def _results_tree(root: Path) -> tuple[Path, Path, Path]:
    """A synthetic results tree holding every file the synthetic manuscript declares."""
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    for stem in ("table_2", "table_s1"):
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / f"{stem}.csv").write_text("a,b\n1,2\n")
    (tables_dir / "table_2.tex").write_text(r"\begin{longtable}{ll}" + "\n")
    for placement, name in (("main", "calibration"), ("supplement", "model_behaviour")):
        (figures_dir / placement).mkdir(parents=True, exist_ok=True)
        (figures_dir / placement / f"{name}.pdf").write_bytes(b"%PDF-1.4")
        (figures_dir / placement / f"{name}.png").write_bytes(b"PNG")
    manuscript = root / "manuscript.yaml"
    manuscript.write_text(yaml.safe_dump(MANUSCRIPT))
    return manuscript, tables_dir, figures_dir


def test_export_assembles_a_dated_folder_with_an_index(tmp_path):
    manuscript, tables_dir, figures_dir = _results_tree(tmp_path)
    out = tmp_path / "results" / "manuscript"
    dated = export.export(out, manuscript, tables_dir, figures_dir, today=dt.date(2026, 3, 4))
    assert dated == out / "2026-03-04"
    for rel in (
        "main/table_2.csv",
        "main/table_2.tex",
        "main/calibration.pdf",
        "main/calibration.png",
        "supplement/table_s1.csv",
        "supplement/model_behaviour.pdf",
        "supplement/model_behaviour.png",
        "index.csv",
    ):
        assert (dated / rel).is_file(), rel
    index = pl.read_csv(dated / "index.csv")
    assert set(index["item"]) == {"Table 2", "Table S1", "Figure 2", "Figure S1"}
    row = index.filter(pl.col("item") == "Figure 2").row(0, named=True)
    assert row["placement"] == "main"
    assert row["caption"] == "Calibration of predicted 5-year risk."
    assert "main/calibration.pdf" in row["files"]
    assert "main/calibration.png" in row["files"]
    table_row = index.filter(pl.col("item") == "Table 2").row(0, named=True)
    assert "main/table_2.tex" in table_row["files"]


def test_export_fails_when_a_declared_file_is_missing(tmp_path):
    manuscript, tables_dir, figures_dir = _results_tree(tmp_path)
    (tables_dir / "table_s1.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Table S1"):
        export.export(tmp_path / "out", manuscript, tables_dir, figures_dir)


def test_export_rebuilds_so_one_dated_folder_remains(tmp_path):
    manuscript, tables_dir, figures_dir = _results_tree(tmp_path)
    out = tmp_path / "results" / "manuscript"
    export.export(out, manuscript, tables_dir, figures_dir, today=dt.date(2026, 3, 4))
    export.export(out, manuscript, tables_dir, figures_dir, today=dt.date(2026, 3, 5))
    assert [path.name for path in out.iterdir()] == ["2026-03-05"]


def test_export_command_writes_the_dated_folder(tmp_path):
    manuscript, tables_dir, figures_dir = _results_tree(tmp_path)
    out = tmp_path / "export"
    result = CliRunner().invoke(
        cli.app,
        [
            "export",
            "--out-dir",
            str(out),
            "--manuscript",
            str(manuscript),
            "--tables-dir",
            str(tables_dir),
            "--figures-dir",
            str(figures_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / dt.date.today().isoformat() / "index.csv").is_file()
