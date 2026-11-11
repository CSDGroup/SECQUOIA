"""Tests for the headless ``SECQUOIA-batch`` command."""

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from SECQUOIA.cli import batch
from SECQUOIA.core.logging_setup import attach_experiment_log

DEMO_ZIP = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "data"
    / "250615MA40.zip"
)
TRACKING = "Analysis/Tracking_tTt/250615MA40"
# A small slice of the demo, to keep the end-to-end runs quick.
SUBSET = [
    "--channels",
    "w00",
    "w02",
    "--segmentation",
    "Segmentation_250615",
    "--t-max",
    "4",
]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1-3,5", {1, 2, 3, 5}),
        ("3-1", {1, 2, 3}),
        (" 2 , 4 ", {2, 4}),
    ],
)
def test_parse_position_spec(spec, expected):
    assert batch.parse_position_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "a", "1-x", ","])
def test_parse_position_spec_rejects_garbage(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        batch.parse_position_spec(spec)


def test_defaults_match_the_loading_window():
    args = batch.build_parser().parse_args(["exp", "--tracking-folder", "t"])
    assert args.tracking_format == "tTt"
    assert args.image_format == "png"
    assert args.project_name == "Project_1"
    assert args.min_cell_size == 5
    assert args.max_distance == 20
    assert args.interval == 300
    assert args.t_min == 1
    assert args.basic is None and args.real_time is None


@pytest.fixture(scope="module")
def demo_experiment(tmp_path_factory) -> Path:
    if not DEMO_ZIP.is_file():
        pytest.skip("demo dataset archive not available")
    root = tmp_path_factory.mktemp("demo")
    with zipfile.ZipFile(DEMO_ZIP) as zf:
        zf.extractall(root)
    return root / "250615MA40"


@pytest.fixture(autouse=True)
def _detach_experiment_log():
    yield
    attach_experiment_log(None)


def _run(experiment: Path, *extra: str) -> int:
    return batch.main(
        [str(experiment), "--tracking-folder", TRACKING, "-q", *extra]
    )


@pytest.mark.slow
def test_dry_run_writes_nothing(demo_experiment, capsys):
    assert _run(demo_experiment, "--dry-run", "--project-name", "Dry") == 0
    out = capsys.readouterr().out
    assert "M1=Segmentation_250615, M2=Segmentation_250703" in out
    assert "2 positions, 20 Tree-IDs" in out
    assert not (demo_experiment / "Analysis/SECQUOIA_files_tTt/Dry").exists()


@pytest.mark.slow
def test_batch_run_writes_a_reopenable_project(demo_experiment):
    assert _run(demo_experiment, "--project-name", "Batch", *SUBSET) == 0

    project = demo_experiment / "Analysis/SECQUOIA_files_tTt/Batch"
    full = pd.read_csv(project / "Updated_CSV_File.csv")
    assert sorted(full["Position"].unique()) == [1, 2]
    assert full["original_t"].between(1, 4).all()
    assert {"MeanNoBgCorrectedCh00M1", "MeanNoBgCorrectedCh02M1"} <= set(
        full.columns
    )
    assert "MeanNoBgCorrectedCh01M1" not in full.columns

    # Every per position CSV matches its rows in the combined file.
    for pos in (1, 2):
        per = pd.read_csv(
            project / f"SECQUOIA_p{pos:04d}_t00001-00004_m1_ch2.csv"
        )
        pd.testing.assert_frame_equal(
            per,
            full[full["Position"] == pos].reset_index(drop=True),
        )

    meta = json.loads((project / "project_metadata.json").read_text())
    assert meta["loading_format"] == "All"
    assert meta["ids_channels"] == ["w00", "w02"]
    assert meta["n_masks"] == 1
    assert (meta["time_min_selected"], meta["time_max_selected"]) == (1, 4)
    assert meta["threshold"] == 20
    assert Path(meta["track_df_csv_path"]) == project / "Updated_CSV_File.csv"


@pytest.mark.slow
def test_position_and_tree_id_selection(demo_experiment):
    assert (
        _run(
            demo_experiment,
            "--project-name",
            "Picked",
            "--positions",
            "2",
            "--tree-ids",
            "250615MA40-p0002-001",
            "250615MA40-p0002-002",
            *SUBSET,
        )
        == 0
    )
    project = demo_experiment / "Analysis/SECQUOIA_files_tTt/Picked"
    full = pd.read_csv(project / "Updated_CSV_File.csv")
    assert set(full["Identification"]) == {
        "250615MA40-p0002-001",
        "250615MA40-p0002-002",
    }
    assert not list(project.glob("SECQUOIA_p0001_*.csv"))


@pytest.mark.slow
def test_existing_project_needs_overwrite(demo_experiment, capsys):
    args = ("--project-name", "Again", *SUBSET)
    assert _run(demo_experiment, *args) == 0
    capsys.readouterr()

    assert _run(demo_experiment, *args) == batch.EXIT_USAGE
    assert "already exists" in capsys.readouterr().err

    assert _run(demo_experiment, *args, "--overwrite") == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--channels", "w09"], "Channel(s) not found: w09"),
        (["--segmentation", "Nope"], "Segmentation folder(s) not found"),
        (["--basic"], "No BaSiC folder found"),
        (["--t-max", "99"], "--t-max must be between"),
        (["--positions", "9"], "No tracking data left after the selection"),
    ],
)
def test_bad_arguments_are_usage_errors(
    demo_experiment, capsys, extra, message
):
    assert _run(demo_experiment, "--dry-run", *extra) == batch.EXIT_USAGE
    assert message in capsys.readouterr().err


def test_missing_experiment_folder(tmp_path, capsys):
    assert _run(tmp_path / "missing") == batch.EXIT_USAGE
    assert "Experiment folder not found" in capsys.readouterr().err
