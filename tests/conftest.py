"""Shared SECQUOIA test suite."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import importlib.util  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def find_module_name(filename: str) -> str:
    """Return the dotted module name for ``filename`` inside the package."""
    spec = importlib.util.find_spec("SECQUOIA")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "SECQUOIA is not importable. Run `pip install -e '.[testing]'`."
        )

    root = Path(next(iter(spec.submodule_search_locations)))
    matches = sorted(root.rglob(filename))
    if not matches:
        raise ModuleNotFoundError(
            f"No {filename} found anywhere under {root}."
        )

    relative = matches[0].relative_to(root).with_suffix("")
    return ".".join(("SECQUOIA", *relative.parts))


@pytest.fixture(autouse=True, scope="session")
def _isolate_system_temp_dir(tmp_path_factory):
    """Keep the suite out of the real system temp directory."""
    original = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("system_temp"))
    try:
        yield
    finally:
        tempfile.tempdir = original


@pytest.fixture(autouse=True, scope="session")
def _isolate_ttt_patterns(tmp_path_factory):
    """Keep tTt pattern tests off this machine's real ~/.SECQUOIA config."""
    import SECQUOIA.core.ttt_naming as ttt_naming
    import SECQUOIA.core.ttt_patterns as ttt_patterns

    real_config_path = ttt_patterns.CONFIG_PATH
    ttt_patterns.CONFIG_PATH = (
        tmp_path_factory.mktemp("secquoia_home") / "ttt_patterns.json"
    )
    ttt_naming.reload_patterns()
    try:
        yield
    finally:
        ttt_patterns.CONFIG_PATH = real_config_path
        ttt_naming.reload_patterns()


@pytest.fixture(autouse=True, scope="session")
def _stock_log_record_factory():
    """Log the way the application does, not the way napari's plugin does."""
    original = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    try:
        yield
    finally:
        logging.setLogRecordFactory(original)


def make_lineage_frame(
    ident: str = "pos1_id1",
    *,
    tracks: dict[int, tuple[int, int]] | None = None,
    n_channels: int = 2,
    with_label_ids: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small but realistic tracking dataframe.

    ``tracks`` maps ``TrackNumber -> (first frame, last frame)``. The default
    is a three-generation tree: track 1 divides at t=10 into 2 and 3, and
    track 2 divides again at t=20 into 4 and 5.
    """
    if tracks is None:
        tracks = {
            1: (0, 10),
            2: (10, 20),
            3: (10, 30),
            4: (20, 30),
            5: (20, 28),
        }

    rng = np.random.default_rng(seed)
    rows = []
    for track, (t0, t1) in tracks.items():
        for t in range(t0, t1 + 1):
            row = {
                "Identification": ident,
                "TrackNumber": track,
                "t": t,
                "Cellfate": None,
                "Calculated_Time": float(t) * 3.0,
                "AreaMorphologyM1": 100.0 + 5.0 * track + t,
                "XMorphologyM1": rng.uniform(0, 512),
                "YMorphologyM1": rng.uniform(0, 512),
            }
            for ch in range(1, n_channels + 1):
                row[f"MeanNoBgCorrectedCh{ch}M1"] = (
                    1000.0 * ch + 10.0 * t + track
                )
                row[f"MeanRawCh{ch}M1"] = 2000.0 * ch + t
            if with_label_ids:
                row["label_id_m1"] = track * 100 + t
            rows.append(row)

    df = pd.DataFrame(rows)
    # Give the last frame of track 5 a fate, as a real curated file would.
    df.loc[(df["TrackNumber"] == 5) & (df["t"] == 28), "Cellfate"] = "Dead"
    return df


@pytest.fixture
def lineage_df() -> pd.DataFrame:
    """The default three-generation lineage."""
    return make_lineage_frame()


@pytest.fixture
def lineage_df_with_gaps() -> pd.DataFrame:
    """A lineage where track 3 has no mask for frames 15-19."""
    df = make_lineage_frame()
    gap = (df["TrackNumber"] == 3) & df["t"].between(15, 19)
    df.loc[gap, "label_id_m1"] = 0
    return df


@pytest.fixture
def ident() -> str:
    """The Identification used by the default fixtures."""
    return "pos1_id1"


@pytest.fixture
def no_progress():
    """Progress callback that accepts both common callback styles."""

    def _callback(*args, **kwargs):
        return None

    return _callback


# Main window stand-ins
class FakeMainWindow(SimpleNamespace):
    """A ``main_window`` that is not a widget."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeMainWindow({sorted(vars(self))})"


@pytest.fixture
def fake_main_window():
    """Factory for a Qt-free ``main_window`` stand-in.

    ``fake_main_window(track_df=df, n_masks=2)`` returns an object with those
    attributes set.
    """
    return FakeMainWindow


@pytest.fixture
def fake_main_window_widget(qtbot):
    """Factory for a ``main_window`` stand-in that is a real ``QWidget``."""
    from qtpy.QtWidgets import QWidget

    class FakeMainWindowWidget(QWidget):
        def __init__(self, **attrs):
            super().__init__()
            for name, value in attrs.items():
                setattr(self, name, value)

    def _make(**attrs):
        widget = FakeMainWindowWidget(**attrs)
        qtbot.addWidget(widget)
        return widget

    return _make


def loaded_track_frame(n_frames: int = 4) -> pd.DataFrame:
    """A track dataframe with the column layout a real export has."""
    rows = []
    for t in range(n_frames):
        row = {
            "Position": 1,
            "Identification": "exp-p0001-001",
            "TrackNumber": 1,
            "t": t,
            "Calculated_Time": float(t) * 3.0,
            "Cellfate": "Healthy",
            "active": 1,
            "inspected": 0,
            "track_id": 1,
            "XMorphology": 10.0 + t,
            "YMorphology": 20.0 + t,
            "Outlier_detection": "OK",
        }
        for mask in (1, 2):
            row[f"label_id_m{mask}"] = 1
            row[f"AreaMorphologyM{mask}"] = 100.0 * mask + t
            for channel in ("00", "01"):
                row[f"MeanNoBgCorrectedCh{channel}M{mask}"] = 1000.0 * mask + t
                row[f"MeanRawCh{channel}M{mask}"] = 2000.0 * mask + t
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def silence_modals(monkeypatch):
    """Stop modal dialogs from blocking, and record that they were opened."""
    from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox

    shown: list[str] = []

    for cls in (QMessageBox, QDialog, QFileDialog):
        for name in ("exec_", "exec"):
            monkeypatch.setattr(
                cls,
                name,
                lambda self, _cls=cls: shown.append(_cls.__name__),
                raising=False,
            )

    for name in ("critical", "warning", "information", "question", "about"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            classmethod(lambda cls, *a, _n=name, **k: shown.append(_n)),
            raising=False,
        )

    return shown


@pytest.fixture
def dialog_main_window(fake_main_window_widget, tmp_path):
    """A ``QWidget`` main window populated the way a loaded session is."""

    experiment = tmp_path / "exp"
    (experiment / "exp_p0001").mkdir(parents=True)
    segmentation = [
        experiment / "Analysis" / "Segmentation1",
        experiment / "Analysis" / "Segmentation2",
    ]
    for path in segmentation:
        (path / "exp_p0001").mkdir(parents=True)

    df = loaded_track_frame()
    stack = np.zeros((4, 10, 10), dtype=np.uint16)

    return fake_main_window_widget(
        # data
        track_df=df,
        filtered_df=df,
        df_subset=df,
        labels=[stack.copy(), stack.copy()],
        masks=[stack.copy(), stack.copy()],
        images={
            "w00": np.zeros((4, 10, 10), dtype=np.float32),
            "w01": np.zeros((4, 10, 10), dtype=np.float32),
        },
        # layout
        n_masks=2,
        n_channels=2,
        ids_channels=["w00", "w01"],
        available_channels=["w00", "w01"],
        image_format="png",
        tracking_format="tTt",
        # selection
        unique_ids=["exp-p0001-001"],
        current_ident_index=0,
        ident="exp-p0001-001",
        current_time_index=0,
        current_TrackNumber_plot=1,
        # paths
        folder=str(experiment),
        folder_list=["exp_p0001"],
        experiment_name="exp",
        project_name="Project_1",
        segmentation_paths=[str(p) for p in segmentation],
        position_folders=[str(experiment / "exp_p0001")],
        position_selection=str(experiment / "exp_p0001"),
        # ranges
        current_position_index=0,
        current_position_number=1,
        position_min=1,
        position_max=1,
        time_min_selected=1,
        time_max_selected=4,
        dt_seconds=180,
        threshold=5,
        min_mask_size=1,
        # plot / feature state
        _derived_features={},
        _feature_defs={},
        last_run_config={},
        selected_feature_by_row={0: "AreaMorphology"},
        selected_m_by_channel={0: 1},
        selected_ch_by_channel={0: 1},
        user="tester",
    )


@dataclass
class FakeLineEdit:
    """Replacement for the Qt QLineEdit the GUI code reads."""

    value: str = ""

    def text(self) -> str:
        return self.value

    def clear(self) -> None:
        self.value = ""


class FakeBasicCorrection:
    """Minimal stand-in for the BaSiC state object, with correction off."""

    flag = False

    def add_channels_valid(self, *_args, **_kwargs) -> None:
        return None

    def add_exp_name(self, *_args, **_kwargs) -> None:
        return None

    def add_t_range(self, *_args, **_kwargs) -> None:
        return None

    def update_all(self, *_args, **_kwargs) -> None:
        return None


def make_fake_images() -> dict[str, np.ndarray]:
    """One fluorescence channel over two time points, shaped ``(t, y, x)``."""
    image = np.zeros((2, 10, 10), dtype=np.float32)
    image[0, 2:5, 2:5] = 10
    image[1, 2:5, 2:5] = 20
    return {"w01": image}


def make_fake_labels() -> list[np.ndarray]:
    """Two segmentation masks, both holding label 1 for the same tracked cell."""
    mask_1 = np.zeros((2, 10, 10), dtype=np.uint16)
    mask_2 = np.zeros((2, 10, 10), dtype=np.uint16)

    mask_1[0, 2:5, 2:5] = 1
    mask_1[1, 2:5, 2:5] = 1

    mask_2[0, 2:4, 2:4] = 1
    mask_2[1, 2:4, 2:4] = 1

    return [mask_1, mask_2]


def make_fake_tracking_df() -> pd.DataFrame:
    """The minimal tracking frame for one cell over two time points."""
    return pd.DataFrame(
        {
            "track_id": [1, 1],
            "XMorphology": [3.0, 3.0],
            "YMorphology": [3.0, 3.0],
            "t": [0, 1],
            "Position": [1, 1],
            "TrackNumber": [1, 1],
            "Cellfate": ["Healthy", "Healthy"],
            "Identification": ["fake-001", "fake-001"],
        }
    )


def make_fake_main_window(tmp_path, *, basic=None) -> SimpleNamespace:
    """The smallest ``main_window``-like object ``quantify`` accepts."""
    tracking_df = make_fake_tracking_df()
    images = make_fake_images()
    labels = make_fake_labels()

    experiment_folder = tmp_path / "experiment"
    position_folder = experiment_folder / "fake_p0001"
    segmentation_folder_1 = experiment_folder / "Analysis" / "Segmentation1"
    segmentation_folder_2 = experiment_folder / "Analysis" / "Segmentation2"
    project_dir = (
        experiment_folder / "Analysis" / "SECQUOIA_files_tTt" / "Project_1"
    )

    position_folder.mkdir(parents=True)
    segmentation_folder_1.mkdir(parents=True)
    segmentation_folder_2.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    tracking_csv = tmp_path / "fake_tracking.csv"
    tracking_df.to_csv(tracking_csv, index=False)

    return SimpleNamespace(
        folder=str(experiment_folder),
        experiment_name="fake_experiment",
        project_name="Project_1",
        project_dir=str(project_dir),
        tracking_format="tTt",
        tracking_path=str(tracking_csv),
        image_format="png",
        position_folders=[str(position_folder)],
        position_selection=str(position_folder),
        current_position_index=0,
        current_position_number=1,
        current_time_index=0,
        position_min=1,
        position_max=1,
        dt_seconds=60,
        n_channels=1,
        ids_channels=["w01"],
        available_channels=["w01"],
        FL_inputs=[FakeLineEdit("w01")],
        FL_identifiers_1="w01",
        n_masks=2,
        segmentation_paths=[
            str(segmentation_folder_1),
            str(segmentation_folder_2),
        ],
        threshold=5,
        min_mask_size=1,
        images=images,
        corrected_images=images,
        corrected_images_ratioflat={},
        corrected_images_noratioflat={},
        labels=labels,
        masks=labels,
        track_df=tracking_df.copy(),
        filtered_df=tracking_df.copy(),
        df_subset=tracking_df.copy(),
        basic=FakeBasicCorrection() if basic is None else basic,
        use_background_correction=False,
        background_correction_path=None,
        _derived_features={},
        _time_mode="t",
    )


def assert_column_values(
    df: pd.DataFrame,
    column: str,
    expected: list[float],
) -> None:
    """Assert that a numeric dataframe column matches expected values."""
    assert column in df.columns, (
        f"Expected column {column!r} was not created. "
        f"Available columns:\n{df.columns.tolist()}"
    )

    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    expected_array = np.asarray(expected, dtype=float)

    assert np.allclose(values, expected_array, equal_nan=False), (
        f"Column {column!r} has wrong values.\n"
        f"Expected: {expected}\n"
        f"Observed: {values.tolist()}\n"
        f"Full dataframe:\n{df}"
    )


@pytest.fixture
def quant_main_window():
    return make_fake_main_window


def combo_entries(widget, *, lower: bool = False) -> set[str]:
    """Every entry offered by every combo box under ``widget``."""
    from qtpy.QtWidgets import QComboBox

    return {
        combo.itemText(i).lower() if lower else combo.itemText(i)
        for combo in widget.findChildren(QComboBox)
        for i in range(combo.count())
    }
