"""Tests for SECQUOIA CLT files.

CLT is the format tracking data is exchanged in, so these tests write a
track dataframe out, read it back, and check the two describe the same
tracks. The data is one lineage over five frames that divides at frame 3,
run once plain and once with TAT metadata so the coordinate conversion is
covered as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.core.tracking.clt_io import CLTParser, _as_scalar


@pytest.fixture
def parser() -> CLTParser:
    """A parser with no TAT metadata, so coordinates stay in pixels."""
    return CLTParser()


def track_frame(ident: str = "250615MA40-p0002-001") -> pd.DataFrame:
    """One dividing lineage: track 1 over t=0..2, tracks 2 and 3 over t=3..4."""
    rows = [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
    ]
    return pd.DataFrame(
        {
            "Identification": [ident] * len(rows),
            "TrackNumber": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "Position": [2] * len(rows),
            "XMorphology": [100.0 + 10 * r[1] for r in rows],
            "YMorphology": [200.0 + 5 * r[1] for r in rows],
            "Cellfate": ["Healthy"] * len(rows),
        }
    )


def write_clt(tmp_path, parser: CLTParser, df: pd.DataFrame, ident: str):
    """Write ``df`` as a .clt file named after ``ident`` and return the folder."""
    folder = tmp_path / "experiment"
    folder.mkdir(exist_ok=True)
    (folder / f"{ident}.clt").write_text(parser._new_clt_text(df, "tester"))
    return str(folder)


# Small helpers
class TestAsScalar:
    def test_passes_a_plain_value_through(self):
        assert _as_scalar(5) == 5

    def test_takes_the_first_element_of_a_series(self):
        assert _as_scalar(pd.Series([7, 8, 9])) == 7

    def test_takes_the_first_element_of_an_array(self):
        assert _as_scalar(np.array([[1, 2], [3, 4]])) == 1

    def test_falls_back_to_the_default_for_an_empty_series(self):
        assert _as_scalar(pd.Series(dtype=float), default=-1) == -1

    def test_falls_back_to_the_default_for_an_empty_array(self):
        assert _as_scalar(np.array([]), default=-1) == -1


class TestIdentificationFromPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/e/250615MA40-p0002-001.clt", "250615MA40-p0002-001"),
            ("/e/250615MA40-p0002-001a.clt", "250615MA40-p0002-001"),
            ("exp-p0001-7xyz.clt", "exp-p0001-7"),
        ],
    )
    def test_strips_the_extension_and_trailing_letters(self, path, expected):
        assert CLTParser._identification_from_path(path) == expected


class TestTrackIdFromIdent:
    @pytest.mark.parametrize(
        ("ident", "expected"),
        [
            ("250615MA40-p0002-001", 1),
            ("250615MA40-p0002-042", 42),
            ("exp-p0001-7", 7),
        ],
    )
    def test_reads_the_trailing_number(self, ident, expected):
        assert CLTParser._track_id_from_ident(ident) == expected

    def test_defaults_to_one_without_a_number(self):
        assert CLTParser._track_id_from_ident("no-number-here") == 1


class TestNormalizeCellFate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", "Healthy"),
            ("   ", "Healthy"),
            ("Division", "Healthy"),
            ("Death", "Dead"),
            ("Lost", "Lost"),
        ],
    )
    def test_maps_file_vocabulary_to_app_vocabulary(self, raw, expected):
        assert CLTParser._normalize_cell_fate(raw) == expected


class TestRemoveSecquoiaQuantBlocks:
    def test_removes_a_block_this_app_wrote(self, parser):
        text = (
            '%% TrackingData Units="LocalPixel"\n'
            "\t% Data\n"
            '%% Quantification Software="SECQUOIA"\n'
            "\tsome rows\n"
        )

        out = parser._remove_secquoia_quant_blocks(text)

        assert "Quantification" not in out
        assert "TrackingData" in out

    def test_keeps_a_block_another_tool_wrote(self, parser):
        text = (
            '%% TrackingData Units="LocalPixel"\n'
            '%% Quantification Software="SomethingElse"\n'
            "\tsome rows\n"
        )

        out = parser._remove_secquoia_quant_blocks(text)

        assert "SomethingElse" in out

    def test_leaves_a_file_without_quantification_alone(self, parser):
        text = '%% TrackingData Units="LocalPixel"\n'

        assert parser._remove_secquoia_quant_blocks(text) == text


# Writing
class TestBuildTrackingDataBlock:
    def test_writes_one_cell_block_per_tracknumber(self, parser):
        block = parser._build_trackingdata_block_from_df(track_frame())

        assert block.count("% Cell id=") == 3

    def test_writes_time_points_one_based(self, parser):
        """Frame 0 in the dataframe is Time point 1 in the file."""
        df = track_frame()
        block = parser._build_trackingdata_block_from_df(df)

        rows = [ln.strip() for ln in block.splitlines() if ";" in ln]
        assert rows[0].startswith("1;")

    def test_writes_the_declared_column_header(self, parser):
        block = parser._build_trackingdata_block_from_df(track_frame())

        assert "TimePoint" in block
        assert "double x" in block
        assert "double y" in block

    def test_records_a_death_on_the_cell_block(self, parser):
        df = track_frame()
        df.loc[(df["TrackNumber"] == 3) & (df["t"] == 4), "Cellfate"] = "Dead"

        block = parser._build_trackingdata_block_from_df(df)

        assert 'cellFate="Death"' in block

    def test_truncates_a_track_after_its_death(self, parser):
        """Nothing is written past the frame the cell died in."""
        df = track_frame()
        df.loc[(df["TrackNumber"] == 2) & (df["t"] == 3), "Cellfate"] = "Dead"

        block = parser._build_trackingdata_block_from_df(df)

        cell_2 = block.split("% Cell id=2")[1].split("% Cell")[0]
        rows = [ln for ln in cell_2.splitlines() if ";" in ln]
        assert len(rows) == 1  # only t=3, not t=4

    def test_skips_rows_without_a_track_or_time(self, parser):
        df = track_frame()
        df.loc[0, "TrackNumber"] = np.nan
        df.loc[1, "t"] = np.nan

        block = parser._build_trackingdata_block_from_df(df)

        rows = [ln for ln in block.splitlines() if ";" in ln]
        assert len(rows) == 5

    def test_an_empty_frame_produces_no_block(self, parser):
        empty = track_frame().iloc[0:0]

        assert parser._build_trackingdata_block_from_df(empty) == ""


class TestNewCltText:
    def test_writes_a_parseable_header(self, parser):
        text = parser._new_clt_text(track_frame(), "tester")

        assert text.startswith("%% CellLineageTree")
        assert "%% FrameIndex" in text

    def test_includes_the_tracking_block(self, parser):
        text = parser._new_clt_text(track_frame(), "tester")

        assert "%% TrackingData" in text
        assert "% Cell id=1" in text


# Round trip
class TestRoundTrip:
    IDENT = "250615MA40-p0002-001"

    def test_recovers_every_track_and_time_point(self, tmp_path, parser):
        original = track_frame(self.IDENT)
        folder = write_clt(tmp_path, parser, original, self.IDENT)

        out = parser.load_folder(folder, folder, cp_tracking=False)

        recovered = out.groupby("TrackNumber")["t"].apply(sorted).to_dict()
        assert recovered == {1: [0, 1, 2], 2: [3, 4], 3: [3, 4]}

    def test_recovers_the_coordinates(self, tmp_path, parser):
        original = track_frame(self.IDENT)
        folder = write_clt(tmp_path, parser, original, self.IDENT)

        out = parser.load_folder(folder, folder, cp_tracking=False)

        merged = original.merge(
            out, on=["TrackNumber", "t"], suffixes=("_in", "_out")
        )
        assert merged["XMorphology_in"].tolist() == pytest.approx(
            merged["XMorphology_out"].tolist()
        )
        assert merged["YMorphology_in"].tolist() == pytest.approx(
            merged["YMorphology_out"].tolist()
        )

    def test_recovers_the_position(self, tmp_path, parser):
        folder = write_clt(
            tmp_path, parser, track_frame(self.IDENT), self.IDENT
        )

        out = parser.load_folder(folder, folder, cp_tracking=False)

        assert out["Position"].unique().tolist() == [2]

    def test_recovers_the_identification(self, tmp_path, parser):
        folder = write_clt(
            tmp_path, parser, track_frame(self.IDENT), self.IDENT
        )

        out = parser.load_folder(folder, folder, cp_tracking=False)

        assert out["Identification"].unique().tolist() == [self.IDENT]

    def test_the_track_id_comes_from_the_identification_tail(
        self, tmp_path, parser
    ):
        folder = write_clt(
            tmp_path, parser, track_frame("exp-p0002-042"), "exp-p0002-042"
        )

        out = parser.load_folder(folder, folder, cp_tracking=False)

        assert out["track_id"].unique().tolist() == [42]

    def test_a_death_survives_the_round_trip(self, tmp_path, parser):
        original = track_frame(self.IDENT)
        original.loc[
            (original["TrackNumber"] == 3) & (original["t"] == 4), "Cellfate"
        ] = "Dead"
        folder = write_clt(tmp_path, parser, original, self.IDENT)

        out = parser.load_folder(folder, folder, cp_tracking=False)

        dead = out[out["Cellfate"] == "Dead"]
        assert dead["TrackNumber"].tolist() == [3]
        assert dead["t"].tolist() == [4]

    def test_a_fate_lands_only_on_the_last_row_of_a_track(
        self, tmp_path, parser
    ):
        """The file stores one fate per cell; the frame puts it on the end."""
        original = track_frame(self.IDENT)
        original.loc[
            (original["TrackNumber"] == 3) & (original["t"] == 4), "Cellfate"
        ] = "Dead"
        folder = write_clt(tmp_path, parser, original, self.IDENT)

        out = parser.load_folder(folder, folder, cp_tracking=False)

        track_3 = out[out["TrackNumber"] == 3].sort_values("t")
        assert track_3["Cellfate"].tolist() == ["Healthy", "Dead"]

    def test_several_files_load_together(self, tmp_path, parser):
        folder = tmp_path / "experiment"
        folder.mkdir()
        for ident in ("exp-p0002-001", "exp-p0002-002"):
            (folder / f"{ident}.clt").write_text(
                parser._new_clt_text(track_frame(ident), "tester")
            )

        out = parser.load_folder(str(folder), str(folder), cp_tracking=False)

        assert sorted(out["Identification"].unique()) == [
            "exp-p0002-001",
            "exp-p0002-002",
        ]

    def test_files_are_found_recursively(self, tmp_path, parser):
        nested = tmp_path / "experiment" / "sub" / "deeper"
        nested.mkdir(parents=True)
        (nested / f"{self.IDENT}.clt").write_text(
            parser._new_clt_text(track_frame(self.IDENT), "tester")
        )

        out = parser.load_folder(
            str(tmp_path / "experiment"),
            str(tmp_path),
            cp_tracking=False,
        )

        assert not out.empty

    def test_macos_resource_forks_are_ignored(self, tmp_path, parser):
        """Cloud-synced drives leave ._ sidecar files next to the real ones."""
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / f"{self.IDENT}.clt").write_text(
            parser._new_clt_text(track_frame(self.IDENT), "tester")
        )
        (folder / f"._{self.IDENT}.clt").write_bytes(b"\x00binary junk")

        parser.load_folder(str(folder), str(folder), cp_tracking=False)

        assert len(parser.clt_paths) == 1

    def test_the_progress_callback_reports_each_file(self, tmp_path, parser):
        folder = tmp_path / "experiment"
        folder.mkdir()
        for ident in ("exp-p0002-001", "exp-p0002-002"):
            (folder / f"{ident}.clt").write_text(
                parser._new_clt_text(track_frame(ident), "tester")
            )
        seen = []

        parser.load_folder(
            str(folder),
            str(folder),
            cp_tracking=False,
            progress_fun=lambda i, total: seen.append((i, total)),
        )

        assert seen == [(0, 2), (1, 2), (2, 2)]

    def test_a_file_with_no_tracking_block_contributes_nothing(
        self, tmp_path, parser
    ):
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / f"{self.IDENT}.clt").write_text(
            parser._new_clt_text(track_frame(self.IDENT), "tester")
        )
        (folder / "exp-p0002-002.clt").write_text(
            '%% CellLineageTree Version="0.0.5"\n\n%% Changelog\n'
        )

        out = parser.load_folder(str(folder), str(folder), cp_tracking=False)

        assert out["Identification"].unique().tolist() == [self.IDENT]

    def test_an_unreadable_file_does_not_abort_the_load(
        self, tmp_path, parser
    ):
        """One bad file must not cost the user every other one."""
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / f"{self.IDENT}.clt").write_text(
            parser._new_clt_text(track_frame(self.IDENT), "tester")
        )
        # A directory where a file is expected: open() raises OSError.
        (folder / "exp-p0002-002.clt").mkdir()

        out = parser.load_folder(str(folder), str(folder), cp_tracking=False)

        assert out["Identification"].unique().tolist() == [self.IDENT]

    def test_an_empty_folder_yields_the_empty_schema(self, tmp_path, parser):
        folder = tmp_path / "empty"
        folder.mkdir()

        out = parser.load_folder(str(folder), str(folder), cp_tracking=False)

        assert out.empty
        assert list(out.columns) == CLTParser._ROW_COLS


# Coordinate conversion via TAT metadata
TAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<TATSettings>
  <MicrometerPerPixel value="0.5" />
  <WavelengthData>
    <WavelengthInformation>
      <WLInfo Name="1" Comment="phase" width="100" height="80" />
    </WavelengthInformation>
  </WavelengthData>
  <PositionData>
    <PositionInformation>
      <PosInfoDimension index="2" posX="1000" posY="2000" />
    </PositionInformation>
  </PositionData>
</TATSettings>
"""


@pytest.fixture
def tat_parser(tmp_path) -> CLTParser:
    """A parser that converts between pixel and global stage coordinates."""
    path = tmp_path / "TATexp.xml"
    path.write_text(TAT_XML)
    return CLTParser(str(path))


class TestCoordinateConversion:
    def test_global_xy_scales_and_offsets_pixels(self, tat_parser):
        """x_global = x_px * um_per_px + posX."""
        assert tat_parser._global_xy(2, 100.0, 200.0) == (1050.0, 2100.0)

    def test_global_xy_is_the_identity_without_tat(self, parser):
        assert parser._global_xy(2, 100.0, 200.0) == (100.0, 200.0)

    def test_pixel_conversion_inverts_global_xy(self, tat_parser):
        df = pd.DataFrame(
            {
                "Position": [2],
                "XMorphology": [1050.0],
                "YMorphology": [2100.0],
            }
        )

        out = tat_parser._apply_pixel_conversion(df)

        assert out["XMorphology"].tolist() == [100]
        assert out["YMorphology"].tolist() == [200]

    def test_pixel_conversion_is_a_no_op_without_tat(self, parser):
        df = pd.DataFrame(
            {"Position": [2], "XMorphology": [1.5], "YMorphology": [2.5]}
        )

        out = parser._apply_pixel_conversion(df)

        assert out["XMorphology"].tolist() == [1.5]

    def test_the_round_trip_survives_the_conversion(
        self, tmp_path, tat_parser
    ):
        """Written as global coordinates, read back as the original pixels."""
        original = track_frame("exp-p0002-001")
        folder = write_clt(tmp_path, tat_parser, original, "exp-p0002-001")

        out = tat_parser.load_folder(folder, folder, cp_tracking=False)

        merged = original.merge(
            out, on=["TrackNumber", "t"], suffixes=("_in", "_out")
        )
        assert merged["XMorphology_out"].tolist() == pytest.approx(
            merged["XMorphology_in"].tolist()
        )
        assert merged["YMorphology_out"].tolist() == pytest.approx(
            merged["YMorphology_in"].tolist()
        )

    def test_cross_position_tracking_derives_position_from_coordinates(
        self, tmp_path, tat_parser
    ):
        """With cp_tracking the coordinates decide the position, not the file.

        The file below declares FieldOfView 7, but its global coordinates sit
        inside the tile declared for index 2, so the loader must report 2.
        """
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / "exp-p0002-001.clt").write_text(
            '%% TrackingData Units="LocalPixel"\n'
            "\t% Cell id=1\n"
            "\t\t1;7;1;0;1005;2010\n"
            "\t\t2;7;1;0;1010;2015\n"
        )

        out = tat_parser.load_folder(
            str(folder), str(folder), cp_tracking=True
        )

        assert out["Position"].unique().tolist() == [2]

    def test_cross_position_tracking_still_converts_to_pixels(
        self, tmp_path, tat_parser
    ):
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / "exp-p0002-001.clt").write_text(
            '%% TrackingData Units="LocalPixel"\n'
            "\t% Cell id=1\n"
            "\t\t1;7;1;0;1005;2010\n"
        )

        out = tat_parser.load_folder(
            str(folder), str(folder), cp_tracking=True
        )

        # (1005 - 1000) / 0.5 = 10 and (2010 - 2000) / 0.5 = 20.
        assert out["XMorphology"].tolist() == [10]
        assert out["YMorphology"].tolist() == [20]
