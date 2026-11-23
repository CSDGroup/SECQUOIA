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

from SECQUOIA.core.tracking.clt_io import CLTParser


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


class TestQuantificationBlockText:
    """The tTt quantification text must stay exactly as earlier versions wrote it."""

    def test_matches_the_established_format(self):
        """Sorting, frame columns, number format and text values."""
        df = pd.DataFrame(
            {
                "Identification": "e-p0001-001",
                "TrackNumber": [2.0, 1.0, 1.0, 3.5, 1.0],
                "t": [1.0, 2.0, 0.0, 0.0, np.nan],
                "Position": [1, 1, np.nan, 1, 1],
                "ZIndex": [np.nan, 2, 1, 1, 1],
                "Mean": [12.3456789, np.nan, 1e7, 0.5, 1.0],
                "Count": [3, 4, 5, 6, 7],
                "Flag": [True, False, True, True, True],
                "Fate": ["Healthy", "x;y", 2.5, None, "z"],
            }
        )

        text = CLTParser()._build_quantification_block_from_df(
            df,
            user="MA",
            quant_name="Q",
            software="S",
            remark="",
            imaging_channel=0,
            default_zindex=1,
            t_offset=10,
        )

        assert text == (
            '%% Quantification Name="Q" User="MA" Software="S"\n'
            "\t% Columns\n\t\t% Frame\n\t\t\tTimePoint\n\t\t\tFieldOfView\n"
            "\t\t\tZIndex\n\t\t\tImagingChannel\n\t\t% Data\n"
            '\t\t\tdouble "Mean"\n\t\t\tdouble "Count"\n'
            '\t\t\tdouble "Flag"\n\t\t\tdouble "Fate"\n'
            "\t% Cell id=1\n"
            "\t\t11;0;1;0;1e+07;5;1;2.5\n"
            "\t\t13;1;2;0;nan;4;0;x;y\n"
            "\t% Cell id=2\n"
            "\t\t12;1;1;0;12.3457;3;1;Healthy\n"
            "\t% Cell id=3\n"
            "\t\t11;1;1;0;0.5;6;1;nan"
        )


class TestCorrectedTrackingPointExport:
    """Simulate a right-click correction, then the export that saves it."""

    IDENT = "exp_p0002-001"

    @staticmethod
    def tracking_lines(folder, ident) -> list[str]:
        text = (folder / f"{ident}.clt").read_text()
        block = CLTParser._TRACKINGDATA_RE.search(text).group(0)
        return [ln for ln in block.splitlines() if ";" in ln]

    def load_correct_export(
        self, tmp_path, clt_parser, moves, *, t_offset=0, file_df=None
    ):
        """Write a CLT, load it, move points as right-click does, export it."""
        file_df = track_frame(self.IDENT) if file_df is None else file_df
        folder = tmp_path / "experiment"
        folder.mkdir(exist_ok=True)
        (folder / f"{self.IDENT}.clt").write_text(
            clt_parser._new_clt_text(file_df, "tester")
        )
        df = clt_parser.load_folder(str(folder), str(folder), False)
        df = df[df["t"] >= t_offset].assign(t=lambda d: d["t"] - t_offset)
        for (track, t), (x, y) in moves.items():
            row = (df["TrackNumber"] == track) & (df["t"] == t)
            df.loc[row, ["XMorphology", "YMorphology"]] = (float(x), float(y))

        clt_parser.export_quantifications_to_clt(
            df, user="tester", t_offset=t_offset
        )
        reloaded = CLTParser(None)
        reloaded.tat = clt_parser.tat
        return folder, reloaded.load_folder(str(folder), str(folder), False)

    @staticmethod
    def point(df, track, t):
        row = df[(df["TrackNumber"] == track) & (df["t"] == t)].iloc[0]
        return row["XMorphology"], row["YMorphology"]

    def test_an_export_without_corrections_leaves_tracking_untouched(
        self, tmp_path, tat_parser
    ):
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / f"{self.IDENT}.clt").write_text(
            tat_parser._new_clt_text(track_frame(self.IDENT), "tester")
        )
        before = self.tracking_lines(folder, self.IDENT)

        self.load_correct_export(tmp_path, tat_parser, {})

        assert self.tracking_lines(folder, self.IDENT) == before

    def test_the_corrected_point_is_loaded_back(self, tmp_path, tat_parser):
        _, out = self.load_correct_export(
            tmp_path, tat_parser, {(2, 3): (37, 41)}
        )

        assert self.point(out, 2, 3) == (37, 41)

    def test_only_the_corrected_line_changes(self, tmp_path, tat_parser):
        folder = tmp_path / "experiment"
        folder.mkdir()
        (folder / f"{self.IDENT}.clt").write_text(
            tat_parser._new_clt_text(track_frame(self.IDENT), "tester")
        )
        before = self.tracking_lines(folder, self.IDENT)

        self.load_correct_export(tmp_path, tat_parser, {(2, 3): (37, 41)})

        after = self.tracking_lines(folder, self.IDENT)
        changed = [
            (b, a) for b, a in zip(before, after, strict=True) if b != a
        ]
        assert changed == [
            ("\t\t4;2;1;0;1065;2107.5", "\t\t4;2;1;0;1018.5;2020.5")
        ]

    def test_a_rebased_interval_patches_the_right_time_point(
        self, tmp_path, tat_parser
    ):
        """Loaded from frame 2 on, rebased t=0 is file TimePoint 3."""
        folder, out = self.load_correct_export(
            tmp_path, tat_parser, {(1, 0): (5, 6)}, t_offset=2
        )

        assert self.point(out, 1, 2) == (5, 6)
        assert self.point(out, 1, 0) == (100, 200)
        assert "\t\t3;2;1;0;1002.5;2003" in self.tracking_lines(
            folder, self.IDENT
        )

    def test_without_tat_the_pixels_are_written_as_they_are(
        self, tmp_path, parser
    ):
        _, out = self.load_correct_export(tmp_path, parser, {(3, 4): (7, 8)})

        assert self.point(out, 3, 4) == (7, 8)

    def test_large_stage_coordinates_keep_the_exact_pixel(self, tmp_path):
        """Global values above 100000 um need more than six digits."""
        xml = TAT_XML.replace('value="0.5"', 'value="0.6455"').replace(
            'posX="1000" posY="2000"', 'posX="123456.789" posY="98765.4321"'
        )
        (tmp_path / "TATexp.xml").write_text(xml)
        clt_parser = CLTParser(str(tmp_path / "TATexp.xml"))
        moves = {(1, 0): (1, 79), (1, 1): (99, 2), (2, 4): (57, 33)}

        _, out = self.load_correct_export(tmp_path, clt_parser, moves)

        for (track, t), xy in moves.items():
            assert self.point(out, track, t) == xy

    def test_a_row_missing_from_the_file_adds_no_line(self, tat_parser):
        df = track_frame(self.IDENT)
        gap = (df["TrackNumber"] == 1) & (df["t"] == 1)
        text = tat_parser._new_clt_text(df[~gap], "tester")
        df.loc[gap, ["XMorphology", "YMorphology"]] = (9.0, 9.0)

        assert tat_parser._patch_tracking_xy(text, df) == text
