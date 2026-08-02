"""Tests for the SECQUOIA tTt data format transformer.

The transformer copies images into the tTt folder layout and renames them on
the way. The test data is four images named in a Nikon style, two positions
over two time points, copied into a temporary output folder so the resulting
names and folders can be checked.
"""

from __future__ import annotations

import os

import pytest

from SECQUOIA.gui.ttt_data_format_transformer import (
    find_one_image_file,
    iter_image_files,
    no_images_message,
    normalize_initials,
    normalize_pos,
    normalize_setup,
    normalize_t,
    normalize_w,
    normalize_z,
    parse_experiment_token_from_filename,
    parse_name,
    yymmdd_compact,
    yyyymmdd_from_prefix,
)


# Token normalisation
class TestNormalizePos:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("p1", "p0001"),
            ("p0001", "p0001"),
            ("p42", "p0042"),
            ("P7", "p0007"),
            ("xy1", "p0001"),
            ("xy12", "p0012"),
            ("XY0003", "p0003"),
        ],
    )
    def test_pads_to_four_digits(self, raw, expected):
        assert normalize_pos(raw) == expected

    def test_leaves_an_unrecognised_token_alone(self):
        assert normalize_pos("well_A1") == "well_a1"

    def test_an_empty_token_stays_empty(self):
        assert normalize_pos("") == ""


class TestNormalizeT:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("t1", "t00001"), ("t00001", "t00001"), ("T123", "t00123")],
    )
    def test_pads_to_five_digits(self, raw, expected):
        assert normalize_t(raw) == expected

    def test_leaves_an_unrecognised_token_alone(self):
        assert normalize_t("time1") == "time1"


class TestNormalizeZ:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("z1", "z001"), ("z001", "z001"), ("Z12", "z012")],
    )
    def test_pads_to_three_digits(self, raw, expected):
        assert normalize_z(raw) == expected


class TestNormalizeW:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("w0", "w00"), ("w1", "w01"), ("w00", "w00"), ("W2", "w02")],
    )
    def test_w_tokens_are_zero_based(self, raw, expected):
        assert normalize_w(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"), [("c1", "w00"), ("c2", "w01"), ("C3", "w02")]
    )
    def test_c_tokens_are_one_based_and_shift_down(self, raw, expected):
        """This is the conversion that is easy to get wrong by one."""
        assert normalize_w(raw) == expected

    def test_c0_is_rejected_rather_than_wrapping_negative(self):
        assert normalize_w("c0") == ""

    def test_surrounding_whitespace_is_ignored(self):
        assert normalize_w("  c2  ") == "w01"


class TestNormalizeSetup:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("3", "03"), ("30", "30"), ("035", "35")]
    )
    def test_pads_to_two_digits(self, raw, expected):
        assert normalize_setup(raw) == expected

    def test_strips_non_digits(self):
        assert normalize_setup("setup 7") == "07"

    def test_nothing_numeric_gives_nothing(self):
        assert normalize_setup("abc") == ""


class TestNormalizeInitials:
    def test_uppercases_and_keeps_letters_only(self):
        assert normalize_initials("ma") == "MA"
        assert normalize_initials("m.a.") == "MA"

    def test_truncates_to_four(self):
        assert normalize_initials("abcdef") == "ABCD"


class TestDateHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2024-03-23", "240323"),
            ("20240323", "240323"),
            ("24-03-23", "240323"),
            ("240323", "240323"),
        ],
    )
    def test_yymmdd_compact_accepts_every_documented_form(self, raw, expected):
        assert yymmdd_compact(raw) == expected

    @pytest.mark.parametrize("raw", ["", "not a date", "2024-13-01", "24"])
    def test_yymmdd_compact_rejects_the_impossible(self, raw):
        assert yymmdd_compact(raw) == ""

    def test_yymmdd_compact_rejects_an_impossible_day(self):
        assert yymmdd_compact("2024-03-45") == ""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("240323", "2024-03-23"), ("20240323", "2024-03-23")],
    )
    def test_yyyymmdd_from_prefix_expands(self, raw, expected):
        assert yyyymmdd_from_prefix(raw) == expected

    def test_yyyymmdd_from_prefix_rejects_other_lengths(self):
        assert yyyymmdd_from_prefix("2403") == ""


# Filename parsing
class TestParseName:
    def test_reads_every_token_from_a_ttt_name(self):
        assert parse_name("240323MA35_p0001_t00001_z001_w00.png") == {
            "date": "2024-03-23",
            "initials": "",
            "pos": "p0001",
            "t": "t00001",
            "z": "z001",
            "ch": "w00",
        }

    def test_reads_a_nikon_style_name(self):
        """xy/c tokens, no date prefix."""
        info = parse_name("experiment_xy03t007c2.tif")

        assert info["pos"] == "xy03"
        assert info["t"] == "t007"
        assert info["ch"] == "c2"

    def test_ignores_letters_that_merely_end_in_a_token_letter(self):
        """The lookbehind stops 'plate1' being read as position 1."""
        info = parse_name("plate1_experiment.tif")

        assert info["pos"] == ""

    def test_missing_tokens_come_back_empty(self):
        info = parse_name("just_an_image.png")

        assert info == {
            "date": "",
            "initials": "",
            "pos": "",
            "t": "",
            "z": "",
            "ch": "",
        }

    def test_tokens_are_lowercased(self):
        info = parse_name("240323MA35_P0002_T00003_Z004_W05.png")

        assert (info["pos"], info["t"], info["z"], info["ch"]) == (
            "p0002",
            "t00003",
            "z004",
            "w05",
        )

    def test_the_extension_is_not_parsed(self):
        assert parse_name("/a/b/240323MA35_p0001_t1_z1_w0.tiff")["ch"] == "w0"


class TestParseExperimentTokenFromFilename:
    def test_reads_date_initials_and_setup(self):
        assert parse_experiment_token_from_filename(
            "240323MA35_p0001_t00001_z001_w00.png"
        ) == {"date": "2024-03-23", "initials": "MA", "setup": "35"}

    def test_accepts_an_eight_digit_date(self):
        result = parse_experiment_token_from_filename("20240323MA35_p1.png")

        assert result["date"] == "2024-03-23"

    def test_falls_back_to_the_date_alone(self):
        """A date prefix with no initials/setup still yields the date."""
        result = parse_experiment_token_from_filename("240323_something.png")

        assert result == {"date": "2024-03-23", "initials": "", "setup": ""}

    def test_returns_blanks_for_an_unrecognised_name(self):
        assert parse_experiment_token_from_filename("image.png") == {
            "date": "",
            "initials": "",
            "setup": "",
        }


# Folder scanning
@pytest.fixture
def image_folder(tmp_path):
    """A folder with three images, a non-image, and a nested image."""
    for name in ("b.png", "a.tif", "c.JPG", "notes.txt"):
        (tmp_path / name).touch()
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.png").touch()
    return tmp_path


class TestIterImageFiles:
    def test_yields_only_image_files(self, image_folder):
        names = [os.path.basename(p) for p in iter_image_files(image_folder)]

        assert "notes.txt" not in names

    def test_yields_them_sorted(self, image_folder):
        names = [os.path.basename(p) for p in iter_image_files(image_folder)]

        assert names == ["a.tif", "b.png", "c.JPG"]

    def test_does_not_descend_into_subfolders(self, image_folder):
        names = [os.path.basename(p) for p in iter_image_files(image_folder)]

        assert "deep.png" not in names

    def test_the_extension_check_is_case_insensitive(self, image_folder):
        names = [os.path.basename(p) for p in iter_image_files(image_folder)]

        assert "c.JPG" in names

    def test_find_one_returns_the_first(self, image_folder):
        assert os.path.basename(find_one_image_file(str(image_folder))) == (
            "a.tif"
        )

    def test_find_one_returns_empty_for_a_folder_with_no_images(
        self, tmp_path
    ):
        (tmp_path / "notes.txt").touch()

        assert find_one_image_file(str(tmp_path)) == ""


class TestNoImagesMessage:
    def test_points_at_subfolders_when_that_is_the_mistake(self, tmp_path):
        """The most likely user error: picking the parent folder."""
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "img.png").touch()

        message = no_images_message(str(tmp_path))

        assert "subfolders" in message

    def test_stays_short_when_there_is_nothing_anywhere(self, tmp_path):
        message = no_images_message(str(tmp_path))

        assert "subfolders" not in message


# The rename itself
@pytest.fixture
def transformer(qtbot, silence_modals):
    """The real widget, built offscreen and never shown."""
    from SECQUOIA.gui.ttt_data_format_transformer import (
        TttDataFormatTransformer,
    )

    widget = TttDataFormatTransformer()
    qtbot.addWidget(widget)
    widget._shown = silence_modals
    return widget


def configure(
    widget,
    in_folder,
    out_folder,
    *,
    date="240323",
    initials="MA",
    setup="35",
    pos="p1",
    time="t1",
    z="z1",
    channel="w0",
):
    """Fill the widget's fields the way the user would type them."""
    widget.in_folder = str(in_folder)
    widget.out_folder = str(out_folder)
    widget.ed_date.setText(date)
    widget.ed_initials.setText(initials)
    widget.ed_setup.setText(setup)
    widget.ed_pos.setText(pos)
    widget.ed_time.setText(time)
    widget.ed_z.setText(z)
    widget.ed_ch.setText(channel)


@pytest.fixture
def source_images(tmp_path):
    """Two positions, two time points, in a Nikon-ish naming scheme."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for pos in (1, 2):
        for t in (1, 2):
            (folder / f"acq_xy{pos:02d}t{t:03d}c1.tif").write_bytes(b"img")
    return folder


@pytest.fixture
def out_folder(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


EXP = "240323MA35"


class TestRunRenames:
    def test_creates_the_experiment_folder(
        self, transformer, source_images, out_folder
    ):
        configure(transformer, source_images, out_folder)

        transformer.run()

        assert (out_folder / EXP).is_dir()

    def test_creates_the_analysis_folder(
        self, transformer, source_images, out_folder
    ):
        configure(transformer, source_images, out_folder)

        transformer.run()

        assert (out_folder / EXP / "Analysis").is_dir()

    def test_creates_one_folder_per_position(
        self, transformer, source_images, out_folder
    ):
        configure(transformer, source_images, out_folder)

        transformer.run()

        assert (out_folder / EXP / f"{EXP}_p0001").is_dir()
        assert (out_folder / EXP / f"{EXP}_p0002").is_dir()

    def test_renames_every_image_into_the_ttt_layout(
        self, transformer, source_images, out_folder
    ):
        configure(transformer, source_images, out_folder)

        transformer.run()

        position_1 = out_folder / EXP / f"{EXP}_p0001"
        assert sorted(p.name for p in position_1.iterdir()) == [
            f"{EXP}_p0001_t00001_z001_w00.tif",
            f"{EXP}_p0001_t00002_z001_w00.tif",
        ]

    def test_the_c_channel_becomes_a_zero_based_w_channel(
        self, transformer, source_images, out_folder
    ):
        """The source files are all c1, which is the first channel: w00."""
        configure(transformer, source_images, out_folder)

        transformer.run()

        names = [p.name for p in (out_folder / EXP / f"{EXP}_p0001").iterdir()]
        assert all("_w00." in name for name in names)

    def test_copies_rather_than_moves(
        self, transformer, source_images, out_folder
    ):
        """The originals are the user's data and must survive."""
        configure(transformer, source_images, out_folder)

        transformer.run()

        assert len(list(source_images.iterdir())) == 4

    def test_the_file_contents_are_preserved(
        self, transformer, source_images, out_folder
    ):
        configure(transformer, source_images, out_folder)

        transformer.run()

        copied = (
            out_folder
            / EXP
            / f"{EXP}_p0001"
            / (f"{EXP}_p0001_t00001_z001_w00.tif")
        )
        assert copied.read_bytes() == b"img"

    def test_keeps_the_original_extension(
        self, transformer, tmp_path, out_folder
    ):
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "acq_xy01t001c1.png").write_bytes(b"img")
        configure(transformer, folder, out_folder)

        transformer.run()

        assert (
            out_folder
            / EXP
            / f"{EXP}_p0001"
            / f"{EXP}_p0001_t00001_z001_w00.png"
        ).exists()

    def test_falls_back_to_the_typed_fields_when_the_name_has_no_tokens(
        self, transformer, tmp_path, out_folder
    ):
        """An image with an opaque name still lands somewhere sensible."""
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "opaque.tif").write_bytes(b"img")
        configure(
            transformer, folder, out_folder, pos="p7", time="t9", channel="c2"
        )

        transformer.run()

        assert (
            out_folder
            / EXP
            / f"{EXP}_p0007"
            / f"{EXP}_p0007_t00009_z001_w01.tif"
        ).exists()

    def test_a_name_collision_is_given_a_dup_suffix(
        self, transformer, tmp_path, out_folder
    ):
        """Two sources that normalise to the same name must not overwrite."""
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "a_xy1t1c1.tif").write_bytes(b"first")
        (folder / "b_xy01t001c1.tif").write_bytes(b"second")
        configure(transformer, folder, out_folder)

        transformer.run()

        position_1 = out_folder / EXP / f"{EXP}_p0001"
        names = sorted(p.name for p in position_1.iterdir())
        assert names == [
            f"{EXP}_p0001_t00001_z001_w00.tif",
            f"{EXP}_p0001_t00001_z001_w00_dup2.tif",
        ]

    def test_both_colliding_files_keep_their_own_contents(
        self, transformer, tmp_path, out_folder
    ):
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "a_xy1t1c1.tif").write_bytes(b"first")
        (folder / "b_xy01t001c1.tif").write_bytes(b"second")
        configure(transformer, folder, out_folder)

        transformer.run()

        position_1 = out_folder / EXP / f"{EXP}_p0001"
        contents = {p.read_bytes() for p in position_1.iterdir()}
        assert contents == {b"first", b"second"}


class TestRunGuards:
    def test_refuses_without_folders_selected(
        self, transformer, silence_modals
    ):
        transformer.in_folder = ""
        transformer.out_folder = ""

        transformer.run()

        assert "warning" in silence_modals

    def test_refuses_without_the_required_fields(
        self, transformer, source_images, out_folder, silence_modals
    ):
        configure(transformer, source_images, out_folder, date="", initials="")

        transformer.run()

        assert "warning" in silence_modals
        assert list(out_folder.iterdir()) == []

    def test_rejects_initials_shorter_than_two_letters(
        self, transformer, source_images, out_folder, silence_modals
    ):
        configure(transformer, source_images, out_folder, initials="M")

        transformer.run()

        assert "warning" in silence_modals
        assert list(out_folder.iterdir()) == []

    def test_normalises_the_typed_fields_in_place(
        self, transformer, source_images, out_folder
    ):
        configure(
            transformer,
            source_images,
            out_folder,
            date="2024-03-23",
            initials="ma",
            setup="5",
        )

        transformer.run()

        assert transformer.ed_date.text() == "240323"
        assert transformer.ed_initials.text() == "MA"
        assert transformer.ed_setup.text() == "05"

    def test_warns_when_the_folder_holds_no_images(
        self, transformer, tmp_path, out_folder, silence_modals
    ):
        empty = tmp_path / "empty"
        empty.mkdir()
        configure(transformer, empty, out_folder)

        transformer.run()

        assert "warning" in silence_modals
        assert [p for p in out_folder.rglob("*") if p.is_file()] == []

    def test_an_unresolvable_file_is_reported_but_the_rest_still_copy(
        self, transformer, tmp_path, out_folder, silence_modals
    ):
        """One bad file must not cost the user the whole run."""
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "good_xy1t1c1.tif").write_bytes(b"img")
        (folder / "bad.tif").write_bytes(b"img")
        configure(transformer, folder, out_folder, channel="c0")

        transformer.run()

        assert "warning" in silence_modals
        assert (
            out_folder
            / EXP
            / f"{EXP}_p0001"
            / f"{EXP}_p0001_t00001_z001_w00.tif"
        ).exists()

    def test_the_unresolvable_file_is_not_copied(
        self, transformer, tmp_path, out_folder
    ):
        folder = tmp_path / "raw"
        folder.mkdir()
        (folder / "good_xy1t1c1.tif").write_bytes(b"img")
        (folder / "bad.tif").write_bytes(b"img")
        configure(transformer, folder, out_folder, channel="c0")

        transformer.run()

        copied = list((out_folder / EXP).rglob("*.tif"))
        assert len(copied) == 1
