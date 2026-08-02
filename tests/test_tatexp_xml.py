"""Tests for SECQUOIA TATexp metadata.

The test file is a small TATexp.xml with two channels of 100 x 80 pixels at
0.5 micrometres per pixel, and two stage positions. Each field of view is
therefore 50 by 40 micrometres, which makes it possible to say by hand which
coordinates fall inside a position and which fall outside.
"""

from __future__ import annotations

import pytest

from SECQUOIA.core.tatexp_xml import (
    TATParser,
    extract_channel_comments_from_xml,
    extract_position_comments_from_xml,
)


def tat_xml(
    *,
    um_per_px: str = "0.5",
    wavelengths: str | None = None,
    positions: str | None = None,
    root_tag: str = "TATSettings",
) -> str:
    """Build a TATexp.xml document, defaulting to a valid two-position layout."""
    if wavelengths is None:
        wavelengths = """
            <WavelengthInformation>
              <WLInfo Name="1" Comment="phase" width="100" height="80" />
            </WavelengthInformation>
            <WavelengthInformation>
              <WLInfo Name="2" Comment="GFP" width="100" height="80" />
            </WavelengthInformation>
        """
    if positions is None:
        positions = """
            <PositionInformation>
              <PosInfoDimension index="1" posX="0" posY="0" comments="first" />
            </PositionInformation>
            <PositionInformation>
              <PosInfoDimension index="2" posX="1000" posY="1000" comments="second" />
            </PositionInformation>
        """
    return f"""<?xml version="1.0" encoding="utf-8"?>
<{root_tag}>
  <MicrometerPerPixel value="{um_per_px}" />
  <WavelengthData>{wavelengths}</WavelengthData>
  <PositionData>{positions}</PositionData>
</{root_tag}>
"""


@pytest.fixture
def xml_path(tmp_path):
    """Factory writing a TATexp.xml into the temporary directory."""

    def _write(text: str = None, name: str = "TATexp.xml") -> str:
        path = tmp_path / name
        path.write_text(tat_xml() if text is None else text)
        return str(path)

    return _write


# TATParser
class TestTATParserFromXml:
    def test_reads_the_pixel_scale(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert parser.um_per_px == 0.5

    def test_reads_the_shared_frame_size(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert (parser.width, parser.height) == (100, 80)

    def test_counts_the_channels(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert parser.channel_count == 2

    def test_exposes_the_channel_name_from_the_comment(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert [w["ChannelName"] for w in parser.wavelengths] == [
            "phase",
            "GFP",
        ]

    def test_maps_each_position_index_to_its_stage_coordinates(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert parser.pos_map == {1: (0.0, 0.0), 2: (1000.0, 1000.0)}

    def test_harvests_the_remaining_metadata(self, xml_path):
        parser = TATParser.from_xml(xml_path())

        assert "TATSettings" in parser.extras

    def test_rejects_a_document_that_is_not_a_tatexp_file(self, xml_path):
        with pytest.raises(ValueError, match="TATSettings"):
            TATParser.from_xml(xml_path(tat_xml(root_tag="SomethingElse")))

    def test_rejects_a_missing_pixel_scale(self, xml_path):
        with pytest.raises(ValueError, match="MicrometerPerPixel"):
            TATParser.from_xml(xml_path(tat_xml(um_per_px="not a number")))

    def test_rejects_a_document_with_no_channels(self, xml_path):
        with pytest.raises(ValueError, match="WLInfo"):
            TATParser.from_xml(xml_path(tat_xml(wavelengths="")))

    def test_rejects_channels_of_differing_size(self, xml_path):
        """A mismatch would make the coverage rectangles meaningless."""
        mismatched = """
            <WavelengthInformation>
              <WLInfo Name="1" Comment="a" width="100" height="80" />
            </WavelengthInformation>
            <WavelengthInformation>
              <WLInfo Name="2" Comment="b" width="200" height="80" />
            </WavelengthInformation>
        """
        with pytest.raises(ValueError, match="identical width/height"):
            TATParser.from_xml(xml_path(tat_xml(wavelengths=mismatched)))

    def test_rejects_a_channel_without_a_size(self, xml_path):
        sizeless = """
            <WavelengthInformation>
              <WLInfo Name="1" Comment="a" />
            </WavelengthInformation>
        """
        with pytest.raises(ValueError, match="numeric 'width' and 'height'"):
            TATParser.from_xml(xml_path(tat_xml(wavelengths=sizeless)))

    def test_rejects_a_document_with_no_usable_positions(self, xml_path):
        with pytest.raises(ValueError, match="PositionData"):
            TATParser.from_xml(xml_path(tat_xml(positions="")))

    def test_skips_incomplete_position_entries(self, xml_path):
        partial = """
            <PositionInformation>
              <PosInfoDimension index="1" posX="0" posY="0" />
            </PositionInformation>
            <PositionInformation>
              <PosInfoDimension index="2" posX="500" />
            </PositionInformation>
        """
        parser = TATParser.from_xml(xml_path(tat_xml(positions=partial)))

        assert set(parser.pos_map) == {1}

    def test_skips_positions_with_unparsable_coordinates(self, xml_path):
        bad = """
            <PositionInformation>
              <PosInfoDimension index="1" posX="0" posY="0" />
            </PositionInformation>
            <PositionInformation>
              <PosInfoDimension index="2" posX="left" posY="0" />
            </PositionInformation>
        """
        parser = TATParser.from_xml(xml_path(tat_xml(positions=bad)))

        assert set(parser.pos_map) == {1}


class TestTATParserQueryIndex:
    @pytest.fixture
    def parser(self, xml_path):
        return TATParser.from_xml(xml_path())

    def test_finds_the_position_covering_a_point(self, parser):
        assert parser.query_index(10.0, 10.0) == 1

    def test_finds_the_second_position(self, parser):
        assert parser.query_index(1010.0, 1010.0) == 2

    def test_the_tile_origin_is_inside_the_tile(self, parser):
        assert parser.query_index(0.0, 0.0) == 1

    def test_the_far_tile_corner_is_inside_the_tile(self, parser):
        """100 px x 0.5 um/px = 50 um wide, 80 px = 40 um high."""
        assert parser.query_index(50.0, 40.0) == 1

    def test_a_point_just_past_the_tile_is_outside(self, parser):
        assert parser.query_index(50.1, 40.0) is None

    def test_a_point_covered_by_nothing_returns_none(self, parser):
        assert parser.query_index(500.0, 500.0) is None

    def test_overlapping_tiles_tie_break_on_the_closest_centroid(
        self, xml_path
    ):
        overlapping = """
            <PositionInformation>
              <PosInfoDimension index="1" posX="0" posY="0" />
            </PositionInformation>
            <PositionInformation>
              <PosInfoDimension index="2" posX="10" posY="0" />
            </PositionInformation>
        """
        parser = TATParser.from_xml(xml_path(tat_xml(positions=overlapping)))

        # Tile 1 spans x 0..50 (centroid 25), tile 2 spans 10..60 (centroid 35).
        assert parser.query_index(20.0, 20.0) == 1
        assert parser.query_index(40.0, 20.0) == 2


# Comment extraction
class TestExtractChannelComments:
    def test_maps_channel_keys_to_comments(self, xml_path):
        assert extract_channel_comments_from_xml(xml_path()) == {
            "w01": "phase",
            "w02": "GFP",
        }

    def test_pads_the_channel_key_to_two_digits(self, xml_path):
        single = """
            <WavelengthInformation>
              <WLInfo Name="7" Comment="far red" width="10" height="10" />
            </WavelengthInformation>
        """
        result = extract_channel_comments_from_xml(
            xml_path(tat_xml(wavelengths=single))
        )

        assert result == {"w07": "far red"}

    def test_falls_back_to_the_digits_in_a_non_numeric_name(self, xml_path):
        odd = """
            <WavelengthInformation>
              <WLInfo Name="ch3" Comment="odd" width="10" height="10" />
            </WavelengthInformation>
        """
        result = extract_channel_comments_from_xml(
            xml_path(tat_xml(wavelengths=odd))
        )

        assert result == {"w03": "odd"}

    def test_a_channel_without_a_comment_maps_to_an_empty_string(
        self, xml_path
    ):
        no_comment = """
            <WavelengthInformation>
              <WLInfo Name="1" width="10" height="10" />
            </WavelengthInformation>
        """
        result = extract_channel_comments_from_xml(
            xml_path(tat_xml(wavelengths=no_comment))
        )

        assert result == {"w01": ""}

    def test_a_missing_file_yields_an_empty_mapping(self, tmp_path):
        assert (
            extract_channel_comments_from_xml(str(tmp_path / "no.xml")) == {}
        )

    def test_no_path_yields_an_empty_mapping(self):
        assert extract_channel_comments_from_xml(None) == {}

    def test_a_malformed_document_yields_an_empty_mapping(self, xml_path):
        assert extract_channel_comments_from_xml(xml_path("<not xml")) == {}


class TestExtractPositionComments:
    def test_maps_position_numbers_to_comments(self, xml_path):
        assert extract_position_comments_from_xml(xml_path()) == {
            1: "first",
            2: "second",
        }

    def test_strips_leading_zeros_from_the_index(self, xml_path):
        padded = """
            <PositionInformation>
              <PosInfoDimension index="0007" posX="0" posY="0" comments="p7" />
            </PositionInformation>
        """
        result = extract_position_comments_from_xml(
            xml_path(tat_xml(positions=padded))
        )

        assert result == {7: "p7"}

    def test_an_all_zero_index_becomes_zero(self, xml_path):
        zeroes = """
            <PositionInformation>
              <PosInfoDimension index="0000" posX="0" posY="0" comments="z" />
            </PositionInformation>
        """
        result = extract_position_comments_from_xml(
            xml_path(tat_xml(positions=zeroes))
        )

        assert result == {0: "z"}

    def test_a_position_without_comments_maps_to_an_empty_string(
        self, xml_path
    ):
        bare = """
            <PositionInformation>
              <PosInfoDimension index="1" posX="0" posY="0" />
            </PositionInformation>
        """
        result = extract_position_comments_from_xml(
            xml_path(tat_xml(positions=bare))
        )

        assert result == {1: ""}

    def test_skips_an_unparsable_index(self, xml_path):
        bad = """
            <PositionInformation>
              <PosInfoDimension index="one" posX="0" posY="0" comments="x" />
            </PositionInformation>
            <PositionInformation>
              <PosInfoDimension index="2" posX="0" posY="0" comments="ok" />
            </PositionInformation>
        """
        result = extract_position_comments_from_xml(
            xml_path(tat_xml(positions=bad))
        )

        assert result == {2: "ok"}

    def test_a_missing_file_yields_an_empty_mapping(self, tmp_path):
        assert (
            extract_position_comments_from_xml(str(tmp_path / "no.xml")) == {}
        )

    def test_a_malformed_document_yields_an_empty_mapping(self, xml_path):
        assert extract_position_comments_from_xml(xml_path("<not xml")) == {}
