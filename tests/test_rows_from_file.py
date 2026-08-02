"""Tests for CLTParser._rows_from_file."""

from __future__ import annotations

from pathlib import Path

from SECQUOIA.core.tracking.clt_io import CLTParser


def write_clt(tmp_path: Path, text: str, name: str = "case-1.clt") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_normal_file_with_declared_columns(tmp_path):
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        "\t% Columns\n"
        "\t\t% Frame\n"
        "\t\t\tTimePoint\n"
        "\t\t\tFieldOfView\n"
        "\t\t\tZIndex\n"
        "\t\t\tImagingChannel\n"
        "\t\t% Data\n"
        "\t\t\tdouble x\n"
        "\t\t\tdouble y\n"
        '\t% Cell id=1 cellFate=""\n'
        "\t\t1;0;1;0;100.0;200.0\n"
        "\t\t2;0;1;0;101.0;201.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [
        (1, 100.0, 200.0, 0, 0, 1, "Healthy", "case-1"),
        (1, 101.0, 201.0, 1, 0, 1, "Healthy", "case-1"),
    ]


def test_file_without_columns_block_falls_back_to_defaults(tmp_path):
    """No '% Columns' section at all -> default Frame/Data schema is used."""
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell id=5 cellFate="Division"\n'
        "\t\t1;2;1;0;10.0;20.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(1, 10.0, 20.0, 0, 2, 5, "Healthy", "case-1")]


def test_declared_data_columns_are_ignored_in_favor_of_defaults(tmp_path):
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        "\t% Columns\n"
        "\t\t% Frame\n"
        "\t\t\tTimePoint\n"
        "\t\t\tFieldOfView\n"
        "\t\t\tZIndex\n"
        "\t\t\tImagingChannel\n"
        "\t\t% Data\n"
        "\t\t\tdouble x\n"
        "\t\t\tdouble y\n"
        "\t\t\tdouble area\n"
        '\t% Cell id=1 cellFate=""\n'
        "\t\t1;0;1;0;10.0;20.0;999.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(1, 10.0, 20.0, 0, 0, 1, "Healthy", "case-1")]


def test_cellfate_normalization(tmp_path):
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell id=1 cellFate="Death"\n'
        "\t\t1;0;1;0;1.0;1.0\n"
        '\t% Cell id=2 cellFate="Custom"\n'
        "\t\t1;0;1;0;2.0;2.0\n"
        "\t% Cell id=3\n"
        "\t\t1;0;1;0;3.0;3.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [
        (1, 1.0, 1.0, 0, 0, 1, "Dead", "case-1"),
        (1, 2.0, 2.0, 0, 0, 2, "Custom", "case-1"),
        (1, 3.0, 3.0, 0, 0, 3, "Healthy", "case-1"),
    ]


def test_malformed_lines_are_skipped(tmp_path):
    """Too few fields, and non-numeric values, are silently skipped."""
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell id=1 cellFate=""\n'
        "\t\t1;0;1;0;10.0\n"
        "\t\t1;0;1;0;abc;20.0\n"
        "\t\t2;0;1;0;11.0;21.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(1, 11.0, 21.0, 1, 0, 1, "Healthy", "case-1")]


def test_cell_without_id_is_skipped_entirely(tmp_path):
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell cellFate=""\n'
        "\t\t1;0;1;0;10.0;20.0\n"
        '\t% Cell id=9 cellFate=""\n'
        "\t\t1;0;1;0;11.0;21.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(1, 11.0, 21.0, 0, 0, 9, "Healthy", "case-1")]


def test_no_trackingdata_block_returns_empty(tmp_path):
    text = "%% Changelog\n\t% Columns\n\t\tstring user\n"
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == []


def test_secquoia_quant_block_is_stripped_before_parsing(tmp_path):
    text = (
        '%% Quantification Name="q" User="u" Software="SECQUOIA"\n'
        "\t% Columns\n"
        "\t\t% Frame\n"
        "\t\t\tTimePoint\n"
        "\t% Cell id=999\n"
        "\t\t1;0;1;0;999;999\n"
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell id=1 cellFate=""\n'
        "\t\t1;0;1;0;5.0;6.0\n"
    )
    path = write_clt(tmp_path, text)
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(1, 5.0, 6.0, 0, 0, 1, "Healthy", "case-1")]


def test_track_id_and_identification_come_from_filename(tmp_path):
    text = (
        '%% TrackingData Units="LocalPixel"\n'
        '\t% Cell id=1 cellFate=""\n'
        "\t\t1;0;1;0;5.0;6.0\n"
    )
    path = write_clt(tmp_path, text, name="pos3-7abc.clt")
    rows = CLTParser()._rows_from_file(path)
    assert rows == [(7, 5.0, 6.0, 0, 0, 1, "Healthy", "pos3-7")]
