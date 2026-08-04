"""Tests for SECQUOIA outlier flag syncing.

Flags are found on the position currently loaded and written back to the
dataframe that holds every position. The data is two positions with two
cells each over two frames, the smallest layout in which it can be shown
that reviewing one position leaves the other one alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from SECQUOIA.core.outlier_detection.track_sync import (
    _column_match_mask,
    update_outlier_detection_in_track_df,
    update_outlier_detection_in_track_df_fast,
    update_unique_outliers_ids,
)

OUTCOL = "Outlier_detection"


def track_frame() -> pd.DataFrame:
    """Two positions, two identifications each, two frames each."""
    rows = []
    for position in (1, 2):
        for ident in ("a", "b"):
            for t in (0, 1):
                rows.append((position, f"p{position}_{ident}", t, 1))
    return pd.DataFrame(
        rows, columns=["Position", "Identification", "t", "TrackNumber"]
    )


def filtered_frame(position: int = 1, outliers=()) -> pd.DataFrame:
    """The rows of one position, with ``outliers`` as (ident, t) pairs."""
    df = track_frame()
    df = df[df["Position"] == position].reset_index(drop=True)
    df[OUTCOL] = "OK"
    for ident, t in outliers:
        df.loc[(df["Identification"] == ident) & (df["t"] == t), OUTCOL] = (
            "Outlier"
        )
    return df


# Full sync
class TestUpdateOutlierDetectionInTrackDf:
    def test_marks_the_matching_rows(self, fake_main_window):
        main_window = fake_main_window(
            track_df=track_frame(),
            filtered_df=filtered_frame(1, [("p1_a", 1)]),
        )

        update_outlier_detection_in_track_df(main_window)

        out = main_window.track_df
        flagged = out[out[OUTCOL] == "Outlier"]
        assert flagged["Identification"].tolist() == ["p1_a"]
        assert flagged["t"].tolist() == [1]

    def test_creates_the_column_when_missing(self, fake_main_window):
        main_window = fake_main_window(
            track_df=track_frame(), filtered_df=filtered_frame(1)
        )

        update_outlier_detection_in_track_df(main_window)

        assert OUTCOL in main_window.track_df.columns

    def test_resets_stale_flags_for_the_reviewed_position(
        self, fake_main_window
    ):
        """A row that is no longer an outlier must go back to OK."""
        track_df = track_frame()
        track_df[OUTCOL] = "Outlier"
        main_window = fake_main_window(
            track_df=track_df, filtered_df=filtered_frame(1, [("p1_a", 1)])
        )

        update_outlier_detection_in_track_df(main_window)

        out = main_window.track_df
        position_1 = out[out["Position"] == 1]
        assert (position_1[OUTCOL] == "Outlier").sum() == 1

    def test_leaves_other_positions_untouched(self, fake_main_window):
        """Reviewing position 1 must not clear what was found in position 2."""
        track_df = track_frame()
        track_df[OUTCOL] = "OK"
        track_df.loc[track_df["Position"] == 2, OUTCOL] = "Outlier"
        main_window = fake_main_window(
            track_df=track_df, filtered_df=filtered_frame(1, [("p1_a", 0)])
        )

        update_outlier_detection_in_track_df(main_window)

        out = main_window.track_df
        assert (out[out["Position"] == 2][OUTCOL] == "Outlier").all()

    def test_resets_everything_when_position_is_not_a_shared_key(
        self, fake_main_window
    ):
        """Without a Position column there is no way to scope the reset."""
        track_df = track_frame().drop(columns=["Position"])
        track_df[OUTCOL] = "Outlier"
        filtered = filtered_frame(1).drop(columns=["Position"])
        main_window = fake_main_window(track_df=track_df, filtered_df=filtered)

        update_outlier_detection_in_track_df(main_window)

        assert (main_window.track_df[OUTCOL] == "OK").all()

    def test_no_outliers_leaves_the_subset_clean(self, fake_main_window):
        track_df = track_frame()
        track_df[OUTCOL] = "Outlier"
        main_window = fake_main_window(
            track_df=track_df, filtered_df=filtered_frame(1)
        )

        update_outlier_detection_in_track_df(main_window)

        out = main_window.track_df
        assert (out[out["Position"] == 1][OUTCOL] == "OK").all()

    def test_does_nothing_without_a_track_frame(self, fake_main_window):
        main_window = fake_main_window(
            track_df=None, filtered_df=filtered_frame(1)
        )

        update_outlier_detection_in_track_df(main_window)

        assert main_window.track_df is None

    def test_does_nothing_for_empty_frames(self, fake_main_window):
        main_window = fake_main_window(
            track_df=pd.DataFrame(), filtered_df=filtered_frame(1)
        )

        update_outlier_detection_in_track_df(main_window)

        assert main_window.track_df.empty

    def test_does_nothing_without_shared_keys(self, fake_main_window):
        main_window = fake_main_window(
            track_df=pd.DataFrame({"other": [1, 2]}),
            filtered_df=filtered_frame(1),
        )

        update_outlier_detection_in_track_df(main_window)

        assert OUTCOL not in main_window.track_df.columns

    def test_a_filtered_frame_without_the_column_only_resets(
        self, fake_main_window
    ):
        track_df = track_frame()
        track_df[OUTCOL] = "Outlier"
        main_window = fake_main_window(
            track_df=track_df,
            filtered_df=filtered_frame(1).drop(columns=[OUTCOL]),
        )

        update_outlier_detection_in_track_df(main_window)

        out = main_window.track_df
        assert (out[out["Position"] == 1][OUTCOL] == "OK").all()


# Fast sync
class TestUpdateOutlierDetectionInTrackDfFast:
    def test_updates_only_the_named_row(self, fake_main_window):
        filtered = filtered_frame(1, [("p1_a", 1)])
        main_window = fake_main_window(
            track_df=track_frame(), filtered_df=filtered
        )
        changed = filtered.index[filtered[OUTCOL] == "Outlier"].tolist()

        update_outlier_detection_in_track_df_fast(main_window, changed)

        out = main_window.track_df
        flagged = out[out[OUTCOL] == "Outlier"]
        assert len(flagged) == 1
        assert flagged["Identification"].tolist() == ["p1_a"]

    def test_accepts_a_bare_index(self, fake_main_window):
        filtered = filtered_frame(1, [("p1_a", 1)])
        main_window = fake_main_window(
            track_df=track_frame(), filtered_df=filtered
        )
        changed = int(filtered.index[filtered[OUTCOL] == "Outlier"][0])

        update_outlier_detection_in_track_df_fast(main_window, changed)

        assert (main_window.track_df[OUTCOL] == "Outlier").sum() == 1

    def test_can_clear_a_flag_again(self, fake_main_window):
        """The fast path copies the value across, not just the Outlier state."""
        track_df = track_frame()
        track_df[OUTCOL] = "Outlier"
        filtered = filtered_frame(1)  # everything back to OK
        main_window = fake_main_window(track_df=track_df, filtered_df=filtered)

        update_outlier_detection_in_track_df_fast(
            main_window, filtered.index.tolist()
        )

        out = main_window.track_df
        assert (out[out["Position"] == 1][OUTCOL] == "OK").all()

    def test_never_touches_another_position(self, fake_main_window):
        track_df = track_frame()
        track_df[OUTCOL] = "OK"
        track_df.loc[track_df["Position"] == 2, OUTCOL] = "Outlier"
        filtered = filtered_frame(1, [("p1_a", 0)])
        main_window = fake_main_window(track_df=track_df, filtered_df=filtered)

        update_outlier_detection_in_track_df_fast(
            main_window, filtered.index.tolist()
        )

        out = main_window.track_df
        assert (out[out["Position"] == 2][OUTCOL] == "Outlier").all()

    def test_agrees_with_the_full_sync(self, fake_main_window):
        """Both implementations must produce the same track_df."""
        filtered = filtered_frame(1, [("p1_a", 1), ("p1_b", 0)])

        slow_window = fake_main_window(
            track_df=track_frame(), filtered_df=filtered
        )
        update_outlier_detection_in_track_df(slow_window)

        fast_window = fake_main_window(
            track_df=track_frame(), filtered_df=filtered
        )
        update_outlier_detection_in_track_df_fast(
            fast_window, filtered.index.tolist()
        )

        pd.testing.assert_frame_equal(
            slow_window.track_df, fast_window.track_df
        )


class TestColumnMatchMask:
    def test_matches_an_integer_column(self):
        series = pd.Series([1, 2, 3], dtype="int64")

        assert _column_match_mask(2, series).tolist() == [False, True, False]

    def test_coerces_a_string_onto_an_integer_column(self):
        series = pd.Series([1, 2, 3], dtype="int64")

        assert _column_match_mask("2", series).tolist() == [
            False,
            True,
            False,
        ]

    def test_coerces_onto_a_float_column(self):
        series = pd.Series([1.0, 2.0], dtype="float64")

        assert _column_match_mask("2", series).tolist() == [False, True]

    def test_matches_a_string_column(self):
        series = pd.Series(["a", "b"])

        assert _column_match_mask("b", series).tolist() == [False, True]

    def test_a_missing_value_matches_missing_entries(self):
        series = pd.Series([1.0, np.nan])

        assert _column_match_mask(np.nan, series).tolist() == [False, True]

    def test_an_uncoercible_value_matches_nothing(self):
        series = pd.Series([1, 2], dtype="int64")

        assert not _column_match_mask("not a number", series).any()


# Outlier identification list
class TestUpdateUniqueOutliersIds:
    def test_collects_identifications_with_an_outlier(self, fake_main_window):
        main_window = fake_main_window(
            filtered_df=filtered_frame(1, [("p1_a", 0), ("p1_a", 1)])
        )

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == ["p1_a"]

    def test_lists_each_identification_once(self, fake_main_window):
        main_window = fake_main_window(
            filtered_df=filtered_frame(
                1, [("p1_a", 0), ("p1_a", 1), ("p1_b", 0)]
            )
        )

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == ["p1_a", "p1_b"]

    def test_is_empty_when_nothing_is_flagged(self, fake_main_window):
        main_window = fake_main_window(filtered_df=filtered_frame(1))

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == []

    def test_is_empty_without_a_frame(self, fake_main_window):
        main_window = fake_main_window(filtered_df=None)

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == []

    def test_is_empty_without_the_outlier_column(self, fake_main_window):
        main_window = fake_main_window(
            filtered_df=filtered_frame(1).drop(columns=[OUTCOL])
        )

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == []

    def test_is_empty_without_an_identification_column(self, fake_main_window):
        main_window = fake_main_window(
            filtered_df=filtered_frame(1, [("p1_a", 0)]).drop(
                columns=["Identification"]
            )
        )

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == []

    def test_accepts_a_lowercase_identification_column(self, fake_main_window):
        df = filtered_frame(1, [("p1_a", 0)]).rename(
            columns={"Identification": "identification"}
        )
        main_window = fake_main_window(filtered_df=df)

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == ["p1_a"]

    def test_ignores_a_short_id_column(self, fake_main_window):
        """Only Identification and its lowercase spelling name a cell."""
        df = filtered_frame(1, [("p1_a", 0)]).rename(
            columns={"Identification": "ID"}
        )
        main_window = fake_main_window(filtered_df=df)

        update_unique_outliers_ids(main_window)

        assert main_window.unique_outliers_ids == []
