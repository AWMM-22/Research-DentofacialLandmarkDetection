"""Tests for landmark validation and the temporal median filter."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.landmark_validation import validate_frame
from app.services.measurement import Landmarks, Point
from app.services.temporal_filter import filter_landmark_series, median_filter_series


def _good_lm() -> Landmarks:
    return Landmarks(
        nasion=Point(320, 180),
        menton=Point(320, 480),
        zygion_right=Point(180, 330),
        zygion_left=Point(460, 330),
    )


GOOD_BOX = (160.0, 150.0, 320.0, 360.0)  # x, y, w, h


class TestValidateFrame:
    def test_missing_landmarks(self):
        v = validate_frame(None)
        assert not v.ok
        assert "missing_landmarks" in v.errors

    def test_good_geometry(self):
        v = validate_frame(_good_lm(), face_box=GOOD_BOX)
        assert v.ok, v.errors

    def test_zero_width(self):
        lm = Landmarks(
            nasion=Point(320, 180),
            menton=Point(320, 480),
            zygion_right=Point(200, 330),
            zygion_left=Point(200, 330),
        )
        v = validate_frame(lm)
        assert not v.ok
        assert "width_not_positive" in v.errors

    def test_menton_above_nasion_rejected(self):
        lm = Landmarks(
            nasion=Point(320, 400),
            menton=Point(320, 180),
            zygion_right=Point(180, 300),
            zygion_left=Point(460, 300),
        )
        v = validate_frame(lm, face_box=GOOD_BOX)
        assert not v.ok
        assert "menton_above_nasion" in v.errors

    def test_width_too_small_vs_face(self):
        # W = 40 px on a 320 px-wide face box -> below MIN_WIDTH_TO_FACE_RATIO.
        lm = Landmarks(
            nasion=Point(320, 180),
            menton=Point(320, 480),
            zygion_right=Point(300, 330),
            zygion_left=Point(340, 330),
        )
        v = validate_frame(lm, face_box=GOOD_BOX)
        assert not v.ok
        assert any("width_too_small_vs_face" in e for e in v.errors)

    def test_h_over_w_ratio_sanity(self):
        # Very short face relative to width -> ratio too small.
        lm = Landmarks(
            nasion=Point(320, 300),
            menton=Point(320, 320),   # H = 20
            zygion_right=Point(160, 310),
            zygion_left=Point(480, 310),  # W = 320, ratio = 0.0625
        )
        v = validate_frame(lm, face_box=GOOD_BOX)
        assert not v.ok


class TestMedianFilter:
    def test_window_must_be_odd(self):
        with pytest.raises(ValueError):
            median_filter_series(np.zeros((5, 2)), 4)

    def test_window_one_is_identity(self):
        arr = np.random.default_rng(0).normal(size=(10, 2))
        out = median_filter_series(arr, 1)
        np.testing.assert_array_equal(out, arr)

    def test_reduces_isolated_spike(self):
        arr = np.array([[0.0, 0.0]] * 5 + [[100.0, 100.0]] + [[0.0, 0.0]] * 5)
        out = median_filter_series(arr, 3)
        # Spike should be smoothed away.
        assert out[5][0] == pytest.approx(0.0)

    def test_preserves_constant(self):
        arr = np.full((7, 2), 42.0)
        out = median_filter_series(arr, 3)
        np.testing.assert_allclose(out, 42.0)

    def test_landmarks_series(self):
        series = [
            Landmarks(
                nasion=Point(320 + i, 180),
                menton=Point(320 + i, 480),
                zygion_right=Point(180 + i, 330),
                zygion_left=Point(460 + i, 330),
            )
            for i in range(7)
        ]
        series[3] = Landmarks(
            nasion=Point(999, 180),
            menton=Point(999, 480),
            zygion_right=Point(999, 330),
            zygion_left=Point(999, 330),
        )
        out = filter_landmark_series(series, 3)
        assert len(out) == 7
        # The spike at index 3 should be replaced by the median of
        # [322, 999, 324] = 324.0.
        assert out[3].nasion.x == pytest.approx(324.0)
