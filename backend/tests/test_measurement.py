"""Unit tests for the measurement engine (AGENT_PLAN.md section 4).

Covers distances, facial height/width, FI computation, invalid widths,
interval boundaries, unrounded classification, and median aggregation.
"""

from __future__ import annotations

import math

import pytest

from app.services.measurement import (
    Landmarks,
    Measurement,
    MeasurementError,
    Point,
    aggregate_fi,
    bizygomatic_width,
    classify,
    distance,
    facial_height,
    facial_index,
    measure,
)


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------
class TestDistance:
    def test_axis_aligned(self):
        assert distance(Point(0, 0), Point(3, 4)) == pytest.approx(5.0)

    def test_zero(self):
        assert distance(Point(10, 20), Point(10, 20)) == 0.0

    def test_negative_coords(self):
        assert distance(Point(-1, -1), Point(2, 3)) == pytest.approx(5.0)

    def test_symmetry(self):
        a, b = Point(1, 2), Point(4, 6)
        assert distance(a, b) == distance(b, a)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            Point(float("nan"), 0.0)
        with pytest.raises(ValueError):
            Point(0.0, float("inf"))


# ---------------------------------------------------------------------------
# H / W / FI
# ---------------------------------------------------------------------------
def _lm() -> Landmarks:
    """Synthetic face: H=400, W=500, FI=80.0 exactly."""
    return Landmarks(
        nasion=Point(500, 100),
        menton=Point(500, 500),   # H = 400
        zygion_right=Point(250, 300),
        zygion_left=Point(750, 300),  # W = 500
    )


class TestFacialIndex:
    def test_height_and_width(self):
        lm = _lm()
        assert facial_height(lm) == pytest.approx(400.0)
        assert bizygomatic_width(lm) == pytest.approx(500.0)

    def test_fi_value(self):
        assert facial_index(400.0, 500.0) == pytest.approx(80.0)

    def test_measure_helper(self):
        m = measure(_lm())
        assert m.height == pytest.approx(400.0)
        assert m.width == pytest.approx(500.0)
        assert m.facial_index == pytest.approx(80.0)

    def test_zero_width_rejected(self):
        with pytest.raises(MeasurementError):
            facial_index(400.0, 0.0)

    def test_negative_width_rejected(self):
        with pytest.raises(MeasurementError):
            facial_index(400.0, -10.0)

    def test_nan_width_rejected(self):
        with pytest.raises(MeasurementError):
            facial_index(400.0, float("nan"))

    def test_negative_height_rejected(self):
        with pytest.raises(MeasurementError):
            facial_index(-1.0, 500.0)

    def test_identical_zygions_zero_width(self):
        lm = Landmarks(
            nasion=Point(0, 0),
            menton=Point(0, 100),
            zygion_right=Point(50, 50),
            zygion_left=Point(50, 50),
        )
        with pytest.raises(MeasurementError):
            measure(lm)


# ---------------------------------------------------------------------------
# classify — half-open intervals [lower, upper)
# ---------------------------------------------------------------------------
class TestClassify:
    @pytest.mark.parametrize(
        "fi,expected",
        [
            (80.0, "Euryprosopic"),
            (84.9999, "Euryprosopic"),
            (85.0, "Mesoprosopic"),   # 85 belongs to the next band
            (89.9999, "Mesoprosopic"),
            (90.0, "Leptoprosopic"),  # 90 belongs to the next band
            (94.9999, "Leptoprosopic"),
            (95.0, "Outside the specified reference ranges"),
            (79.9999, "Outside the specified reference ranges"),
            (100.0, "Outside the specified reference ranges"),
        ],
    )
    def test_boundaries(self, fi, expected):
        assert classify(fi) == expected

    def test_unrounded_matters(self):
        # 84.9996 rounds to 85.0 for display but must classify as Euryprosopic.
        assert classify(84.9996) == "Euryprosopic"
        assert classify(round(84.9996, 1)) == "Mesoprosopic"

    def test_nan_rejected(self):
        with pytest.raises(MeasurementError):
            classify(float("nan"))


# ---------------------------------------------------------------------------
# aggregate_fi — median
# ---------------------------------------------------------------------------
class TestAggregate:
    def test_empty(self):
        assert aggregate_fi([]) is None

    def test_single(self):
        assert aggregate_fi([90.0]) == 90.0

    def test_odd(self):
        assert aggregate_fi([80.0, 90.0, 100.0]) == 90.0

    def test_even_average_of_two_middles(self):
        assert aggregate_fi([80.0, 90.0, 100.0, 110.0]) == 95.0

    def test_robust_to_outlier(self):
        # Median resists a single wild frame; mean would not.
        fis = [88.0, 88.5, 89.0, 200.0]
        assert aggregate_fi(fis) == 88.75

    def test_full_precision_preserved(self):
        v = aggregate_fi([87.123456789])
        assert v == 87.123456789
