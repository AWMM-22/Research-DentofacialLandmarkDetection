"""Tests for report aggregation and category assignment."""

from __future__ import annotations

from app.schemas import (
    AggregateResult,
    FrameMeasurements,
    FrameQuality,
    FrameResult,
    LandmarkPoint,
    LandmarksModel,
)
from app.services.report_builder import (
    build_aggregate,
    build_quality_summary,
    representative_frame_id,
)


def _frame(
    idx: int,
    fi: float | None,
    status: FrameQuality = FrameQuality.accepted,
) -> FrameResult:
    m = (
        FrameMeasurements(height=400.0, width=500.0, facial_index=fi)
        if fi is not None
        else None
    )
    return FrameResult(
        frame_id=f"f_{idx:06d}",
        frame_index=idx,
        timestamp_ms=idx * 500,
        status=status,
        landmarks=LandmarksModel(),
        measurements=m,
    )


class TestBuildAggregate:
    def test_no_accepted_frames(self):
        frames = [_frame(0, None, FrameQuality.rejected), _frame(1, 90.0, FrameQuality.flagged)]
        agg = build_aggregate(frames)
        assert agg.facial_index is None
        assert agg.accepted_frame_count == 0
        assert agg.category == "Outside the specified reference ranges"

    def test_median_of_accepted_only(self):
        frames = [
            _frame(0, 88.0),
            _frame(1, 200.0, FrameQuality.rejected),  # outlier must not affect
            _frame(2, 89.0),
            _frame(3, 87.0),
        ]
        agg = build_aggregate(frames)
        assert agg.facial_index == 88.0
        assert agg.accepted_frame_count == 3

    def test_category_from_unrounded(self):
        frames = [_frame(0, 84.9996), _frame(1, 84.9998)]
        agg = build_aggregate(frames)
        # Median ~84.9997 -> Euryprosopic even though rounds to 85.0.
        assert agg.category == "Euryprosopic"

    def test_outside_range_not_forced(self):
        frames = [_frame(0, 78.0), _frame(1, 79.0)]
        agg = build_aggregate(frames)
        assert agg.category == "Outside the specified reference ranges"


class TestQualitySummary:
    def test_counts_and_reasons(self):
        f0 = _frame(0, 88.0)
        f1 = _frame(1, None, FrameQuality.rejected)
        f1.rejection = type(f1.rejection)(reason="blur:30.0") if f1.rejection else None
        # Simpler: construct rejection properly
        from app.schemas import FrameRejection

        f1 = FrameResult(
            frame_id="f_000001",
            frame_index=1,
            timestamp_ms=500,
            status=FrameQuality.rejected,
            landmarks=LandmarksModel(),
            measurements=None,
            rejection=FrameRejection(reason="blur"),
        )
        f2 = _frame(2, 90.0, FrameQuality.flagged)
        s = build_quality_summary([f0, f1, f2])
        assert s.sampled == 3
        assert s.accepted == 1
        assert s.rejected == 1
        assert s.flagged == 1
        assert s.rejection_reasons.get("blur") == 1


class TestRepresentativeFrame:
    def test_closest_to_median(self):
        frames = [_frame(0, 85.0), _frame(1, 88.0), _frame(2, 92.0)]
        rid = representative_frame_id(frames)
        assert rid == "f_000001"  # 88.0 is the median

    def test_none_when_no_usable(self):
        frames = [_frame(0, None, FrameQuality.rejected)]
        assert representative_frame_id(frames) is None
