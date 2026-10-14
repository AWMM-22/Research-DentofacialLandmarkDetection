"""Live webcam demo — opens the default camera and shows FI on screen.

Usage:
    python webcam_demo.py

Keys:
    q / ESC  quit
    space    reset the temporal buffer
    d        toggle debug overlay (face box + rejection reasons)

Looser quality thresholds than the batch pipeline so a typical webcam
works. Still rejects clearly bad pose/blur for measurement.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services import face_detector  # noqa: E402
from app.services.landmark_validation import validate_frame  # noqa: E402
from app.services.measurement import Landmarks, Point, classify, measure  # noqa: E402
from app.services.quality_gate import QualityConfig, evaluate_frame  # noqa: E402
from app.services.temporal_filter import filter_landmark_series  # noqa: E402

WINDOW = "Facial Index — live (q quit | space reset | d debug)"
HISTORY_LEN = 9

# Looser gates for live webcam demo
QCFG = QualityConfig(
    min_laplacian_var=15.0,   # webcams are often soft
    min_frame_side=120,
    max_yaw_deg=30.0,
    max_pitch_deg=30.0,
    max_roll_deg=25.0,
)

DOT_COLORS = {
    "nasion": (60, 76, 231),
    "menton": (34, 126, 230),
    "zygion_right": (219, 152, 52),
    "zygion_left": (182, 89, 155),
}


def put(img, text, x, y, scale=0.55, color=(240, 240, 240), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open the default camera.")
        sys.exit(1)

    # Bump camera resolution if the driver allows it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Loading Face Landmarker…")
    _ = face_detector.model_version()
    print("Camera ready. q=quit, space=reset, d=debug")

    history: deque[list[Landmarks]] = deque(maxlen=HISTORY_LEN)
    n_total = 0
    n_accepted = 0
    last_fi: float | None = None
    last_category = ""
    debug = True
    analysis_interval = 0.2  # ~5 fps analysis
    next_analysis = 0.0
    last_reasons: list[str] = []
    last_status = "starting…"
    last_pose = "yaw --  pitch --  roll --"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        h_img, w_img = frame.shape[:2]
        now = time.time()

        lm_raw = None
        if now >= next_analysis:
            next_analysis = now + analysis_interval
            n_total += 1
            ts_ms = int(now * 1000)

            lm_raw = face_detector.detect(frame, ts_ms)
            if lm_raw is None:
                history.append([])
                last_reasons = ["no_face_detected"]
                last_status = "NO FACE — move into frame / improve lighting"
                last_pose = "yaw --  pitch --  roll --"
            else:
                verdict = evaluate_frame(frame, lm_raw.all_landmarks_xy, QCFG)
                if verdict.pose is not None:
                    p = verdict.pose
                    last_pose = f"yaw {p.yaw_deg:+.0f}  pitch {p.pitch_deg:+.0f}  roll {p.roll_deg:+.0f}"
                if not verdict.passed:
                    history.append([])
                    last_reasons = verdict.reasons
                    last_status = "REJECTED: " + "; ".join(verdict.reasons[:3])
                else:
                    lm = Landmarks(
                        nasion=Point(*lm_raw.nasion),
                        menton=Point(*lm_raw.menton),
                        zygion_right=Point(*lm_raw.zygion_right),
                        zygion_left=Point(*lm_raw.zygion_left),
                    )
                    v = validate_frame(lm, face_box=lm_raw.face_box)
                    if not v.ok:
                        history.append([])
                        last_reasons = v.errors
                        last_status = "GEOMETRY: " + "; ".join(v.errors[:3])
                    else:
                        history.append([lm])
                        n_accepted += 1
                        last_reasons = []
                        last_status = "OK"

        # Always draw face box + raw candidate points when we have a face
        # this frame (even if rejected) so the user can see detection works.
        if lm_raw is not None:
            bx, by, bw, bh = lm_raw.face_box
            color = (0, 200, 0) if last_status == "OK" else (0, 165, 255)
            cv2.rectangle(frame, (int(bx), int(by)),
                          (int(bx + bw), int(by + bh)), color, 2)
            raw = {
                "nasion": lm_raw.nasion,
                "menton": lm_raw.menton,
                "zygion_right": lm_raw.zygion_right,
                "zygion_left": lm_raw.zygion_left,
            }
            for name, (x, y) in raw.items():
                c = DOT_COLORS[name]
                cv2.circle(frame, (int(x), int(y)), 6, c, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 1, cv2.LINE_AA)
            # Raw H/W lines (candidates)
            cv2.line(frame, (int(lm_raw.nasion[0]), int(lm_raw.nasion[1])),
                     (int(lm_raw.menton[0]), int(lm_raw.menton[1])),
                     (0, 220, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (int(lm_raw.zygion_right[0]), int(lm_raw.zygion_right[1])),
                     (int(lm_raw.zygion_left[0]), int(lm_raw.zygion_left[1])),
                     (255, 200, 80), 1, cv2.LINE_AA)

        # Stabilised (median-filtered) landmarks for FI.
        series = [h[0] for h in history if h]
        lm_show = None
        if series:
            n = len(series)
            window = n if n % 2 == 1 else n - 1
            window = max(1, min(window, 5))
            lm_show = filter_landmark_series(series, window=window)[-1]

        fi_disp: float | None = None
        cat_disp = ""
        if lm_show is not None:
            try:
                m = measure(lm_show)
                fi_disp = m.facial_index
                cat_disp = classify(fi_disp)
                last_fi, last_category = fi_disp, cat_disp
                # Draw stabilised points larger / brighter
                for name, pt in {
                    "nasion": lm_show.nasion,
                    "menton": lm_show.menton,
                    "zygion_right": lm_show.zygion_right,
                    "zygion_left": lm_show.zygion_left,
                }.items():
                    c = DOT_COLORS[name]
                    cv2.circle(frame, (int(pt.x), int(pt.y)), 10, c, 3, cv2.LINE_AA)
            except Exception:
                fi_disp, cat_disp = last_fi, last_category
        else:
            fi_disp, cat_disp = last_fi, last_category

        # ---- HUD ----
        scale = max(0.55, w_img / 1000)
        pad = 14
        y = 30
        put(frame, "FACIAL INDEX (live prototype)", pad, y, scale * 1.1, (255, 255, 255), 2)
        y += int(30 * scale)
        if fi_disp is not None:
            put(frame, f"FI = {fi_disp:.1f}", pad, y, scale * 1.6, (80, 255, 120), 3)
            y += int(36 * scale)
            put(frame, cat_disp, pad, y, scale * 1.2, (80, 200, 255), 2)
            y += int(30 * scale)
        else:
            put(frame, "FI = --", pad, y, scale * 1.6, (180, 180, 180), 2)
            y += int(36 * scale)

        status_color = (80, 255, 120) if last_status == "OK" else (0, 165, 255)
        put(frame, last_status[:80], pad, y, scale, status_color, 1)
        y += int(24 * scale)
        put(frame, last_pose, pad, y, scale, (220, 220, 220))
        y += int(24 * scale)
        put(frame, f"accepted {n_accepted}/{n_total}   history {len(series)}", pad, y, scale, (200, 200, 200))
        y += int(24 * scale)
        blur = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        put(frame, f"blur={blur:.0f} (min {QCFG.min_laplacian_var:.0f})", pad, y, scale, (200, 200, 200))
        y += int(24 * scale)
        put(frame, "q quit | space reset | d debug", pad, y, scale * 0.9, (150, 150, 150))

        if debug and last_reasons:
            yy = h_img - 20
            put(frame, "debug: " + " | ".join(last_reasons[:4]), pad, yy, scale * 0.85, (0, 200, 255))

        put(frame, "MediaPipe candidates only — not clinical", w_img - 320, h_img - 12,
            0.45, (160, 160, 160))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            history.clear()
            last_fi = None
            last_category = ""
            n_accepted = n_total = 0
            last_status = "buffer reset"
            print("Buffer reset.")
        if key == ord("d"):
            debug = not debug
            print("debug:", debug)

    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
