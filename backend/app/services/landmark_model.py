"""Custom four-channel heatmap model (PyTorch) — inference stub.

ARCHITECTURE.md section 3 / AGENT_PLAN.md section 5: target model is a
heatmap-regression network that predicts four spatial heatmaps (N', Me',
Zy_r, Zy_l). Training requires an expert-annotated dataset that does not
exist yet; this module provides the inference interface and a small CNN
scaffold so the pipeline can swap it in once weights are trained.

Until a trained checkpoint is supplied, `is_available()` returns False and
the pipeline uses the MediaPipe baseline candidates with explicit warnings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import MODELS_DIR

HEATMAP_CHECKPOINT = MODELS_DIR / "heatmap_model.pt"
NUM_LANDMARKS = 4  # nasion, menton, zygion_right, zygion_left
HEATMAP_SIZE = 128  # output heatmap spatial size


class HeatmapNet(nn.Module):
    """Small encoder-decoder CNN producing 4 heatmaps.

    Kept deliberately simple; replace the encoder with a licensed
    pretrained backbone after verifying its weight license
    (AGENT_PLAN.md section 5).
    """

    def __init__(self, in_channels: int = 3, num_landmarks: int = NUM_LANDMARKS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_landmarks, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def is_available() -> bool:
    return HEATMAP_CHECKPOINT.exists()


def load_model(checkpoint: Optional[Path] = None) -> Optional[HeatmapNet]:
    """Load a trained checkpoint. Returns None when unavailable."""
    path = checkpoint or HEATMAP_CHECKPOINT
    if not path.exists():
        return None
    net = HeatmapNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def soft_argmax(heatmap: np.ndarray) -> tuple[float, float]:
    """Convert a single (H, W) heatmap to (x, y) via softmax-weighted average.

    Returns coordinates in heatmap pixel space. Callers must scale back to
    the original frame using the recorded crop/resize transform.
    """
    h, w = heatmap.shape
    flat = heatmap.reshape(-1)
    flat = flat - flat.max()
    exp = np.exp(flat)
    prob = exp / exp.sum()
    ys, xs = np.divmod(prob.argmax(), w)
    # Optional local refinement: weighted mean around the peak.
    peak_y, peak_x = int(ys), int(xs)
    r = 2
    y0, y1 = max(0, peak_y - r), min(h, peak_y + r + 1)
    x0, x1 = max(0, peak_x - r), min(w, peak_x + r + 1)
    local = heatmap[y0:y1, x0:x1]
    if local.sum() > 0:
        ly, lx = np.mgrid[y0:y1, x0:x1]
        wsum = local.sum()
        return (float((lx * local).sum() / wsum), float((ly * local).sum() / wsum))
    return (float(peak_x), float(peak_y))


def predict(
    net: HeatmapNet,
    face_crop_bgr: np.ndarray,
) -> Optional[list[tuple[float, float]]]:
    """Run inference on a BGR face crop. Returns 4 (x, y) pairs in crop
    pixel space, or None on failure.
    """
    import cv2

    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (HEATMAP_SIZE, HEATMAP_SIZE))
    tensor = (
        torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    )
    with torch.no_grad():
        heatmaps = net(tensor).squeeze(0).numpy()  # (4, H, W)
    crop_h, crop_w = face_crop_bgr.shape[:2]
    points: list[tuple[float, float]] = []
    for i in range(NUM_LANDMARKS):
        hx, hy = soft_argmax(heatmaps[i])
        points.append((hx / HEATMAP_SIZE * crop_w, hy / HEATMAP_SIZE * crop_h))
    return points
