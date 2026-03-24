# python/diffsoupviewer/__init__.py
"""Interactive viewer for DiffSoup meshes with per-triangle LUT + MLP shading."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ._core import __version__
from . import _core


def launch_viewer(
    verts: np.ndarray,
    faces: np.ndarray,
    face_color_lut: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
    output_dir: str = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
) -> None:
    """Open an interactive viewer window (blocks until closed).

    Args:
        verts: float32 [V, 3] — vertex positions.
        faces: int32   [F, 3] — triangle indices.
        face_color_lut: float32 [H, W, 8] — per-triangle colour LUT
            (channels 0–3 → buffer A, channels 4–7 → buffer B).
        W1, b1, W2, b2, W3, b3: MLP weight matrices (float32).
        output_dir: directory for screenshots.
        up: world-space up direction as (x, y, z).
            Common choices: (0,0,1) for NeRF-synthetic, (0,-1,0) for COLMAP.
            Can also be changed at runtime via the GUI.
    """
    V, _ = verts.shape
    F, _ = faces.shape
    assert verts.shape == (V, 3) and verts.dtype == np.float32
    assert faces.shape == (F, 3) and faces.dtype == np.int32

    H, W, _ = face_color_lut.shape
    assert face_color_lut.shape == (H, W, 8) and face_color_lut.dtype == np.float32

    lut0 = (face_color_lut[..., :4] * 255).clip(0, 255).astype(np.uint8)
    lut1 = (face_color_lut[..., 4:] * 255).clip(0, 255).astype(np.uint8)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    _core.launch_viewer_with_mlp(
        np.ascontiguousarray(verts),
        np.ascontiguousarray(faces),
        np.ascontiguousarray(lut0),
        np.ascontiguousarray(lut1),
        np.ascontiguousarray(W1),
        np.ascontiguousarray(b1),
        np.ascontiguousarray(W2),
        np.ascontiguousarray(b2),
        np.ascontiguousarray(W3),
        np.ascontiguousarray(b3),
        output_dir,
        np.array(up, dtype=np.float32),
    )


def benchmark(
    verts: np.ndarray,
    faces: np.ndarray,
    lut0: np.ndarray,
    lut1: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
    mvps: np.ndarray,
    width: int = 1200,
    height: int = 1200,
    warmup: int = 10,
    save_every: int = 0,
    output_dir: str = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
) -> None:
    """Run a headless rendering benchmark.

    Args:
        verts: float32 [V, 3].
        faces: int32   [F, 3].
        lut0:  uint8   [H, W, 4] — colour buffer A.
        lut1:  uint8   [H, W, 4] — colour buffer B.
        W1, b1, W2, b2, W3, b3: MLP weights (float32).
        mvps: float32 [B, 4, 4] column-major MVP matrices.
        width, height: render resolution.
        warmup: frames to skip before timing.
        save_every: save a screenshot every N frames (0 = disable).
        output_dir: directory for logs and screenshots.
        up: world-space up direction (see launch_viewer).
    """
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert mvps.ndim == 3 and mvps.shape[1:] == (4, 4)

    def c(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=dtype)

    _core.benchmark_viewer_with_mlp(
        c(verts, np.float32),
        c(faces, np.int32),
        c(lut0, np.uint8),
        c(lut1, np.uint8),
        c(W1, np.float32),
        c(b1, np.float32),
        c(W2, np.float32),
        c(b2, np.float32),
        c(W3, np.float32),
        c(b3, np.float32),
        c(mvps, np.float32),
        int(width),
        int(height),
        int(warmup),
        int(save_every),
        output_dir,
        np.array(up, dtype=np.float32),
    )
