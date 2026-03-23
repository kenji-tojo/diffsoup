# python/diffsoupviewer/__init__.py
from __future__ import annotations

import numpy as np
import os

from ._core import __version__
from . import _core

def launch_viewer_with_mlp(
    verts: np.ndarray,
    faces: np.ndarray,
    face_color_lut: np.ndarray,
    W1, b1, W2,b2, W3, b3, enc_freq,
    output_dir: str = "./output",
):
    V, _ = verts.shape
    F, _ = faces.shape
    assert verts.shape == (V, 3) and verts.dtype == np.float32
    assert faces.shape == (F, 3) and faces.dtype == np.int32

    H, W, _ = face_color_lut.shape
    assert face_color_lut.shape == (H, W, 8) and face_color_lut.dtype == np.float32

    face_color_lut0 = face_color_lut[..., 0:4]
    face_color_lut1 = face_color_lut[..., 4:8]
    face_color_lut0 = (face_color_lut0 * 255).clip(0, 255).astype(np.uint8)
    face_color_lut1 = (face_color_lut1 * 255).clip(0, 255).astype(np.uint8)

    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)
    face_color_lut0 = np.ascontiguousarray(face_color_lut0)
    face_color_lut1 = np.ascontiguousarray(face_color_lut1)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    _core.launch_viewer_with_mlp(
        verts, faces,
        face_color_lut0, face_color_lut1,
        W1, b1, W2, b2, W3, b3, enc_freq,
        output_dir
    )
    return

def benchmark_viewer_with_mlp(
    verts,
    faces,
    lut0,
    lut1,
    W1, b1,
    W2, b2,
    W3, b3,
    enc_freq,
    mvps,
    width=1200,
    height=1200,
    warmup=10,
    save_every=0,
    output_dir: str = "./output",
):
    """
    OpenGL benchmark entry point (timing-only, CUDA-comparable).

    Args:
        verts: float32 [V,3]
        faces: int32   [F,3]
        lut0:  uint8   [H,W,4]
        lut1:  uint8   [H,W,4]
        W1,b1,W2,b2,W3,b3: MLP weights (float32)
        enc_freq: float
        mvps: float32 [B,4,4] (column-major)
        width,height: render resolution
        warmup: number of warmup frames (not timed)
        save_every: save every N frames for verification (0 = disable)
    """
    import numpy as np

    # ---- basic sanity ----
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert mvps.ndim == 3 and mvps.shape[1:] == (4, 4)

    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    lut0  = np.ascontiguousarray(lut0,  dtype=np.uint8)
    lut1  = np.ascontiguousarray(lut1,  dtype=np.uint8)

    W1 = np.ascontiguousarray(W1, dtype=np.float32)
    b1 = np.ascontiguousarray(b1, dtype=np.float32)
    W2 = np.ascontiguousarray(W2, dtype=np.float32)
    b2 = np.ascontiguousarray(b2, dtype=np.float32)
    W3 = np.ascontiguousarray(W3, dtype=np.float32)
    b3 = np.ascontiguousarray(b3, dtype=np.float32)

    mvps = np.ascontiguousarray(mvps, dtype=np.float32)

    # ---- dispatch to C++ ----
    _core.benchmark_viewer_with_mlp(
        verts, faces,
        lut0, lut1,
        W1, b1,
        W2, b2,
        W3, b3,
        float(enc_freq),
        mvps,
        int(width),
        int(height),
        int(warmup),
        int(save_every),
        output_dir
    )
