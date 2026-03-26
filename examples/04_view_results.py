# examples/04_view_results.py
# Load a trained DiffSoup checkpoint and launch the interactive viewer.
#
# Usage:
#   python examples/04_view_results.py --ckpt results/02_synthetic/lego/final_params.pt
#   python examples/04_view_results.py --ckpt results/01_mip360/kitchen/final_params.pt
#   python examples/04_view_results.py --ckpt ... --up 0 1 0
#
# Dependencies (beyond diffsoupviewer):
#   pip install numpy torch

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import diffsoupviewer


# ── Helpers ──────────────────────────────────────────────────────────


def level_size(L: int) -> int:
    """Number of texels per face at subdivision level L."""
    if L == 0:
        return 3
    a = (1 << (L - 1)) + 1
    b = (1 << L) + 1
    return a * b


def pack_face_color_lut(
    feat_acc: np.ndarray,
    alpha_acc: np.ndarray,
    num_faces: int,
    level: int,
) -> np.ndarray:
    """Pack per-texel features + alpha into [H, W, 8] float32 LUT."""
    S = level_size(level)
    N = num_faces * S

    if feat_acc.ndim == 3:
        feat_acc = feat_acc.reshape(-1, feat_acc.shape[-1])
    if alpha_acc.ndim == 3:
        alpha_acc = alpha_acc.reshape(-1, alpha_acc.shape[-1])

    assert feat_acc.shape[0] >= N and alpha_acc.shape[0] >= N
    assert feat_acc.shape[-1] == 7, f"expected feat_dim=7, got {feat_acc.shape[-1]}"

    lut_flat = np.concatenate([feat_acc[:N], alpha_acc[:N]], axis=-1)

    tex_W = min(4096, N)
    tex_H = math.ceil(N / tex_W)
    padded = np.zeros((tex_H * tex_W, 8), dtype=np.float32)
    padded[:N] = lut_flat
    return padded.reshape(tex_H, tex_W, 8)


def extract_mlp_weights(state_dict: dict):
    """Pull W1,b1,W2,b2,W3,b3 from the ColorMLP state dict."""
    weights, biases = [], []
    for k in state_dict:
        t = state_dict[k].detach().cpu().numpy().astype(np.float32)
        if "weight" in k:
            weights.append(t)
        elif "bias" in k:
            biases.append(t)

    if len(weights) < 3 or len(biases) < 3:
        raise ValueError(
            f"Expected ≥3 linear layers, found {len(weights)} weights / "
            f"{len(biases)} biases.  Keys: {list(state_dict.keys())}"
        )

    W1, W2, W3 = weights[0], weights[1], weights[2]
    b1, b2, b3 = biases[0], biases[1], biases[2]
    assert W1.shape == (16, 16) and W2.shape == (16, 16) and W3.shape == (3, 16)
    return W1, b1, W2, b2, W3, b3


def detect_up(ckpt: dict) -> Tuple[float, float, float]:
    """Infer the world up-direction from checkpoint metadata.

    - Explicit 'up' key     → use it directly.
    - 'flip_z' present      → COLMAP / MipNeRF-360 → (0, -1, 0).
    - Otherwise             → NeRF-synthetic        → (0, 0, 1).
    """
    if "up" in ckpt:
        u = ckpt["up"]
        return (float(u[0]), float(u[1]), float(u[2]))
    if "flip_z" in ckpt:
        return (0.0, -1.0, 0.0)
    return (0.0, 0.0, 1.0)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive DiffSoup viewer from a checkpoint.",
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to final_params.pt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for screenshots (default: beside ckpt)")
    parser.add_argument("--up", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="World up direction, e.g. --up 0 0 1. "
                             "Auto-detected from checkpoint if omitted.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = args.output_dir or str(ckpt_path.parent / "viewer_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    verts     = ckpt["V"].numpy().astype(np.float32)
    faces     = ckpt["F"].numpy().astype(np.int32)
    feat_acc  = ckpt["feat_acc"].numpy().astype(np.float32)
    alpha_acc = ckpt["alpha_acc"].numpy().astype(np.float32)
    Rmax      = int(ckpt["Rmax"])
    feat_dim  = int(ckpt["feat_dim"])
    num_faces = faces.shape[0]

    print(f"[mesh]  {verts.shape[0]:,} verts, {num_faces:,} faces")
    print(f"[level] Rmax={Rmax}  texels/face={level_size(Rmax)}")
    print(f"[feat]  dim={feat_dim}  feat_acc={feat_acc.shape}  alpha_acc={alpha_acc.shape}")

    up = tuple(args.up) if args.up else detect_up(ckpt)
    print(f"[cam]   up={up}")

    face_color_lut = pack_face_color_lut(feat_acc, alpha_acc, num_faces, Rmax)
    print(f"[lut]   texture {face_color_lut.shape[1]}x{face_color_lut.shape[0]}")

    if "color_mlp" not in ckpt:
        raise KeyError("Checkpoint missing 'color_mlp'.")

    W1, b1, W2, b2, W3, b3 = extract_mlp_weights(ckpt["color_mlp"])
    print(f"[mlp]   W1={W1.shape} W2={W2.shape} W3={W3.shape}")

    print("[viewer] launching …")
    diffsoupviewer.launch_viewer(
        verts=verts, faces=faces, face_color_lut=face_color_lut,
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        output_dir=output_dir, up=up,
    )


if __name__ == "__main__":
    main()
