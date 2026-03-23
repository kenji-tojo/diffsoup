# examples/04_view_results.py
# Load a trained DiffSoup checkpoint and launch the interactive viewer.
#
# Usage:
#   python examples/04_view_results.py --ckpt results/02_synthetic/lego/final_params.pt
#
# Dependencies:
#   pip install diffsoupviewer numpy torch

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import diffsoupviewer


# ── Helpers ──────────────────────────────────────────────────────────


def level_size(L: int) -> int:
    """Number of texels per face at subdivision level L (matches the GLSL shader)."""
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
    """Pack per-texel features + alpha into the [H, W, 8] float32 LUT
    expected by :func:`diffsoupviewer.launch_viewer`.

    Args:
        feat_acc:  float32 [F, S, feat_dim] or [F*S, feat_dim]  (sigmoid'd)
        alpha_acc: float32 [F, S, 1] or [F*S, 1]                (sigmoid'd)
        num_faces: number of triangles F
        level:     finest subdivision level (Rmax)

    Returns:
        float32 [tex_H, tex_W, 8]
    """
    S = level_size(level)
    N = num_faces * S

    # Flatten [F, S, dim] → [F*S, dim] if needed.
    if feat_acc.ndim == 3:
        feat_acc = feat_acc.reshape(-1, feat_acc.shape[-1])
    if alpha_acc.ndim == 3:
        alpha_acc = alpha_acc.reshape(-1, alpha_acc.shape[-1])

    assert feat_acc.shape[0] >= N and alpha_acc.shape[0] >= N

    feat_dim = feat_acc.shape[-1]
    assert feat_dim == 7, f"expected feat_dim=7, got {feat_dim}"

    # Channels 0-3  → buffer A (feat[:4])
    # Channels 4-6  → buffer B rgb (feat[4:7])
    # Channel  7    → buffer B alpha (opacity mask)
    lut_flat = np.concatenate([feat_acc[:N], alpha_acc[:N]], axis=-1)  # [N, 8]

    # Choose a 2D packing — width 4096 is a safe GPU texture limit.
    tex_W = min(4096, N)
    tex_H = math.ceil(N / tex_W)
    padded = np.zeros((tex_H * tex_W, 8), dtype=np.float32)
    padded[:N] = lut_flat
    return padded.reshape(tex_H, tex_W, 8)


def extract_mlp_weights(state_dict: dict):
    """Pull W1,b1,W2,b2,W3,b3 from the ColorMLP state dict.

    Works regardless of key naming convention — we simply collect
    (weight, bias) pairs in parameter order and match by shape.
    """
    # Collect all weight and bias tensors in insertion order.
    weights, biases = [], []
    keys = list(state_dict.keys())
    for k in keys:
        t = state_dict[k].detach().cpu().numpy().astype(np.float32)
        if "weight" in k:
            weights.append(t)
        elif "bias" in k:
            biases.append(t)

    if len(weights) < 3 or len(biases) < 3:
        raise ValueError(
            f"Expected ≥3 linear layers, found {len(weights)} weights and "
            f"{len(biases)} biases.  State dict keys: {keys}"
        )

    W1, W2, W3 = weights[0], weights[1], weights[2]
    b1, b2, b3 = biases[0], biases[1], biases[2]

    assert W1.shape == (16, 16), f"W1 shape {W1.shape}"
    assert W2.shape == (16, 16), f"W2 shape {W2.shape}"
    assert W3.shape == (3, 16),  f"W3 shape {W3.shape}"

    return W1, b1, W2, b2, W3, b3


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive DiffSoup viewer from a checkpoint.",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to final_params.pt",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for viewer screenshots (default: same as checkpoint)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = args.output_dir or str(ckpt_path.parent / "viewer_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    verts = ckpt["V"].numpy().astype(np.float32)         # [V, 3]
    faces = ckpt["F"].numpy().astype(np.int32)            # [F, 3]
    feat_acc  = ckpt["feat_acc"].numpy().astype(np.float32)
    alpha_acc = ckpt["alpha_acc"].numpy().astype(np.float32)

    Rmax = int(ckpt["Rmax"])
    feat_dim = int(ckpt["feat_dim"])
    num_faces = faces.shape[0]

    print(f"[mesh]  {verts.shape[0]:,} verts, {num_faces:,} faces")
    print(f"[level] Rmax={Rmax}  texels/face={level_size(Rmax)}")
    print(f"[feat]  dim={feat_dim}  feat_acc={feat_acc.shape}  alpha_acc={alpha_acc.shape}")

    # ── Build face-colour LUT ────────────────────────────────────────

    face_color_lut = pack_face_color_lut(feat_acc, alpha_acc, num_faces, Rmax)
    print(f"[lut]   texture {face_color_lut.shape[1]}×{face_color_lut.shape[0]}")

    # ── Extract MLP weights ──────────────────────────────────────────

    if "color_mlp" not in ckpt:
        raise KeyError(
            "Checkpoint is missing 'color_mlp'.  Re-run training with the "
            "updated saving code that includes color_mlp state_dict."
        )

    W1, b1, W2, b2, W3, b3 = extract_mlp_weights(ckpt["color_mlp"])
    print(f"[mlp]   W1={W1.shape} W2={W2.shape} W3={W3.shape}")

    # ── Launch viewer ────────────────────────────────────────────────

    print(f"[viewer] launching interactive viewer …")
    diffsoupviewer.launch_viewer(
        verts=verts,
        faces=faces,
        face_color_lut=face_color_lut,
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
