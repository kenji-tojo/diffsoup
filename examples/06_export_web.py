# examples/06_export_web.py
# Export a trained DiffSoup checkpoint to web-viewer-ready assets.
#
# Produces (inside <output_dir>/<scene>/):
#   mesh.ply          — binary PLY mesh (vertices + triangle faces)
#   lut0.png          — per-triangle colour LUT buffer A (RGBA8)
#   lut1.png          — per-triangle colour LUT buffer B (RGBA8)
#   mlp_weights.json  — MLP weight matrices and biases (flat row-major)
#   meta.json         — viewer metadata (up direction, subdivision level, …)
#
# Also updates <output_dir>/models.json so the web viewer can discover
# all exported scenes.
#
# Usage:
#   python examples/06_export_web.py \
#       --ckpt results/01_mip360/kitchen/final_params.pt
#
#   python examples/06_export_web.py \
#       --ckpt results/02_synthetic/lego/final_params.pt
#
# The scene name is inferred from the checkpoint path (the parent
# directory name).  Use --scene to override if needed.
#
# Dependencies:
#   pip install numpy torch Pillow

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image


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
    """Infer the world up-direction from checkpoint metadata."""
    if "up" in ckpt:
        u = ckpt["up"]
        return (float(u[0]), float(u[1]), float(u[2]))
    if "flip_z" in ckpt:
        return (0.0, -1.0, 0.0)
    return (0.0, 0.0, 1.0)


# ── Writers ──────────────────────────────────────────────────────────


def write_ply(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary little-endian PLY with vertices and triangle faces."""
    V, F = verts.shape[0], faces.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {V}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {F}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.astype(np.float32).tobytes())
        for i in range(F):
            f.write(struct.pack("<B", 3))
            f.write(faces[i].astype(np.int32).tobytes())


def save_lut_png(path: str, lut_uint8: np.ndarray) -> None:
    """Save an RGBA8 LUT array as PNG."""
    Image.fromarray(lut_uint8, "RGBA").save(path)


def save_mlp_json(
    path: str,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    W3: np.ndarray, b3: np.ndarray,
) -> None:
    """Save MLP weights as flat row-major arrays in JSON."""
    data = {
        "W1": W1.ravel().tolist(),
        "b1": b1.ravel().tolist(),
        "W2": W2.ravel().tolist(),
        "b2": b2.ravel().tolist(),
        "W3": W3.ravel().tolist(),
        "b3": b3.ravel().tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def update_models_json(data_root: str, scene_name: str) -> None:
    """Add *scene_name* to models.json (creates the file if needed)."""
    models_path = os.path.join(data_root, "models.json")
    models = []
    if os.path.exists(models_path):
        with open(models_path, "r") as f:
            models = json.load(f)
    if scene_name not in models:
        models.append(scene_name)
        models.sort()
    with open(models_path, "w") as f:
        json.dump(models, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export a DiffSoup checkpoint to web-viewer assets.",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to final_params.pt",
    )
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Scene name for the output directory.  "
             "Default: inferred from checkpoint path.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./web/data",
        help="Root output directory (assets go to <output_dir>/<scene>/).",
    )
    parser.add_argument(
        "--up", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="World up direction.  Auto-detected from checkpoint if omitted.",
    )
    parser.add_argument(
        "--background", type=float, nargs=3, default=None,
        metavar=("R", "G", "B"),
        help="Background colour (linear, 0–1).  "
             "Default: black for MipNeRF-360, white for synthetic.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Infer scene name from checkpoint parent directory.
    scene_name = args.scene or ckpt_path.parent.name
    scene_dir = os.path.join(args.output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    verts = ckpt["V"].numpy().astype(np.float32)
    faces = ckpt["F"].numpy().astype(np.int32)
    feat_acc = ckpt["feat_acc"].numpy().astype(np.float32)
    alpha_acc = ckpt["alpha_acc"].numpy().astype(np.float32)
    Rmax = int(ckpt["Rmax"])
    num_faces = faces.shape[0]

    up = tuple(args.up) if args.up else detect_up(ckpt)
    is_mip360 = "flip_z" in ckpt
    if args.background:
        background = list(args.background)
    else:
        background = [0.0, 0.0, 0.0] if is_mip360 else [1.0, 1.0, 1.0]

    print(f"[mesh]  {verts.shape[0]:,} verts, {num_faces:,} faces")
    print(f"[level] Rmax={Rmax}  texels/face={level_size(Rmax)}")
    print(f"[cam]   up={up}")

    # ── Mesh → PLY ───────────────────────────────────────────────────

    ply_path = os.path.join(scene_dir, "mesh.ply")
    write_ply(ply_path, verts, faces)
    print(f"[save] {ply_path}")

    # ── LUT → PNG ────────────────────────────────────────────────────

    face_color_lut = pack_face_color_lut(feat_acc, alpha_acc, num_faces, Rmax)
    lut0 = (face_color_lut[..., :4] * 255).clip(0, 255).astype(np.uint8)
    lut1 = (face_color_lut[..., 4:] * 255).clip(0, 255).astype(np.uint8)
    tex_H, tex_W = lut0.shape[:2]
    print(f"[lut]   texture {tex_W}x{tex_H}")

    save_lut_png(os.path.join(scene_dir, "lut0.png"), lut0)
    save_lut_png(os.path.join(scene_dir, "lut1.png"), lut1)
    print(f"[save] {scene_dir}/lut0.png, lut1.png")

    # ── MLP → JSON ───────────────────────────────────────────────────

    if "color_mlp" not in ckpt:
        raise KeyError("Checkpoint missing 'color_mlp'.")
    W1, b1, W2, b2, W3, b3 = extract_mlp_weights(ckpt["color_mlp"])
    print(f"[mlp]   W1={W1.shape} W2={W2.shape} W3={W3.shape}")

    mlp_path = os.path.join(scene_dir, "mlp_weights.json")
    save_mlp_json(mlp_path, W1, b1, W2, b2, W3, b3)
    print(f"[save] {mlp_path}")

    # ── Metadata ─────────────────────────────────────────────────────

    meta = {
        "up": list(up),
        "level": Rmax,
        "background": background,
        "num_faces": num_faces,
        "num_verts": int(verts.shape[0]),
    }
    meta_path = os.path.join(scene_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {meta_path}")

    # ── Update model manifest ────────────────────────────────────────

    update_models_json(args.output_dir, scene_name)
    print(f"[save] {args.output_dir}/models.json")

    # ── Summary ──────────────────────────────────────────────────────

    total_kb = sum(
        os.path.getsize(os.path.join(scene_dir, f))
        for f in os.listdir(scene_dir)
    ) / 1024
    print(f"\n[done] Exported '{scene_name}' → {scene_dir}/  ({total_kb:.0f} KB)")
    print(f"       Serve with:  cd {os.path.dirname(args.output_dir)} && "
          f"python -m http.server 8080")
    print(f"       Open:        http://localhost:8080/?model={scene_name}")


if __name__ == "__main__":
    main()
