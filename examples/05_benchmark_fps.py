# examples/05_benchmark_fps.py
# Headless FPS benchmark for a trained DiffSoup model.
#
# Loads a checkpoint produced by 01_mip360.py, 02_synthetic.py, or
# 03_random_init.py, rebuilds MVP matrices from the original dataset,
# and runs the OpenGL viewer in benchmark mode to measure rendering
# throughput.
#
# Usage (MipNeRF-360):
#   python examples/05_benchmark_fps.py \
#       --ckpt results/01_mip360/kitchen/final_params.pt \
#       --scene_root ./datasets/360_v2/kitchen
#
# Usage (NeRF-Synthetic / Shelly):
#   python examples/05_benchmark_fps.py \
#       --ckpt results/02_synthetic/lego/final_params.pt \
#       --scene_root ./datasets/nerf_synthetic/lego \
#       --dataset_type synthetic
#
#   python examples/05_benchmark_fps.py \
#       --ckpt results/02_synthetic/lego/final_params.pt \
#       --scene_root ./datasets/nerf_synthetic/lego \
#       --dataset_type synthetic --downscale 2 --no_png_suffix
#
# Dependencies:
#   pip install diffsoupviewer numpy torch

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

import diffsoupviewer
from utils import load_mipnerf360_scene, mvp_from_K_Tcw


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


# ── NeRF-Synthetic camera loading ────────────────────────────────────


def _detect_image_size(test_dir: Path) -> Tuple[int, int]:
    """Find a non-depth PNG in *test_dir* and return ``(width, height)``."""
    from PIL import Image as PILImage

    pngs = sorted(test_dir.glob("*.png"))
    for p in pngs:
        if "depth" not in p.name:
            with PILImage.open(p) as im:
                return im.width, im.height
    raise RuntimeError(f"No RGB images found in {test_dir}")


NEAR_SYNTHETIC, FAR_SYNTHETIC = 0.1, 10.0


def _build_projection_matrix(
    camera_angle_x: float, width: int, height: int,
) -> np.ndarray:
    """OpenGL perspective projection for NeRF-synthetic cameras."""
    aspect = width / float(height)
    fovy = 2.0 * np.arctan((1.0 / aspect) * np.tan(camera_angle_x / 2.0))
    f = 1.0 / np.tan(fovy / 2.0)
    n, fa = NEAR_SYNTHETIC, FAR_SYNTHETIC
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (fa + n) / (n - fa), (2 * fa * n) / (n - fa)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def load_synthetic_mvps(
    scene_root: Path,
    splits: Sequence[str] = ("train", "test"),
    png_suffix: bool = True,
    downscale: int = 1,
) -> Tuple[np.ndarray, int, int]:
    """Load MVP matrices for NeRF-synthetic / Shelly datasets.

    Returns:
        mvps:   float32 [B, 4, 4] row-major MVP matrices.
        width:  render width (after downscale).
        height: render height (after downscale).
    """
    mvps_all: List[np.ndarray] = []
    width, height = 0, 0

    for split in splits:
        json_path = scene_root / f"transforms_{split}.json"
        if not json_path.exists():
            print(f"[warn] {json_path} not found, skipping split '{split}'")
            continue

        with open(json_path, "r") as f:
            meta = json.load(f)

        camera_angle_x = float(meta["camera_angle_x"])

        # Detect image dimensions from test folder
        if width == 0:
            test_dir = scene_root / "test"
            w_full, h_full = _detect_image_size(test_dir)
            width = w_full // max(downscale, 1)
            height = h_full // max(downscale, 1)

        P = _build_projection_matrix(camera_angle_x, width, height)

        for frame in meta["frames"]:
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            V = np.linalg.inv(c2w).astype(np.float32)
            mvps_all.append(P @ V)

    if not mvps_all:
        raise FileNotFoundError(
            f"No transforms_*.json found in {scene_root} for splits {splits}"
        )

    return np.stack(mvps_all, axis=0), width, height


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="DiffSoup headless FPS benchmark.",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to final_params.pt",
    )
    parser.add_argument(
        "--scene_root", type=str, required=True,
        help="Path to the dataset scene directory "
             "(e.g. ./datasets/360_v2/kitchen or ./datasets/nerf_synthetic/lego)",
    )
    parser.add_argument(
        "--dataset_type", type=str, default=None,
        choices=["mip360", "synthetic"],
        help="Dataset type.  Auto-detected from checkpoint if omitted "
             "(flip_z present → mip360, otherwise → synthetic).",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "test"],
        help="Camera splits to benchmark over (default: train test).",
    )
    parser.add_argument(
        "--downscale", type=int, default=None,
        help="Image downscale factor (mip360 only; default: inferred from ckpt).",
    )
    parser.add_argument(
        "--no_png_suffix", action="store_true",
        help="Don't append .png to file_path in transforms JSON "
             "(synthetic only; use for Shelly dataset).",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Override render width (default: from dataset / checkpoint).",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Override render height (default: from dataset / checkpoint).",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warm-up frames before timing (default: 10).",
    )
    parser.add_argument(
        "--save_every", type=int, default=0,
        help="Save a screenshot every N frames; 0 disables (default: 0).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for benchmark results (default: beside checkpoint).",
    )
    parser.add_argument(
        "--up", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="World up direction.  Auto-detected from checkpoint if omitted.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    scene_root = Path(args.scene_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")

    output_dir = args.output_dir or str(ckpt_path.parent / "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    verts = ckpt["V"].numpy().astype(np.float32)
    faces = ckpt["F"].numpy().astype(np.int32)
    feat_acc = ckpt["feat_acc"].numpy().astype(np.float32)
    alpha_acc = ckpt["alpha_acc"].numpy().astype(np.float32)
    Rmax = int(ckpt["Rmax"])
    feat_dim = int(ckpt["feat_dim"])
    num_faces = faces.shape[0]

    print(f"[mesh]  {verts.shape[0]:,} verts, {num_faces:,} faces")
    print(f"[level] Rmax={Rmax}  texels/face={level_size(Rmax)}")

    # Auto-detect dataset type
    dataset_type = args.dataset_type
    if dataset_type is None:
        dataset_type = "mip360" if "flip_z" in ckpt else "synthetic"
        print(f"[auto]  dataset_type={dataset_type}")

    up = tuple(args.up) if args.up else detect_up(ckpt)
    print(f"[cam]   up={up}")

    # ── Pack LUTs ────────────────────────────────────────────────────

    face_color_lut = pack_face_color_lut(feat_acc, alpha_acc, num_faces, Rmax)
    lut0 = (face_color_lut[..., :4] * 255).clip(0, 255).astype(np.uint8)
    lut1 = (face_color_lut[..., 4:] * 255).clip(0, 255).astype(np.uint8)
    print(f"[lut]   texture {face_color_lut.shape[1]}x{face_color_lut.shape[0]}")

    # ── Extract MLP weights ──────────────────────────────────────────

    if "color_mlp" not in ckpt:
        raise KeyError("Checkpoint missing 'color_mlp'.")
    W1, b1, W2, b2, W3, b3 = extract_mlp_weights(ckpt["color_mlp"])
    print(f"[mlp]   W1={W1.shape} W2={W2.shape} W3={W3.shape}")

    # ── Build MVP matrices from dataset ──────────────────────────────

    if dataset_type == "mip360":
        flip_z = bool(ckpt.get("flip_z", True))
        z_near, z_far = 0.5, 100.0

        mvps_list = []
        render_W, render_H = 0, 0

        for split in args.splits:
            data = load_mipnerf360_scene(
                str(scene_root), split=split,
                downscale=args.downscale or ckpt.get("downscale", 4),
                device="cpu",
            )
            K, H, W = data["K"], data["H"], data["W"]
            render_H, render_W = H, W

            for fr in data["frames"]:
                mvp = mvp_from_K_Tcw(
                    K, fr["Tcw"], (H, W),
                    z_near=z_near, z_far=z_far, flip_z=flip_z,
                )
                mvps_list.append(mvp)

        mvps_row = torch.stack(mvps_list, dim=0).numpy().astype(np.float32)

    elif dataset_type == "synthetic":
        png_suffix = not args.no_png_suffix
        downscale = args.downscale or 1

        mvps_row, render_W, render_H = load_synthetic_mvps(
            scene_root, splits=args.splits,
            png_suffix=png_suffix, downscale=downscale,
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Allow CLI overrides for render resolution
    render_W = args.width or render_W
    render_H = args.height or render_H

    # Transpose row-major → column-major for OpenGL (glm::make_mat4)
    mvps = np.ascontiguousarray(
        np.transpose(mvps_row, (0, 2, 1)), dtype=np.float32,
    )
    B = mvps.shape[0]
    print(f"[views] {B} benchmark views, render size {render_W}x{render_H}")

    # ── Run benchmark ────────────────────────────────────────────────

    print("[benchmark] starting …")
    diffsoupviewer.benchmark(
        verts=verts,
        faces=faces,
        lut0=np.ascontiguousarray(lut0),
        lut1=np.ascontiguousarray(lut1),
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        mvps=mvps,
        width=render_W,
        height=render_H,
        warmup=args.warmup,
        save_every=args.save_every,
        output_dir=os.path.abspath(output_dir),
        up=up,
    )
    print("[benchmark] done.")

    # ── Report ───────────────────────────────────────────────────────

    summary_path = os.path.join(output_dir, "benchmark_summary.txt")
    if os.path.exists(summary_path):
        print()
        with open(summary_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    main()
