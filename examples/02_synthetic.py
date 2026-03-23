# examples/02_synthetic.py
# Synthetic / MobileNeRF scene optimisation with DiffSoup.
#
# Loads a MobileNeRF mesh (shape.obj) and NeRF-synthetic camera poses,
# then optimises a triangle-soup radiance field.
#
# Usage:
#   # NeRF Synthetic (Blender)
#   python examples/02_synthetic.py --scene lego
#
#   # Shelly dataset (half resolution, file_path already includes .png)
#   python examples/02_synthetic.py --scene <scene> \
#       --datasets_root ./datasets/shelly_data_release \
#       --downscale 2 --no_png_suffix
#
# Dependencies (beyond diffsoup):
#   pip install imageio tqdm pytorch-msssim scikit-image matplotlib
#
# Note: LPIPS is intentionally excluded from this example.  For fair
# comparison with other methods, ensure you use a consistent LPIPS model
# and weights (e.g. VGG vs AlexNet, v0.0 vs v0.1) across all baselines.

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as sk_ssim
from torch.optim import Adam
from tqdm import tqdm

import diffsoup as ds
from utils import (
    project_vertices,
    exp_decay_mult,
    count_visible_triangles,
    build_keep_map,
    split_edges_from_training_views,
)

# ── Reproducibility ──────────────────────────────────────────────────

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

NEAR, FAR = 0.1, 10.0


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class CameraView:
    file_path: Path
    c2w: np.ndarray       # 4×4 camera-to-world
    K: np.ndarray          # 3×3 intrinsics
    P: np.ndarray          # 4×4 projection
    V: np.ndarray          # 4×4 view (world-to-camera)
    MVP: np.ndarray        # 4×4
    camera_angle_x: float
    width: int
    height: int


@dataclass
class Mesh:
    vertices: np.ndarray        # (N, 3)
    faces: np.ndarray           # (F, 3)
    uvs: Optional[np.ndarray]
    uv_faces: Optional[np.ndarray]


# ── Helpers ──────────────────────────────────────────────────────────

def detect_image_size(test_dir: Path) -> Tuple[int, int]:
    """Find a non-depth PNG in *test_dir* and return ``(width, height)``."""
    pngs = sorted(test_dir.glob("*.png"))
    for p in pngs:
        if "depth" not in p.name:
            with Image.open(p) as im:
                return im.width, im.height
    raise RuntimeError(f"No RGB images found in {test_dir}")


def build_intrinsics(camera_angle_x: float, width: int, height: int) -> np.ndarray:
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)


def build_projection_matrix(camera_angle_x: float, width: int, height: int) -> np.ndarray:
    aspect = width / float(height)
    fovy = 2.0 * np.arctan((1.0 / aspect) * np.tan(camera_angle_x / 2.0))
    f = 1.0 / np.tan(fovy / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (FAR + NEAR) / (NEAR - FAR), (2 * FAR * NEAR) / (NEAR - FAR)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def load_obj_mesh(obj_path: Path) -> Mesh:
    """Minimal OBJ loader (triangles only, with optional UVs)."""
    vertices, uvs, faces, uv_faces = [], [], [], []
    with open(obj_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                vertices.append([x, -z, y])
            elif parts[0] == "vt" and len(parts) >= 3:
                u, v = map(float, parts[1:3])
                uvs.append([u, v])
            elif parts[0] == "f":
                v_idx, vt_idx = [], []
                for vert in parts[1:]:
                    tokens = vert.split("/")
                    vi = int(tokens[0]) if tokens[0] else 0
                    ti = int(tokens[1]) if len(tokens) > 1 and tokens[1] else 0
                    v_idx.append(vi - 1)
                    vt_idx.append(ti - 1 if ti != 0 else -1)
                if len(v_idx) != 3:
                    raise NotImplementedError("Non-triangle faces not supported.")
                faces.append(v_idx)
                uv_faces.append(vt_idx)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    uvs_arr = np.array(uvs, dtype=np.float32) if uvs else None
    uv_faces_arr = np.array(uv_faces, dtype=np.int32) if uvs else None

    print(f"[mesh] {obj_path.name}: {len(vertices)} verts, {len(faces)} faces")
    return Mesh(vertices=vertices, faces=faces, uvs=uvs_arr, uv_faces=uv_faces_arr)


def load_cameras(
    json_path: Path,
    png_suffix: bool = True,
) -> List[CameraView]:
    """Load camera poses from a ``transforms_*.json`` file.

    Args:
        json_path:  Path to the transforms JSON.
        png_suffix: If ``True``, append ``.png`` to each ``file_path``
                    (NeRF-synthetic convention).  Set ``False`` for datasets
                    where paths already include the extension.
    """
    base_dir = json_path.parent
    with open(json_path, "r") as f:
        meta = json.load(f)

    camera_angle_x = float(meta["camera_angle_x"])
    frames = meta["frames"]

    test_dir = base_dir / "test"
    width, height = detect_image_size(test_dir)

    K = build_intrinsics(camera_angle_x, width, height)
    P = build_projection_matrix(camera_angle_x, width, height)

    views: List[CameraView] = []
    for frame in frames:
        rel_path = frame["file_path"]
        img_path = base_dir / (f"{rel_path}.png" if png_suffix else rel_path)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        V = np.linalg.inv(c2w)
        MVP = P @ V

        views.append(CameraView(
            file_path=img_path, c2w=c2w, K=K.copy(), P=P.copy(), V=V,
            MVP=MVP, camera_angle_x=camera_angle_x, width=width, height=height,
        ))

    print(f"[cameras] {len(views)} views, {width}×{height}")
    return views


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DiffSoup synthetic scene example")
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--datasets_root", type=str, default="./datasets/nerf_synthetic",
                        help="Root of the NeRF-synthetic (or Shelly) dataset")
    parser.add_argument("--mobilenerf_root", type=str, default="./datasets/mobilenerf_results",
                        help="Root of MobileNeRF results (contains <scene>/obj/shape_15K.obj)")
    parser.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4],
                        help="Downscale GT images by this factor (e.g. 2 for Shelly)")
    parser.add_argument("--no_png_suffix", action="store_true",
                        help="Don't append .png to file_path (use for Shelly)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--target_prims", type=int, default=15_000)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: ./results/02_synthetic/<scene>)")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    mobilenerf_root = Path(args.mobilenerf_root)
    scene = args.scene
    png_suffix = not args.no_png_suffix

    out_dir = args.out_dir or f"./results/02_synthetic/{scene}"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")

    # ── Load mesh ────────────────────────────────────────────────────

    obj_path = mobilenerf_root / scene / "obj" / "shape_15K.obj"
    if not obj_path.exists():
        print(f"[error] {obj_path} not found"); return
    mesh = load_obj_mesh(obj_path)

    # ── Load train cameras ───────────────────────────────────────────

    json_path = datasets_root / scene / "transforms_train.json"
    if not json_path.exists():
        print(f"[error] {json_path} not found"); return
    train_views = load_cameras(json_path, png_suffix=png_suffix)

    H, W = train_views[0].height, train_views[0].width

    # ── Load train GT images ─────────────────────────────────────────

    mvps, gt_rgb_list = [], []
    for view in train_views:
        mvp = torch.from_numpy(view.MVP).to(dtype=torch.float32, device=device).unsqueeze(0)
        mvps.append(mvp)

        gt_path = str(view.file_path)
        gt_rgba = iio.imread(gt_path) / 255.0
        gt_rgba = np.ascontiguousarray(np.flip(gt_rgba, axis=0)).astype(np.float32)

        img = torch.from_numpy(gt_rgba).float().permute(2, 0, 1).unsqueeze(0).to(device)

        if args.downscale > 1:
            img = torch.nn.functional.interpolate(
                img, size=(H // args.downscale, W // args.downscale),
                mode="bilinear", align_corners=False,
            )

        gt_rgba_t = img.permute(0, 2, 3, 1)
        alpha = gt_rgba_t[..., -1:]
        gt_rgb = gt_rgba_t[..., :3] * alpha + (1.0 - alpha)
        gt_rgb_list.append(gt_rgb)

    mvps = torch.cat(mvps, dim=0)
    mvps_inv = torch.inverse(mvps).contiguous()
    gt_rgb = torch.cat(gt_rgb_list, dim=0)

    if args.downscale > 1:
        H, W = H // args.downscale, W // args.downscale

    N_train = mvps.shape[0]
    print(f"[train] {N_train} views, {H}×{W}")

    # ── Convert mesh to triangle soup ────────────────────────────────

    num_faces = mesh.faces.shape[0]
    new_verts = mesh.vertices[mesh.faces.reshape(-1)].copy()
    new_faces = np.arange(num_faces * 3, dtype=mesh.faces.dtype).reshape(num_faces, 3)

    verts = torch.from_numpy(new_verts).float().cuda()
    faces = torch.from_numpy(new_faces).int().cuda()

    # ── Feature buffers ──────────────────────────────────────────────

    Rmin, Rmax = 0, 0
    feat_dim = 7

    feat_src = ds.build_multires_triangle_color(
        faces.shape[0], Rmin, Rmax, feat_dim,
    ).to(device="cuda").requires_grad_(True)

    alpha_src = ds.build_multires_triangle_color(
        faces.shape[0], Rmin, Rmax, feat_dim=1,
    ).to(device="cuda").requires_grad_(True)

    verts.requires_grad = True

    # ── Colour MLP ───────────────────────────────────────────────────

    color_mlp = ds.ColorMLP(
        input_dim=feat_dim + 9, hidden_dim=16, n_layers=2, output_dim=3,
    ).to(device="cuda")

    # ── Optimisers ───────────────────────────────────────────────────

    optimizer_soup = Adam([
        {"params": [feat_src], "lr": 5e-2},
        {"params": [alpha_src], "lr": 5e-2},
    ])
    optimizer_vert = ds.optimize.VectorAdam(params=[verts], lr=1e-3)
    optimizer_shader = Adam([{"params": color_mlp.parameters(), "lr": 1e-2}])

    base_soup_lrs = [pg["lr"] for pg in optimizer_soup.param_groups]
    base_vert_lrs = [pg["lr"] for pg in optimizer_vert.param_groups]
    base_shader_lrs = [pg["lr"] for pg in optimizer_shader.param_groups]

    # ── Training loop ────────────────────────────────────────────────

    steps = args.steps
    batch_size = args.batch_size
    TARGET_PRIMS = args.target_prims
    losses: List[float] = []

    perm = torch.randperm(N_train, device=device)
    ptr = 0

    pbar = tqdm(range(1, steps + 1), desc="optimising", leave=True)
    for i_iter in pbar:
        end = min(ptr + batch_size, N_train)
        batch_idx = perm[ptr:end]
        ptr = end
        if ptr == N_train:
            perm = torch.randperm(N_train, device=device)
            ptr = 0

        # LR schedule
        mult = exp_decay_mult(i_iter, steps, final_mult=0.01)
        for opt, base_lrs in (
            (optimizer_soup, base_soup_lrs),
            (optimizer_vert, base_vert_lrs),
            (optimizer_shader, base_shader_lrs),
        ):
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = base_lr * mult

        batch_mvps = mvps[batch_idx]
        batch_mvps_inv = mvps_inv[batch_idx]
        batch_gt = gt_rgb[batch_idx]

        clip_verts = project_vertices(verts, batch_mvps)
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

        rast_out = ds.rasterize_multires_triangle_alpha(
            (H, W), clip_verts, faces, level=Rmax, alpha_src=alpha_acc,
        )
        feat = ds.multires_triangle_color(
            rast_out, level=Rmax, feat=feat_acc,
        ).view(-1, H, W, feat_dim)
        feat = torch.cat([feat, ds.encode_view_dir_sh2(rast_out, batch_mvps_inv)], dim=-1)
        color = color_mlp.forward(feat, mask=rast_out[..., -1] > 0).view(-1, H, W, 3)

        mask = (rast_out.detach()[..., -1:] > 0).float()
        color = mask * color + (1.0 - mask)

        # Opacity auxiliary loss (zero-valued; hooks gradient into alpha_src)
        aux_loss = ds.opacity_aux_loss(
            color.detach(), batch_gt, rast_out, clip_verts, faces,
            level=Rmax, alpha_src=alpha_acc,
        )
        color = ds.edge_grad(color, rast_out, clip_verts, faces)
        l1_loss = (batch_gt - color).abs().mean()
        ssim_loss = 0.5 * (1 - ssim(
            batch_gt.permute(0, 3, 1, 2), color.permute(0, 3, 1, 2), data_range=1.0,
        ))
        loss = aux_loss + 0.8 * l1_loss + 0.2 * ssim_loss

        optimizer_soup.zero_grad(set_to_none=True)
        optimizer_vert.zero_grad(set_to_none=True)
        optimizer_shader.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_soup.step()
        optimizer_vert.step()
        optimizer_shader.step()

        l = float(loss.detach().item())
        losses.append(l)
        pbar.set_postfix(loss=f"{l:.6f}")

        # ── Lift multi-resolution levels at step 5 000 ───────────────
        if i_iter == 5_000 and i_iter < steps:
            with torch.no_grad():
                Rmin, Rmax = 2, 5
                feat_src_lifted = ds.accumulate_to_level(0, 0, feat_src, target_level=Rmin)
                new_feat_src = ds.build_multires_triangle_color(
                    faces.shape[0], Rmin, Rmax, feat_dim=feat_dim,
                ).to(device="cuda")
                new_feat_src[..., : feat_src_lifted.shape[1], :] = feat_src_lifted
                feat_src = new_feat_src

                alpha_src = ds.build_multires_triangle_color(
                    faces.shape[0], Rmin, Rmax, feat_dim=1,
                ).to(device="cuda")

            feat_src.requires_grad = True
            alpha_src.requires_grad = True

            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[verts], lr=old_lr_vert)

        # ── Resample soup ────────────────────────────────────────────
        if i_iter % 100 == 0 and i_iter < 9_550:
            with torch.no_grad():
                if faces.shape[0] > TARGET_PRIMS:
                    alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
                    tri_counts = count_visible_triangles(
                        (H // 2, W // 2), mvps, verts, faces,
                        level=Rmax, alpha_src=alpha_acc, batch_size=1,
                    )
                    keep_map = build_keep_map(tri_counts, remove=faces.shape[0] - TARGET_PRIMS)
                    faces = faces[keep_map]
                    verts, faces = ds.remove_unreferenced_vertices_from_soup(verts, faces)
                    feat_src = ds.expand_by_index(feat_src, keep_map)
                    alpha_src = ds.expand_by_index(alpha_src, keep_map)

                if i_iter < 9_500:
                    alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
                    verts, faces, face_map = split_edges_from_training_views(
                        (H // 2, W // 2), mvps, verts, faces,
                        Rmax, alpha_acc, tau_ratio=1 / 5, num_views_cap=20,
                    )
                    feat_src = ds.expand_by_index(feat_src, face_map)
                    alpha_src = ds.expand_by_index(alpha_src, face_map)

            print(f"  [resample] verts={verts.shape[0]:,}  faces={faces.shape[0]:,}")

            verts.requires_grad = True
            feat_src.requires_grad = True
            alpha_src.requires_grad = True

            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[verts], lr=old_lr_vert)

    # ── Loss curve ───────────────────────────────────────────────────

    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.2)
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_png, bbox_inches="tight"); plt.close()
    print(f"[save] loss curve → {loss_png}")

    # ── Checkpoint ───────────────────────────────────────────────────

    with torch.no_grad():
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

    ckpt_path = os.path.join(out_dir, "final_params.pt")
    torch.save({
        "feat_acc": feat_acc.detach().cpu(),
        "alpha_acc": alpha_acc.detach().cpu(),
        "V": verts.detach().cpu(),
        "F": faces.detach().cpu(),
        "color_mlp": color_mlp.state_dict(),
        "Rmin": Rmin,
        "Rmax": Rmax,
        "feat_dim": feat_dim,
        "H": H, "W": W,
        "steps": steps,
        "losses": losses,
        "seed": SEED,
    }, ckpt_path)
    print(f"[save] checkpoint → {ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────

    json_test = datasets_root / scene / "transforms_test.json"
    test_views = load_cameras(json_test, png_suffix=png_suffix)
    print(f"[test] {len(test_views)} views")

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for i, view in enumerate(test_views):
            mvp = torch.from_numpy(view.MVP).to(dtype=torch.float32, device=device).unsqueeze(0)
            mvp_inv = torch.inverse(mvp).contiguous()
            clip_verts = project_vertices(verts, mvp)

            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), clip_verts, faces,
                level=Rmax, alpha_src=alpha_acc, stochastic=False,
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax, feat=feat_acc,
            ).view(-1, H, W, feat_dim)
            feat = torch.cat([feat, ds.encode_view_dir_sh2(rast_out, mvp_inv)], dim=-1)
            color = color_mlp.forward(feat, mask=rast_out[..., -1] > 0).view(-1, H, W, 3)

            mask = (rast_out.detach()[..., -1:] > 0).float()
            color = mask * color + (1.0 - mask)

            color_np = color.squeeze(0).cpu().numpy()
            color_np = np.flip(color_np, axis=0)

            gt_path = str(view.file_path)
            gt_rgba = iio.imread(gt_path) / 255.0
            gt_rgba = np.ascontiguousarray(gt_rgba).astype(np.float32)

            if args.downscale > 1:
                gt_t = torch.from_numpy(gt_rgba).float().permute(2, 0, 1).unsqueeze(0)
                gt_t = torch.nn.functional.interpolate(
                    gt_t, size=(H, W), mode="bilinear", align_corners=False,
                )
                gt_rgba = gt_t.squeeze(0).permute(1, 2, 0).numpy()

            alpha_np = gt_rgba[..., -1:]
            gt_rgb_np = gt_rgba[..., :3] * alpha_np + (1.0 - alpha_np)

            mse = np.mean((color_np - gt_rgb_np) ** 2)
            psnr_val = float("inf") if mse == 0 else -10.0 * math.log10(mse)
            psnr_list.append(psnr_val)

            ssim_val = sk_ssim(gt_rgb_np, color_np, data_range=1.0, channel_axis=2)
            ssim_list.append(ssim_val)

            iio.imsave(
                os.path.join(out_dir, f"render_{i:04d}.png"),
                (color_np * 255).clip(0, 255).astype(np.uint8),
            )
            iio.imsave(
                os.path.join(out_dir, f"gt_{i:04d}.png"),
                (gt_rgb_np * 255).clip(0, 255).astype(np.uint8),
            )

    print(f"[save] test renders → {out_dir}/")

    if psnr_list:
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        print(f"[metrics] PSNR  {avg_psnr:.3f} dB")
        print(f"[metrics] SSIM  {avg_ssim:.4f}")

        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
                f.write(f"{i:04d} PSNR={p:.3f} SSIM={s:.4f}\n")
            f.write(f"\nmean PSNR={avg_psnr:.3f} SSIM={avg_ssim:.4f}\n")


if __name__ == "__main__":
    main()
