# examples/utils.py
# Shared utilities for DiffSoup example scripts.

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import imageio.v3 as iio
import torchvision.transforms.functional as TF


# =====================================================================
#  COLMAP points3D reader
# =====================================================================

def read_points3D(scene_root: str) -> np.ndarray:
    """Read COLMAP points3D from text or binary format. Returns (N, 3) float32."""
    spr = _colmap_root(scene_root)
    txtp = os.path.join(spr, "points3D.txt")
    binp = os.path.join(spr, "points3D.bin")

    if os.path.exists(txtp):
        xs = []
        with open(txtp, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                toks = line.strip().split()
                xs.append(list(map(float, toks[1:4])))
        return np.asarray(xs, np.float32)

    if os.path.exists(binp):
        xs = []
        with open(binp, "rb") as f:
            (npts,) = struct.unpack("<Q", f.read(8))
            for _ in range(npts):
                f.read(8)  # point3D_id
                X = struct.unpack("<ddd", f.read(24))
                f.read(3)  # RGB
                f.read(8)  # error
                (track_len,) = struct.unpack("<Q", f.read(8))
                f.seek(track_len * 8, os.SEEK_CUR)
                xs.append(X)
        return np.asarray(xs, np.float32)

    raise FileNotFoundError("points3D.(txt|bin) not found")


# =====================================================================
#  Farthest-point downsampling (via Open3D)
# =====================================================================

def farthest_point_downsample(points: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sampling via Open3D. Returns (n, 3) float32 subset."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.farthest_point_down_sample(n)
    return np.asarray(pcd_down.points, dtype=np.float32)


# =====================================================================
#  MVP from intrinsics + extrinsics
# =====================================================================

def opengl_P_from_K(
    K: torch.Tensor,
    image_size: Tuple[int, int],
    z_near: float = 0.01,
    z_far: float = 1000.0,
) -> torch.Tensor:
    """OpenGL projection matrix from intrinsics K for NDC in [-1, 1]."""
    H, W = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = 1.0 - 2.0 * (cx / W)
    P[1, 2] = 2.0 * (cy / H) - 1.0
    P[2, 2] = (z_far + z_near) / (z_near - z_far)
    P[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    P[3, 2] = -1.0
    return P


def mvp_from_K_Tcw(
    K: torch.Tensor,
    Tcw: torch.Tensor,
    image_size: Tuple[int, int],
    z_near: float = 0.01,
    z_far: float = 1000.0,
    flip_z: bool = False,
) -> torch.Tensor:
    """Build MVP = P @ V from camera intrinsics and world-to-camera transform."""
    P = opengl_P_from_K(K, image_size, z_near, z_far)
    V = Tcw.clone()
    if flip_z:
        ZF = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=V.dtype, device=V.device,
        )
        V = ZF @ V
    return P @ V


# =====================================================================
#  MipNeRF 360 COLMAP loader
# =====================================================================

def _colmap_root(scene_root: str) -> str:
    p = os.path.join(scene_root, "sparse", "0")
    return p if os.path.isdir(p) else os.path.join(scene_root, "sparse")


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _fread(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))


# ---------- cameras ----------

def _read_cameras_txt(path: str) -> Dict[int, dict]:
    cams = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            toks = line.strip().split()
            cam_id = int(toks[0])
            model = toks[1]
            w, h = int(toks[2]), int(toks[3])
            params = list(map(float, toks[4:]))
            cams[cam_id] = {"model": model, "w": w, "h": h, "params": params}
    return cams


def _read_cameras_bin(path: str) -> Dict[int, dict]:
    _MODEL_ID_TO_NAME = {
        0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
        3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE", 8: "FULL_OPENCV",
    }
    _MODEL_NUM_PARAMS = {
        "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
        "RADIAL": 5, "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12,
    }
    cams = {}
    with open(path, "rb") as f:
        (num_cams,) = _fread(f, "<Q")
        for _ in range(num_cams):
            (cam_id,) = _fread(f, "<I")
            (model_id,) = _fread(f, "<I")
            (width,) = _fread(f, "<Q")
            (height,) = _fread(f, "<Q")
            model = _MODEL_ID_TO_NAME.get(model_id, f"MODEL_{model_id}")
            np_ = _MODEL_NUM_PARAMS.get(model, 4)
            params = list(_fread(f, "<" + "d" * np_))
            cams[cam_id] = {"model": model, "w": int(width), "h": int(height), "params": params}
    return cams


def _read_cameras(spr: str) -> Dict[int, dict]:
    txt = os.path.join(spr, "cameras.txt")
    if os.path.exists(txt):
        return _read_cameras_txt(txt)
    binp = os.path.join(spr, "cameras.bin")
    if os.path.exists(binp):
        return _read_cameras_bin(binp)
    raise FileNotFoundError("cameras.txt/.bin not found in " + spr)


# ---------- images ----------

def _read_images_txt(path: str) -> List[dict]:
    imgs = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f]
    i = 0
    while i < len(lines):
        line = lines[i]; i += 1
        if line.startswith("#") or not line:
            continue
        toks = line.split()
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = toks[9]
        if i < len(lines) and not lines[i].startswith("#"):
            i += 1  # skip correspondences
        imgs.append({
            "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
            "tvec": np.array([tx, ty, tz], dtype=np.float64),
            "camera_id": cam_id,
            "name": name,
        })
    return imgs


def _read_images_bin(path: str) -> List[dict]:
    imgs = []
    with open(path, "rb") as f:
        (num_imgs,) = _fread(f, "<Q")
        for _ in range(num_imgs):
            (_image_id,) = _fread(f, "<I")
            qw, qx, qy, qz = _fread(f, "<dddd")
            tx, ty, tz = _fread(f, "<ddd")
            (cam_id,) = _fread(f, "<I")
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.append(c)
            name = b"".join(name_bytes).decode("utf-8")
            (num_points2D,) = _fread(f, "<Q")
            f.seek(num_points2D * (2 * 8 + 8), 1)
            imgs.append({
                "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
                "tvec": np.array([tx, ty, tz], dtype=np.float64),
                "camera_id": cam_id,
                "name": name,
            })
    return imgs


def _read_images(spr: str) -> List[dict]:
    txt = os.path.join(spr, "images.txt")
    if os.path.exists(txt):
        return _read_images_txt(txt)
    binp = os.path.join(spr, "images.bin")
    if os.path.exists(binp):
        return _read_images_bin(binp)
    raise FileNotFoundError("images.txt/.bin not found in " + spr)


# ---------- intrinsics ----------

def _intrinsics_from_camera(cam: dict) -> Tuple[torch.Tensor, int, int]:
    model = cam["model"].upper()
    w, h = cam["w"], cam["h"]
    p = cam["params"]
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K, h, w


def _image_folder_name(downscale: int) -> str:
    """Map downscale factor to folder name: 0 or 1 → 'images', 2 → 'images_2', etc."""
    if downscale <= 1:
        return "images"
    return f"images_{downscale}"


# ---------- public API ----------

def load_mipnerf360_scene(
    scene_root: str,
    split: str = "train",
    holdout: int = 8,
    downscale: int = 4,
    device: Optional[torch.device] = None,
    linearize: bool = False,
) -> Dict:
    """
    Load a MipNeRF 360 scene in LLFF layout with 3DGS-compatible train/test split.

    Args:
        scene_root: Path to scene directory.
        split: "train" or "test".
        holdout: Every holdout-th view goes to test (default 8, matching 3DGS).
        downscale: Image downscale factor. 0 or 1 = ``images/``, 2 = ``images_2/``,
                   4 = ``images_4/``, 8 = ``images_8/``.
        device: Target device for tensors (None = CPU).
        linearize: If True, convert sRGB to linear.

    Returns:
        Dict with keys: frames, K, H, W, orig_H, orig_W, folder.
        Each frame dict has: c2w, Tcw, image, img_path.
    """
    spr = _colmap_root(scene_root)
    cams = _read_cameras(spr)
    imgs = _read_images(spr)

    # Intrinsics from first camera
    first_cam = cams[imgs[0]["camera_id"]]
    K0, orig_H, orig_W = _intrinsics_from_camera(first_cam)

    # Image folder
    folder = _image_folder_name(downscale)
    img_dir = os.path.join(scene_root, folder)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"'{folder}' folder not found at: {img_dir}")

    # Determine actual resolution from first image
    name0 = imgs[0]["name"]
    p0 = os.path.join(img_dir, name0)
    if not os.path.exists(p0):
        p0 = os.path.join(img_dir, os.path.basename(name0))
    im0 = iio.imread(p0)
    out_H, out_W = im0.shape[0], im0.shape[1]

    # Scale intrinsics to match chosen image folder
    sx, sy = out_W / float(orig_W), out_H / float(orig_H)
    K = K0.clone()
    K[0, 0] *= sx; K[1, 1] *= sy
    K[0, 2] *= sx; K[1, 2] *= sy

    # 3DGS-exact LLFF train/test split
    sorted_indices = sorted(range(len(imgs)), key=lambda i: imgs[i]["name"])
    test_set = set()
    for rank, idx in enumerate(sorted_indices):
        if (rank % holdout) == 0:
            test_set.add(idx)

    if split == "test":
        selected = [i for i in sorted_indices if i in test_set]
    else:
        selected = [i for i in sorted_indices if i not in test_set]

    # Build frames
    frames = []
    for i in selected:
        rec = imgs[i]
        R = _qvec2rotmat(rec["qvec"])
        tvec = rec["tvec"].reshape(3, 1)

        Tcw = np.eye(4); Tcw[:3, :3] = R; Tcw[:3, 3:4] = tvec
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = (-R.T @ tvec).ravel()

        img_path = os.path.join(img_dir, rec["name"])
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, os.path.basename(rec["name"]))

        img = torch.from_numpy(iio.imread(img_path).astype(np.float32) / 255.0)
        if img.ndim == 2:
            img = img[..., None].expand(-1, -1, 3)
        if (img.shape[0], img.shape[1]) != (out_H, out_W):
            img = TF.resize(
                img.permute(2, 0, 1), [out_H, out_W],
                interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
            ).permute(1, 2, 0)
        if linearize:
            a = (img <= 0.04045).float()
            img = a * (img / 12.92) + (1 - a) * torch.pow((img + 0.055) / 1.055, 2.4)

        item = {
            "c2w": torch.from_numpy(c2w).float(),
            "Tcw": torch.from_numpy(Tcw).float(),
            "image": img,
            "img_path": img_path,
        }
        if device:
            for k in ("c2w", "Tcw", "image"):
                if k in item and torch.is_tensor(item[k]):
                    item[k] = item[k].to(device)
        frames.append(item)

    return {
        "frames": frames,
        "K": K.to(device) if device else K,
        "H": out_H, "W": out_W,
        "orig_H": orig_H, "orig_W": orig_W,
        "folder": folder,
    }
