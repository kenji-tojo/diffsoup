# Minimal, plug-and-play voxel downsampling with v = alpha * median 1-NN spacing.
# Works on CPU or GPU. No external deps beyond PyTorch.

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
import torch

def nn_spacing(points: np.ndarray, reduction: str = "median") -> float:
    """
    Exact 1-NN spacing for a 3D point set.

    Args:
        points:    (N, 3) array-like of XYZ coordinates.
        reduction: "median" (robust default) or "mean" (aka "avg"/"average").

    Returns:
        float: reduced 1-NN distance across all points.

    Notes:
        Uses cKDTree.query with k=2 and p=2 (Euclidean). The first neighbor is
        the point itself (distance 0), so we take column 1 as the true 1-NN.
    """
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`points` must be shaped (N, 3).")
    if P.shape[0] < 2:
        return 0.0

    # Exact NN via KD-tree
    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2, p=2)  # (N, 2): [:,0]=0 (self), [:,1]=1-NN
    nn = dists[:, 1]

    # Safety: remove any non-finite values (shouldn't happen but harmless)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0

    r = reduction.lower()
    if r in ("median", "med"):
        return float(np.median(nn))
    elif r in ("mean", "avg", "average"):
        return float(np.mean(nn))
    else:
        raise ValueError("`reduction` must be one of {'median','mean','avg','average'}.")

def sample_uniform_grid3d(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    N: int,
    jitter: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a nearly uniform 3D point set inside an axis-aligned box
    using a regular grid, with optional sub-grid random jitter.

    Chooses integer grid counts (nx, ny, nz) so that nx*ny*nz is as close as
    possible to N while keeping cells roughly cubic. Points are placed at
    cell centers, optionally perturbed by random jitter inside each cell.

    Args:
        bbox_min: (3,) float array of the minimum corner [xmin, ymin, zmin].
        bbox_max: (3,) float array of the maximum corner [xmax, ymax, zmax].
        N:       Target number of samples.
        jitter:  Float in [0,1]. 0 → perfect grid; 1 → uniform random within cell.
        seed:    Optional random seed for reproducibility.

    Returns:
        (M, 3) float32 array of sample coordinates within the box,
        where M ≈ N (M = nx*ny*nz).
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    assert bbox_min.shape == (3,)
    assert bbox_max.shape == (3,)
    N = int(N)
    if N <= 1:
        return bbox_min[None, :].astype(np.float32)

    # Box extents and volume
    L = np.maximum(bbox_max - bbox_min, 1e-12)
    vol = float(L[0] * L[1] * L[2])

    # Ideal cubic cell size from N
    h = (vol / max(N, 1)) ** (1.0 / 3.0)
    nx_f, ny_f, nz_f = L / h

    # Candidate integer resolutions (floor/ceil)
    fx, fy, fz = np.floor([nx_f, ny_f, nz_f])
    cx, cy, cz = np.ceil ([nx_f, ny_f, nz_f])

    def clip_int(x):  # ensure >=1 and not absurd
        return int(np.clip(x, 1, max(1, N)))

    candidates = []
    for ix in (fx, cx):
        for iy in (fy, cy):
            for iz in (fz, cz):
                nx, ny, nz = clip_int(ix), clip_int(iy), clip_int(iz)
                M = nx * ny * nz
                dx, dy, dz = L[0]/nx, L[1]/ny, L[2]/nz
                iso = (dx - dy)**2 + (dy - dz)**2 + (dz - dx)**2
                candidates.append((abs(M - N), iso, (nx, ny, nz)))

    _, _, (nx, ny, nz) = min(candidates, key=lambda t: (t[0], t[1]))

    # Base grid cell centers
    xs = np.linspace(bbox_min[0], bbox_max[0], nx, endpoint=False) + 0.5 * (L[0] / nx)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny, endpoint=False) + 0.5 * (L[1] / ny)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz, endpoint=False) + 0.5 * (L[2] / nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Optional jitter inside each cell
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        dx, dy, dz = L[0]/nx, L[1]/ny, L[2]/nz
        offsets = (rng.random(size=pts.shape) - 0.5) * jitter * np.array([dx, dy, dz])
        pts = pts + offsets

        # Clamp points back inside box (for large jitter)
        pts = np.clip(pts, bbox_min, bbox_max)

    return pts.astype(np.float32)

def sample_latin_hypercube_box(bbox_min, bbox_max, N, seed=None):
    """
    Latin Hypercube points uniformly distributed and well spread in 3D.
    O(N). No NN or FPS needed.

    Returns: (N, 3) float32
    """
    rng = np.random.default_rng(seed)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    L = bbox_max - bbox_min

    # One point per stratum along each axis, then shuffle per-dim
    u = (rng.random((N, 3)) + np.arange(N)[:, None]) / N  # N strata in [0,1)
    for d in range(3):
        rng.shuffle(u[:, d])
    pts01 = u  # in [0,1)^3
    pts = bbox_min + pts01 * L
    return pts.astype(np.float32)

def triangle_soup_from_points(xyz: torch.Tensor, scale: float):
    """
    Create one regular equilateral triangle per 3D point.

    Each triangle:
      • Has circumradius = `scale`
      • Is centered at origin and lies in the XY plane (normal +Z)
      • Is randomly rotated (uniform over SO(3))
      • Is translated so its center coincides with each input point

    Args:
        xyz:   (N, 3) tensor of 3D point positions (CPU or CUDA)
        scale: float, circumradius of each equilateral triangle

    Returns:
        V: (3N, 3) float tensor of vertex positions
        F: ( N, 3) int32 tensor of face indices (per-triangle)
    """
    if xyz.ndim != 2 or xyz.size(-1) != 3:
        raise ValueError("`xyz` must be shaped (N, 3).")

    N = xyz.size(0)
    if N == 0:
        V = xyz.new_zeros((0, 3))
        F = torch.empty((0, 3), dtype=torch.int32, device=xyz.device)
        return V, F

    dtype, device = xyz.dtype, xyz.device
    r = torch.as_tensor(scale, dtype=dtype, device=device)

    # Base equilateral triangle in XY plane, centered at origin, normal +Z
    c = 0.5
    s = (3.0 ** 0.5) * 0.5  # sqrt(3)/2
    base = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-c,  s, 0.0],
            [-c, -s, 0.0],
        ],
        dtype=dtype,
        device=device,
    ) * r  # (3,3)

    # Random rotation (uniform over SO(3) via random quaternion)
    u1 = torch.rand(N, device=device, dtype=dtype)
    u2 = torch.rand(N, device=device, dtype=dtype)
    u3 = torch.rand(N, device=device, dtype=dtype)
    sqrt1_u1 = torch.sqrt(1.0 - u1)
    sqrt_u1 = torch.sqrt(u1)
    two_pi = torch.tensor(6.283185307179586, dtype=dtype, device=device)
    theta1 = two_pi * u2
    theta2 = two_pi * u3
    qx = sqrt1_u1 * torch.sin(theta1)
    qy = sqrt1_u1 * torch.cos(theta1)
    qz = sqrt_u1  * torch.sin(theta2)
    qw = sqrt_u1  * torch.cos(theta2)

    # Quaternion → rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy),
    ], dim=-1).reshape(N, 3, 3)

    # Apply rotations and translate to points
    rotated = torch.einsum('nij,vj->nvi', R, base)  # (N,3,3)
    rotated = rotated + xyz[:, None, :]             # translate

    # Pack vertices and faces
    V = rotated.reshape(-1, 3).contiguous()         # (3N,3)
    base_idx = (torch.arange(N, device=device, dtype=torch.int32) * 3)
    F = torch.stack([base_idx + 0, base_idx + 1, base_idx + 2], dim=1).contiguous()
    return V, F

def remove_unreferenced_vertices_from_soup(
    verts: torch.Tensor,
    faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove unreferenced vertices from a triangle soup.

    Args:
        verts: (Nv, 3) float tensor of vertex positions.
        faces: (Nf, 3) long tensor of vertex indices.

    Returns:
        new_verts: (Nv', 3) float tensor of kept vertices.
        new_faces: (Nf, 3) long tensor with remapped indices.
    """
    verts = verts.contiguous()
    faces = faces.contiguous()
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    used = torch.unique(faces)
    new_verts = verts[used]

    map_old2new = torch.full(
        (verts.shape[0],), -1, dtype=torch.long, device=faces.device
    )
    map_old2new[used] = torch.arange(
        used.numel(), device=faces.device, dtype=torch.long
    )

    new_faces = map_old2new[faces].int()
    return new_verts, new_faces