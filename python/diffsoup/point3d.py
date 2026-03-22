# python/diffsoup/point3d.py
"""Point-cloud utilities: spacing estimation, triangle-soup initialisation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch


def nn_spacing(points: np.ndarray, reduction: str = "median") -> float:
    """Nearest-neighbour spacing of a 3-D point set.

    Args:
        points:    ``(N, 3)`` array of XYZ coordinates.
        reduction: ``"median"`` (robust default) or ``"mean"``.

    Returns:
        Reduced 1-NN Euclidean distance across all points.
    """
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`points` must be shaped (N, 3).")
    if P.shape[0] < 2:
        return 0.0

    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2, p=2)  # [:,0]=self, [:,1]=1-NN
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0

    r = reduction.lower()
    if r in ("median", "med"):
        return float(np.median(nn))
    if r in ("mean", "avg", "average"):
        return float(np.mean(nn))
    raise ValueError("`reduction` must be one of {'median', 'mean'}.")


def triangle_soup_from_points(xyz: torch.Tensor, scale: float):
    """Create one equilateral triangle per 3-D point.

    Each triangle has circumradius ``scale``, lies initially in the XY plane,
    is randomly rotated (uniform over SO(3)), and is translated so that its
    centroid coincides with the corresponding input point.

    Args:
        xyz:   ``(N, 3)`` tensor of point positions (CPU or CUDA).
        scale: Circumradius of each equilateral triangle.

    Returns:
        V: ``(3N, 3)`` float tensor of vertex positions.
        F: ``(N, 3)``  int32 tensor of face indices.
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

    # Base equilateral triangle (XY plane, centred at origin, normal +Z)
    c = 0.5
    s = (3.0**0.5) * 0.5  # sqrt(3)/2
    base = torch.tensor(
        [[1.0, 0.0, 0.0], [-c, s, 0.0], [-c, -s, 0.0]],
        dtype=dtype,
        device=device,
    ) * r  # (3, 3)

    # Uniform random quaternion  →  rotation matrix
    u1 = torch.rand(N, device=device, dtype=dtype)
    u2 = torch.rand(N, device=device, dtype=dtype)
    u3 = torch.rand(N, device=device, dtype=dtype)
    sqrt1_u1 = torch.sqrt(1.0 - u1)
    sqrt_u1 = torch.sqrt(u1)
    two_pi = torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)
    theta1 = two_pi * u2
    theta2 = two_pi * u3

    qx = sqrt1_u1 * torch.sin(theta1)
    qy = sqrt1_u1 * torch.cos(theta1)
    qz = sqrt_u1 * torch.sin(theta2)
    qw = sqrt_u1 * torch.cos(theta2)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(N, 3, 3)

    # Rotate base triangle and translate to each point
    rotated = torch.einsum("nij,vj->nvi", R, base)  # (N, 3, 3)
    rotated = rotated + xyz[:, None, :]

    V = rotated.reshape(-1, 3).contiguous()  # (3N, 3)
    base_idx = torch.arange(N, device=device, dtype=torch.int32) * 3
    F = torch.stack([base_idx, base_idx + 1, base_idx + 2], dim=1).contiguous()
    return V, F


def remove_unreferenced_vertices_from_soup(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove vertices that are not referenced by any face.

    Args:
        verts: ``(Nv, 3)`` float tensor of vertex positions.
        faces: ``(Nf, 3)`` integer tensor of vertex indices.

    Returns:
        new_verts: ``(Nv', 3)`` float tensor of kept vertices.
        new_faces: ``(Nf, 3)``  int32 tensor with remapped indices.
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
    map_old2new[used] = torch.arange(used.numel(), device=faces.device, dtype=torch.long)

    new_faces = map_old2new[faces].int()
    return new_verts, new_faces
