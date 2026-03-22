# python/diffsoup/remesh.py
"""Adaptive triangle-soup subdivision (world-space and clip-space)."""

import numpy as np
import torch
from . import _core


def split_triangle_soup(
    verts: torch.Tensor,   # (N, 3), float32
    faces: torch.Tensor,   # (M, 3), int32
    num_splits: int,
    tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a triangle soup by repeatedly bisecting the longest edge.

    Args:
        verts:      ``(N, 3)`` float32 vertex positions.
        faces:      ``(M, 3)`` int32 triangle indices.
        num_splits: Maximum number of edge bisections.  Use ``-1`` together
                    with ``tau > 0`` for threshold-only mode.
        tau:        Stop once the longest remaining edge ≤ ``tau``
                    (0 disables the threshold).

    Returns:
        out_verts:    ``(N', 3)`` float32 vertex positions.
        out_faces:    ``(M', 3)`` int32 triangle indices.
        face_mapping: ``(M',)`` int32 — maps each output face to its input
                      face index.
        face_flags:   ``(M',)`` int32 — 1 if the face is an exact copy of the
                      original, 0 otherwise.
    """
    assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be (N, 3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be (M, 3)"

    dev = verts.device
    v_np = verts.float().detach().cpu().contiguous().numpy()
    f_np = faces.to(torch.int32).detach().cpu().contiguous().numpy()

    out_v, out_f, out_map, out_flag = _core.split_triangle_soup(
        v_np, f_np, int(num_splits), float(tau)
    )

    return (
        torch.from_numpy(out_v).to(device=dev, dtype=torch.float32),
        torch.from_numpy(out_f).to(device=dev, dtype=torch.int32),
        torch.from_numpy(out_map).to(device=dev, dtype=torch.int32),
        torch.from_numpy(out_flag).to(device=dev, dtype=torch.int32),
    )


def split_triangle_soup_until(
    verts: torch.Tensor,
    faces: torch.Tensor,
    tau: float,
    hard_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split until all edges ≤ ``tau``.

    Args:
        verts:    ``(N, 3)`` float32 vertex positions.
        faces:    ``(M, 3)`` int32 triangle indices.
        tau:      Maximum allowed edge length.
        hard_cap: Optional upper bound on the number of bisections.

    Returns:
        Same four-tuple as :func:`split_triangle_soup`.
    """
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup(verts, faces, ns, tau=float(tau))


def split_triangle_soup_clip(
    resolution: tuple[int, int],
    mvp: torch.Tensor,           # (4, 4)
    verts: torch.Tensor,         # (N, 3)
    faces: torch.Tensor,         # (M, 3)
    valid_faces: torch.Tensor,   # (M,)
    num_splits: int,
    tau_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a triangle soup based on image-space (clip-space) edge lengths.

    Pipeline:  world ``(N, 3)`` → clip ``(N, 4)`` via MVP → split in
    clip space → back to world ``(N', 3)``.

    Image-space length is measured in normalised device coordinates
    ``(x/w, y/w)`` with the x axis scaled by ``W/H`` so that the metric
    unit equals one image height.

    Edges whose **both** endpoints lie outside the NDC cube ``[-1, 1]³``
    are skipped.

    Args:
        resolution:  ``(H, W)`` image resolution.
        mvp:         ``(4, 4)`` model-view-projection matrix (row-major).
        verts:       ``(N, 3)`` float32 world-space positions.
        faces:       ``(M, 3)`` int32 triangle indices.
        valid_faces: ``(M,)`` int32 mask (1 = consider for splitting).
        num_splits:  Maximum bisections (``-1`` for threshold-only).
        tau_ratio:   Threshold in image-height units (0 disables).

    Returns:
        Same four-tuple as :func:`split_triangle_soup`.
    """
    H, W = resolution
    assert mvp.shape == (4, 4)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert valid_faces.shape == (faces.shape[0],)

    dev = verts.device

    verts = verts.float().contiguous()
    faces = faces.to(torch.int32).contiguous()
    valid_faces = valid_faces.to(torch.int32).contiguous()
    mvp = mvp.float().contiguous()

    aspect_wh = float(W) / float(H)
    out_v, out_f, out_map, out_flag = _core.split_triangle_soup_clip(
        mvp.detach().cpu().numpy(),
        verts.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        valid_faces.detach().cpu().numpy(),
        int(num_splits),
        float(tau_ratio),
        aspect_wh,
    )

    return (
        torch.from_numpy(out_v).to(device=dev, dtype=torch.float32),
        torch.from_numpy(out_f).to(device=dev, dtype=torch.int32),
        torch.from_numpy(out_map).to(device=dev, dtype=torch.int32),
        torch.from_numpy(out_flag).to(device=dev, dtype=torch.int32),
    )


def split_triangle_soup_clip_until(
    resolution: tuple[int, int],
    mvp: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    valid_faces: torch.Tensor,
    tau_ratio: float,
    hard_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split until all image-space edges ≤ ``tau_ratio``.

    Args:
        resolution:  ``(H, W)`` image resolution.
        mvp:         ``(4, 4)`` MVP matrix.
        verts:       ``(N, 3)`` float32 world-space positions.
        faces:       ``(M, 3)`` int32 triangle indices.
        valid_faces: ``(M,)`` int32 mask.
        tau_ratio:   Threshold in image-height units.
        hard_cap:    Optional upper bound on the number of bisections.

    Returns:
        Same four-tuple as :func:`split_triangle_soup`.
    """
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup_clip(
        resolution, mvp, verts, faces, valid_faces, ns, tau_ratio
    )


def expand_by_index(
    source: torch.Tensor,
    index_map: torch.Tensor,
) -> torch.Tensor:
    """Gather rows from ``source`` according to ``index_map``.

    Useful for propagating per-face features through a subdivision step:
    if face *j* in the output descends from face *i* in the input, set
    ``index_map[j] = i`` and call this function to copy the features.

    Args:
        source:    ``(N, ...)`` tensor of parent features.
        index_map: ``(N',)`` long tensor of indices into ``[0, N)``.

    Returns:
        ``(N', ...)`` tensor with ``result[j] = source[index_map[j]]``.
    """
    if not torch.is_tensor(source):
        raise TypeError("`source` must be a torch.Tensor.")
    if not torch.is_tensor(index_map):
        raise TypeError("`index_map` must be a torch.Tensor.")

    if index_map.dtype != torch.long:
        index_map = index_map.to(torch.long)
    index_map = index_map.to(source.device)

    N = source.size(0)
    if torch.any((index_map < 0) | (index_map >= N)):
        raise ValueError("`index_map` contains out-of-range indices.")

    return source.index_select(0, index_map)
