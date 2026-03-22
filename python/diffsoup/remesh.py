import numpy as np
import torch
from . import _core

def split_triangle_soup(
    verts: torch.Tensor,   # [num_verts, 3], float32
    faces: torch.Tensor,   # [num_faces, 3], int32
    num_splits: int,
    tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a triangle soup mesh by repeatedly bisecting the longest edges.

    Args:
        verts: [N, 3] float32 tensor of vertex positions.
        faces: [M, 3] int32 tensor of triangle vertex indices.
        num_splits: maximum number of edge splits; use -1 for “until” mode
                    (will require tau > 0 in the C++ code).
        tau: stop once the longest edge length <= tau (0 disables threshold).

    Returns:
        out_verts:   [N', 3] float32 tensor
        out_faces:   [M', 3] int32 tensor
        face_mapping:[M']    int32 tensor mapping each output face to its input face id
        face_flags:  [M']    int32 tensor, 1 = exactly original face, else 0
    """
    assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be [N,3]"
    assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be [M,3]"

    dev = verts.device

    # Ensure correct dtypes & CPU numpy for nanobind
    v_np = (verts if verts.dtype == torch.float32 else verts.float()).detach().cpu().contiguous().numpy()
    f_np = (faces if faces.dtype == torch.int32   else faces.to(torch.int32)).detach().cpu().contiguous().numpy()

    # Call nanobind core: returns NumPy arrays (C-contiguous)
    out_v_np, out_f_np, out_map_np, out_flag_np = _core.split_triangle_soup(v_np, f_np, int(num_splits), float(tau))

    # Convert back to torch (shares memory; safe since arrays are newly created)
    out_verts   = torch.from_numpy(out_v_np).to(device=dev)
    out_faces   = torch.from_numpy(out_f_np).to(device=dev)
    face_mapping= torch.from_numpy(out_map_np).to(device=dev)
    face_flags  = torch.from_numpy(out_flag_np).to(device=dev)

    # Sanity: enforce expected dtypes on the way out (in case NumPy defaulted oddly)
    if out_verts.dtype != torch.float32:
        out_verts = out_verts.to(torch.float32)
    if out_faces.dtype != torch.int32:
        out_faces = out_faces.to(torch.int32)
    if face_mapping.dtype != torch.int32:
        face_mapping = face_mapping.to(torch.int32)
    if face_flags.dtype != torch.int32:
        face_flags = face_flags.to(torch.int32)

    return out_verts, out_faces, face_mapping, face_flags

def split_triangle_soup_until(
    verts: torch.Tensor,
    faces: torch.Tensor,
    tau: float,
    hard_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper: split until all edges <= tau.
    `hard_cap` optionally limits the total splits (defensive).
    """
    # num_splits = -1 engages the "until" mode on the C++ side.
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup(verts, faces, ns, tau=float(tau))

def expand_by_index(
    source: torch.Tensor,
    index_map: torch.Tensor,
) -> torch.Tensor:
    """
    Create a 'child' tensor by gathering rows (or first-dim slices) from `source`
    according to `index_map`.

    Args:
        source (torch.Tensor): Tensor of shape (N, ...) — the parent features.
        index_map (torch.Tensor): Long tensor of shape (N') — indices into [0, N-1],
                                  specifying the parent of each new element.

    Returns:
        torch.Tensor: Tensor of shape (N', ...) where each entry is copied from
                      source[index_map[i]].

    Example:
        >>> parents = torch.randn(5, 3, 4)
        >>> mapping = torch.tensor([0, 2, 2, 4, 1])
        >>> children = expand_by_index(parents, mapping)
        >>> children.shape
        torch.Size([5, 3, 4])
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

    # Works for any trailing shape
    result = source.index_select(0, index_map)
    return result


# TODO: this function needs refactoring
def split_triangle_soup_clip(
    resolution: tuple[int, int], # (H, W) image resolution.
    mvp: torch.Tensor,           # [4,4] MVP (row-major). Applied as v_clip = v_h @ mvp^T
    verts: torch.Tensor,         # [N,3] world-space xyz (float32)
    faces: torch.Tensor,         # [M,3] int32
    valid_faces: torch.Tensor,   # [M,]  int32
    num_splits: int,
    tau_ratio: float = 0.0,      # tau as ratio of image height
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a triangle soup based on an image-space edge lengths.

    Pipeline:
      world [N,3] --(MVP)--> clip [N,4] --(_core.split_triangle_soup_clip)-->
      clip' [N',4] --(inv MVP + divide)--> world' [N',3]

    Notes:
      • Image-space length is measured on NDC (x/w, y/w), with dx scaled by W/H (“image-height units”).
      • Edges with BOTH endpoints outside the NDC cube [-1,1]^3 are ignored.
      • `tau_ratio` compares directly to that metric (height = 1.0).
    """
    H, W = resolution
    assert mvp.shape == (4, 4), "mvp must be [4,4]"
    assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be [N,3]"
    assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be [M,3]"
    assert valid_faces.shape == (faces.shape[0],),  "valid_faces must be [M,]"

    dev = verts.device
    dtype = torch.float32

    # Dtypes / contiguity
    verts = (verts if verts.dtype == dtype else verts.to(dtype)).contiguous()
    faces = (faces if faces.dtype == torch.int32 else faces.to(torch.int32)).contiguous()
    valid_faces = (valid_faces if valid_faces.dtype == torch.int32 else valid_faces.to(torch.int32)).contiguous()
    mvp = (mvp if mvp.dtype == dtype else mvp.to(dtype)).contiguous()

    # Backend call (clip-space)
    aspect_wh = float(W) / float(H)
    mvp_np = mvp.detach().cpu().contiguous().numpy()
    v_np = verts.detach().cpu().contiguous().numpy()
    f_np = faces.detach().cpu().contiguous().numpy()
    vf_np = valid_faces.detach().cpu().contiguous().numpy()

    out_v_np, out_f_np, out_map_np, out_flag_np = _core.split_triangle_soup_clip(
        mvp_np, v_np, f_np, vf_np, int(num_splits), float(tau_ratio), float(aspect_wh)
    )

    # Convert back to torch on original device
    out_verts    = torch.from_numpy(out_v_np).to(device=dev, dtype=dtype)
    out_faces    = torch.from_numpy(out_f_np).to(device=dev, dtype=torch.int32)
    face_mapping = torch.from_numpy(out_map_np).to(device=dev, dtype=torch.int32)
    face_flags   = torch.from_numpy(out_flag_np).to(device=dev, dtype=torch.int32)

    return out_verts, out_faces, face_mapping, face_flags

# TODO: this one too
def split_triangle_soup_clip_until(
    resolution: tuple[int, int],
    mvp: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    valid_faces: torch.Tensor,
    tau_ratio: float,
    hard_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper: split until all image-space edges <= tau_ratio.
    `hard_cap` optionally limits total splits.
    """
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup_clip(
        resolution, mvp, verts, faces, valid_faces, ns, tau_ratio
    )