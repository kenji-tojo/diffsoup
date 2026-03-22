import torch
from torch import nn
from typing import Tuple
from . import _core

def _filter_valid_fragments(
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TODO: description.
    """
    frag_pix_out = torch.empty_like(frag_pix)
    frag_attrs_out = torch.empty_like(frag_attrs)

    valid_count = _core.filter_valid_fragments(frag_pix, frag_attrs, frag_pix_out, frag_attrs_out)

    frag_pix_out = frag_pix_out[:valid_count].contiguous()
    frag_attrs_out = frag_attrs_out[:valid_count].contiguous()

    return frag_pix_out, frag_attrs_out

def _compute_fragments(
    resolution: Tuple[int, int],
    pos: torch.Tensor,            # (B, V, 4)
    tri: torch.Tensor             # (T, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform software rasterization of a triangle mesh into fragments.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri (torch.Tensor): Triangle indices (T, 3), int32 CUDA.

    Returns:
        frag_pix (torch.Tensor): (N, 3) int32 tensor with (batch, h, w) per valid fragment.
        frag_attrs (torch.Tensor): (N, 4) float32 tensor with (bary0, bary1, z, triangle_id+1).
    """
    H, W = resolution
    B, V, _ = pos.shape
    T, _ = tri.shape

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert tri.shape == (T, 3) and tri.dtype == torch.int32 and tri.is_contiguous()

    device = pos.device

    assert pos.is_cuda
    assert tri.device == device

    rects = torch.empty((B * T, 4), dtype=torch.int32, device=device)
    frag_prefix = torch.empty((B * T), dtype=torch.int32, device=device)
    num_frags = _core.compute_triangle_rects(H, W, pos, tri, rects, frag_prefix)

    frag_pix = torch.full((num_frags, 3), -1, dtype=torch.int32, device=device)
    frag_attrs = torch.empty((num_frags, 4), dtype=torch.float32, device=device)

    _core.compute_fragments(
        H, W, pos, tri, frag_prefix, rects, frag_pix, frag_attrs
    )

    frag_pix, frag_attrs = _filter_valid_fragments(frag_pix, frag_attrs)

    return frag_pix, frag_attrs

def _depth_test(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
    frag_alpha: torch.Tensor,
    alpha_thresh: torch.Tensor,
) -> torch.Tensor:
    """
    Perform the depth test.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        frag_pix: (num_frags, 3) int32 tensor with (batch, h, w) per valid fragment.
        frag_attrs: (num_frags, 4) float32 tensor with (bary0, bary1, z, triangle_id+1).
        frag_alpha: (num_frags,) float32 tensor.
        alpha_thresh: (num_frags,) float32 tensor.

    Returns:
        rast_out: Rasterized output (B, H, W, 4), float32 CUDA. Each pixel stores (bary0, bary1, z, triangle_id+1).
    """
    H, W = resolution
    B, V, _ = pos.shape
    num_frags, _ = frag_pix.shape

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert frag_pix.shape == (num_frags, 3) and frag_pix.dtype == torch.int32 and frag_pix.is_contiguous()
    assert frag_attrs.shape == (num_frags, 4) and frag_attrs.dtype == torch.float32 and frag_attrs.is_contiguous()
    assert frag_alpha.shape == (num_frags,) and frag_alpha.dtype == torch.float32 and frag_alpha.is_contiguous()
    assert alpha_thresh.shape == (num_frags,) and alpha_thresh.dtype == torch.float32 and alpha_thresh.is_contiguous()

    device = pos.device
    assert pos.is_cuda
    assert frag_pix.device == device
    assert frag_attrs.device == device
    assert frag_alpha.device == device
    assert alpha_thresh.device == device

    rast = torch.zeros(B, H, W, 4, dtype=torch.float32, device=device)
    _core.depth_test(frag_pix, frag_attrs, frag_alpha, alpha_thresh, rast)

    return rast

def frag_alpha_mobilenerf(
    frag_attrs: torch.Tensor,
    uvs: torch.Tensor,
    tri: torch.Tensor,
    feat0: torch.Tensor,
) -> torch.Tensor:
    frag_alpha = torch.zeros_like(frag_attrs[:, 0])
    _core.frag_alpha_mobilenerf(
        frag_attrs, uvs, tri, feat0, frag_alpha
    )
    return frag_alpha

def rasterize_mobilenerf(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    uvs: torch.Tensor,
    tri: torch.Tensor,
    feat0: torch.Tensor,
    feat1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform full software rasterization with depth testing.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri (torch.Tensor): Triangle indices (T, 3), int32 CUDA.

    Returns:
        rast_out: Rasterized output (B, H, W, 4), float32 CUDA. Each pixel stores (bary0, bary1, z, triangle_id+1).
    """
    H, W = resolution
    B, V, _ = pos.shape
    dev = pos.device
    T, _ = tri.shape

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert tri.shape == (T, 3) and tri.dtype == torch.int32 and tri.is_contiguous()
    assert pos.is_cuda
    assert tri.device == dev

    frag_pix, frag_attrs = _compute_fragments((H, W), pos, tri)
    num_frags = frag_pix.shape[0]

    frag_alpha = frag_alpha_mobilenerf(frag_attrs, uvs, tri, feat0)
    alpha_thresh = (1.0/512.0) * torch.ones(num_frags, dtype=torch.float32, device=dev)

    rast = _depth_test((H, W), pos, frag_pix, frag_attrs, frag_alpha, alpha_thresh)
    mask = rast[..., -1:].clip(0, 1)

    color = torch.zeros(B, H, W, 8, dtype=torch.float32, device=dev)
    _core.lookup_feats_mobilenerf(
        rast, uvs, tri, feat0, feat1, color
    )

    return color, mask

class _RadianceFieldLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        color: torch.Tensor,       # (B, H, W, C)
        target: torch.Tensor,      # (B, H, W, C)
        rast: torch.Tensor,        # (B, H, W, 4)
        pos: torch.Tensor,
        tri: torch.Tensor,
        level: int,
        alpha_src: torch.Tensor,
    ) -> torch.Tensor:
        _, H, W, _ = rast.shape
        dev = rast.device
        assert alpha_src.ndim == 2

        frag_pix, frag_attrs = _compute_fragments((H, W), pos, tri)
        num_frags = frag_pix.shape[0]

        min_level = max_level = level
        frag_alpha = torch.zeros(num_frags, dtype=torch.float32, device=dev)

        _core.multires_triangle_alpha(
            frag_attrs, min_level, max_level, alpha_src, frag_alpha
        )

        grad_frag_alpha = torch.zeros_like(frag_alpha)
        _core.backward_radiance_field_loss(
            color, target, rast, frag_pix, frag_attrs,
            frag_alpha, grad_frag_alpha
        )

        grad_alpha_src = torch.zeros_like(alpha_src)
        _core.backward_multires_triangle_alpha(
            frag_attrs, min_level, max_level,
            grad_alpha_src, grad_frag_alpha
        )

        weight = 1.0 / color.numel()
        ctx.save_for_backward(grad_alpha_src, torch.tensor([weight], dtype=torch.float32))

        return torch.zeros(1, dtype=torch.float32, device=color.device)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_alpha_src, weight = ctx.saved_tensors

        grad_alpha_src = weight.item() * grad_loss * grad_alpha_src

        return None, None, None, None, None, None, grad_alpha_src

class _EdgeGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        color: torch.Tensor,       # (B, H, W, C)
        rast: torch.Tensor,        # (B, H, W, 4)
        pos: torch.Tensor,         # (B, V, 4)
        tri: torch.Tensor,         # (T, 3)
    ) -> torch.Tensor:
        ctx.save_for_backward(color, rast, pos, tri)

        return color

    @staticmethod
    def backward(ctx, grad_color: torch.Tensor):
        color, rast, pos, tri = ctx.saved_tensors

        grad_color = grad_color.contiguous()
        grad_pos = torch.zeros_like(pos)

        _core.backward_edge_grad(
            color, grad_color, rast, pos, grad_pos, tri
        )

        # return None, None, grad_pos, None
        return grad_color, None, grad_pos, None

def edge_grad(
    color: torch.Tensor,       # (B, H, W, C)
    rast: torch.Tensor,        # (B, H, W, 4)
    pos: torch.Tensor,         # (B, V, 4)
    tri: torch.Tensor,         # (T, 3)
) -> torch.Tensor:
    """
    TODO: description.

    Args:
        color: Current rendering (B, H, W, C), float32 CUDA.
        rast: Rasterization output of shape (B, H, W, 4), float32 CUDA.
        pos: Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri: Triangle indices (T, 3), int32 CUDA.

    Returns:
        loss: float32 CUDA.
        color: Color tensor to propagate loss into edge gradients, (B, H, W, C), float32 CUDA.
    """
    B, H, W, C = color.shape
    _, V, _ = pos.shape
    T, _ = tri.shape

    assert color.is_contiguous() and color.dtype == torch.float32
    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert pos.shape == (B, V, 4) and pos.is_contiguous() and pos.dtype == torch.float32
    assert tri.shape == (T, 3) and tri.is_contiguous() and tri.dtype == torch.int32

    assert color.is_cuda
    assert rast.device == color.device
    assert pos.device == color.device
    assert tri.device == color.device

    color = _EdgeGradFn.apply(color, rast, pos, tri)
    return color

def radiance_field_loss(
    color: torch.Tensor,       # (B, H, W, C)
    target: torch.Tensor,      # (B, H, W, C)
    rast: torch.Tensor,        # (B, H, W, 4)
    pos: torch.Tensor,         # (B, V, 4)
    tri: torch.Tensor,         # (T, 3)
    level: int,
    alpha_src: torch.Tensor,
) -> torch.Tensor:
    """
    TODO: description.

    Args:
        color: Current rendering (B, H, W, C), float32 CUDA.
        target: Target image (B, H, W, C), float32 CUDA.
        rast: Rasterization output of shape (B, H, W, 4), float32 CUDA.
        pos: Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri: Triangle indices (T, 3), int32 CUDA.
        level: int (>=0).

    Returns:
        loss: float32 CUDA.
    """
    B, H, W, C = color.shape
    dev = color.device
    _, V, _ = pos.shape
    T, _ = tri.shape
    _, S, _ = alpha_src.shape

    assert color.is_contiguous() and color.dtype == torch.float32
    assert target.shape == (B, H, W, C) and target.is_contiguous() and target.dtype == torch.float32
    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert pos.shape == (B, V, 4) and pos.is_contiguous() and pos.dtype == torch.float32
    assert tri.shape == (T, 3) and tri.is_contiguous() and tri.dtype == torch.int32
    assert alpha_src.shape == (T, S, 1) and alpha_src.dtype == torch.float32 and alpha_src.is_contiguous()
    assert color.is_cuda
    assert target.device == dev
    assert rast.device == dev
    assert pos.device == dev
    assert tri.device == dev

    alpha_src = alpha_src.squeeze(-1)

    loss = _RadianceFieldLossFn.apply(color, target, rast, pos, tri, level, alpha_src)
    return loss

def encode_view_dir_sh2(
    rast: torch.Tensor,           # (B, H, W, 4)
    inv_mvp: torch.Tensor,        # (B, 4, 4)
) -> torch.Tensor:
    """
    Evaluate spherical harmonics.

    Args:
        rast: Rasterization output of shape (B, H, W, 4), float32 CUDA.
        inv_mvp: Inverse camera transforms of shape (B, 4, 4), float32 CUDA.

    Returns:
        encoding: (B, H, W, n_coeffs), float32 CUDA.
    """
    B, H, W, _ = rast.shape
    dev = rast.device

    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert inv_mvp.shape == (B, 4, 4) and inv_mvp.is_contiguous() and inv_mvp.dtype == torch.float32
    assert rast.is_cuda
    assert inv_mvp.device == dev

    encoding = torch.zeros(B, H, W, 9, dtype=torch.float32, device=dev)

    _core.encode_view_dir_sh2(
        rast, inv_mvp, encoding
    )
    return encoding

def encode_view_dir_freq(
    rast: torch.Tensor,    # [B, H, W, 4]
    inv_mvp: torch.Tensor, # [B, 4, 4]
    freq: float,
    vmf_kappa: float = -1.0
) -> torch.Tensor:
    """
    Compute sinusoidal encoding of view directions.

    Args:
        rast: Rasterization output of shape (B, H, W, 4), float32 CUDA.
        inv_mvp: Inverse camera transforms of shape (B, 4, 4), float32 CUDA.
        freq: float

    Returns:
        encoding: (B, H, W, 9), float323 CUDA.
    """
    B, H, W, _ = rast.shape
    dev = rast.device

    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert inv_mvp.shape == (B, 4, 4) and inv_mvp.is_contiguous() and inv_mvp.dtype == torch.float32
    assert rast.is_cuda
    assert inv_mvp.device == dev

    encoding = torch.zeros(B, H, W, 9, dtype=torch.float32, device=dev)

    if vmf_kappa > 0.0: vmf_samples = torch.rand(B, H, W, 2, dtype=torch.float32, device=dev)
    else:               vmf_samples = torch.empty(0, 0, 0, 2, dtype=torch.float32, device=dev)

    _core.encode_view_dir_freq(
        rast, inv_mvp, float(freq), encoding,
        float(vmf_kappa), vmf_samples
    )
    return encoding

def count_triangle_ids(
    rast: torch.Tensor,
    num_tris: int
) -> torch.Tensor:
    """
    Count how many pixels in the rasterization belong to each triangle.

    Args:
        rast: (B, H, W, 4) float32 CUDA tensor. The last channel stores
            1-based triangle IDs (0 = background).
        num_tris: Total number of triangles.

    Returns:
        (num_tris,) long tensor with per-triangle pixel counts.
    """
    tri_ids = rast[..., -1].long()
    tri_ids = tri_ids[tri_ids > 0] - 1
    count = torch.bincount(tri_ids, minlength=num_tris)
    assert count.shape[0] == num_tris
    return count