# python/diffsoup/multires.py
"""Multi-resolution triangle features: colour, opacity, and level accumulation."""

import torch
from torch import nn
from typing import Tuple

from . import _core
from . import rasterize as _rz


# ---------------------------------------------------------------------------
#  Level geometry helpers
# ---------------------------------------------------------------------------

def feats_at_level(level: int) -> int:
    """Number of feature samples on a triangle at the given resolution level.

    Level 0 stores one feature per vertex (3 total).  Higher levels
    subdivide the triangle with an increasingly dense Bézier-like lattice.
    """
    assert level >= 0
    return 3 if level == 0 else ((1 << (level - 1)) + 1) * ((1 << level) + 1)


def build_multires_triangle_color(
    T: int,
    min_level: int,
    max_level: int,
    feat_dim: int,
) -> torch.Tensor:
    """Allocate a zero-initialised multi-resolution colour buffer.

    Args:
        T:         Number of triangles.
        min_level: Lowest resolution level (inclusive).
        max_level: Highest resolution level (inclusive).
        feat_dim:  Feature dimensionality per sample.

    Returns:
        ``(T, S, feat_dim)`` float32 tensor where
        ``S = sum(feats_at_level(l) for l in range(min_level, max_level+1))``.
    """
    S = sum(feats_at_level(l) for l in range(min_level, max_level + 1))
    return torch.zeros(T, S, feat_dim, dtype=torch.float32)


# ---------------------------------------------------------------------------
#  Stochastic-opacity rasterisation at a single level
# ---------------------------------------------------------------------------

def rasterize_multires_triangle_alpha(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    tri: torch.Tensor,
    level: int,
    alpha_src: torch.Tensor,
    stochastic: bool = True,
) -> torch.Tensor:
    """Rasterise with per-fragment multi-resolution opacity and depth testing.

    Args:
        resolution:  Image size ``(H, W)``.
        pos:         Homogeneous vertex positions ``(B, V, 4)``, float32 CUDA.
        tri:         Triangle indices ``(T, 3)``, int32 CUDA.
        level:       Multi-resolution level (≥ 0).
        alpha_src:   Per-triangle opacity features ``(T, S, 1)``, float32 CUDA,
                     where ``S = feats_at_level(level)``.
        stochastic:  If ``True`` (default), the alpha threshold is sampled
                     uniformly per fragment; otherwise a fixed 0.5 threshold
                     is used.

    Returns:
        rast_out: ``(B, H, W, 4)`` float32 CUDA — each pixel stores
        ``(bary0, bary1, z, triangle_id+1)``.
    """
    H, W = resolution
    B, V, _ = pos.shape
    T, _ = tri.shape
    _, S, _ = alpha_src.shape
    dev = pos.device

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert tri.shape == (T, 3) and tri.dtype == torch.int32 and tri.is_contiguous()
    assert alpha_src.shape == (T, S, 1) and alpha_src.dtype == torch.float32 and alpha_src.is_contiguous()
    assert S == feats_at_level(level)
    assert pos.is_cuda and tri.device == dev and alpha_src.device == dev

    frag_pix, frag_attrs = _rz._compute_fragments((H, W), pos, tri)
    num_frags = frag_pix.shape[0]

    min_level = max_level = level
    alpha_src_2d = alpha_src.squeeze(-1)
    frag_alpha = torch.zeros(num_frags, dtype=torch.float32, device=dev)
    _core.multires_triangle_alpha(frag_attrs, min_level, max_level, alpha_src_2d, frag_alpha)

    if stochastic:
        alpha_thresh = torch.rand(num_frags, dtype=torch.float32, device=dev)
    else:
        alpha_thresh = torch.full((num_frags,), 0.5, dtype=torch.float32, device=dev)

    return _rz._depth_test((H, W), pos, frag_pix, frag_attrs, frag_alpha, alpha_thresh)


# ---------------------------------------------------------------------------
#  Colour MLP (neural shading)
# ---------------------------------------------------------------------------

class ColorMLP(nn.Module):
    """Small MLP that maps rasterised features to RGB colour.

    The network produces a residual correction blended with a base-colour
    input using a per-pixel resolution weight:

        ``output = (1 - res) * rgb_base + res * mlp(x)``

    where ``rgb_base = x[..., :3]`` and ``res = x[..., 3:4]``.

    Args:
        input_dim:  Number of input features.
        output_dim: Number of output channels (typically 3 for RGB).
        hidden_dim: Hidden-layer width (default 16).
        n_layers:   Number of hidden layers (default 2).
        zero_last:  If ``True``, initialise the output layer to zero so the
                    network starts near the identity (residual-friendly).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 16,
        n_layers: int = 2,
        zero_last: bool = False,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(*layers)
        self._init_weights(zero_last=zero_last)

    # -- initialisation -----------------------------------------------------

    def _init_weights(self, zero_last: bool = False):
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        *hidden, last = linears

        for lin in hidden:
            nn.init.kaiming_uniform_(lin.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(lin.bias)

        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            nn.init.xavier_uniform_(last.weight, gain=1.0)
            nn.init.zeros_(last.bias)

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x:    ``(B, H, W, input_dim)`` — concatenated rasterised features.
            mask: Optional ``(B, H, W)`` bool mask.  When provided, the MLP
                  is evaluated only at ``True`` pixels.

        Returns:
            ``(B, H, W, output_dim)`` blended output.
        """
        B, H, W, _ = x.shape
        assert x.shape == (B, H, W, self.input_dim)
        assert mask is None or mask.shape == (B, H, W)

        rgb = x[..., :3]
        res = x[..., 3:4]

        if mask is not None:
            x_flat = x.view(-1, self.input_dim)
            mask_flat = mask.view(-1)

            output_flat = torch.zeros(
                B * H * W, self.output_dim, device=x.device, dtype=x.dtype
            )
            if mask_flat.any():
                valid_input = x_flat[mask_flat]
                valid_output = self.mlp(valid_input)

                valid_rgb = rgb[mask]
                valid_res = res[mask]
                valid_output = (1.0 - valid_res) * valid_rgb + valid_res * valid_output
                output_flat[mask_flat] = valid_output

            return output_flat.view(B, H, W, self.output_dim)

        y = self.mlp(x.view(-1, self.input_dim)).view(B, H, W, self.output_dim)
        return (1.0 - res) * rgb + res * y


# ---------------------------------------------------------------------------
#  Multi-resolution colour interpolation (differentiable)
# ---------------------------------------------------------------------------

class _MultiresTriangleColorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rast, min_level, max_level, feat):
        B, H, W, _ = rast.shape
        _, _, feat_dim = feat.shape
        dev = rast.device

        out = torch.zeros((B, H, W, feat_dim), dtype=torch.float32, device=dev)
        _core.multires_triangle_color(rast, min_level, max_level, feat, out)

        ctx.save_for_backward(
            rast,
            torch.tensor([min_level, max_level], dtype=torch.int32),
            feat,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rast, levels, feat = ctx.saved_tensors
        min_level = levels[0].item()
        max_level = levels[1].item()

        grad_out = grad_out.contiguous()
        grad_feat = torch.zeros_like(feat)
        _core.backward_multires_triangle_color(
            rast, min_level, max_level, grad_feat, grad_out
        )
        return None, None, None, grad_feat


def multires_triangle_color(
    rast: torch.Tensor,
    level: int,
    feat: torch.Tensor,
) -> torch.Tensor:
    """Interpolate per-triangle colour features at a single resolution level.

    Args:
        rast:  Rasterisation buffer ``(B, H, W, 4)``, float32 CUDA.
        level: Multi-resolution level (≥ 0).
        feat:  Per-triangle colour features ``(T, S, C)``, float32 CUDA,
               where ``S = feats_at_level(level)`` and ``C`` is the colour
               dimensionality.

    Returns:
        ``(B, H, W, C)`` float32 CUDA — interpolated colours per pixel.
    """
    B, H, W, _ = rast.shape
    _, S, _ = feat.shape
    dev = rast.device

    assert rast.shape == (B, H, W, 4) and rast.dtype == torch.float32 and rast.is_contiguous()
    assert feat.dim() == 3 and feat.dtype == torch.float32 and feat.is_contiguous()
    assert S == feats_at_level(level)
    assert rast.is_cuda and feat.device == dev

    min_level = max_level = level
    return _MultiresTriangleColorFn.apply(rast, min_level, max_level, feat)


# ---------------------------------------------------------------------------
#  Cross-level accumulation (differentiable)
# ---------------------------------------------------------------------------

class _AccumulateToLevelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, min_level, max_level, target_level, feat):
        T, _, feat_dim = feat.shape
        dev = feat.device

        S_L = feats_at_level(target_level)
        feat_out = torch.zeros(T, S_L, feat_dim, dtype=torch.float32, device=dev)
        _core.accumulate_to_level_forward(
            min_level, max_level, target_level, feat, feat_out
        )

        ctx.save_for_backward(
            torch.tensor([min_level, max_level, target_level], dtype=torch.int32),
            feat,
        )
        return feat_out

    @staticmethod
    def backward(ctx, grad_feat_out):
        levels, feat = ctx.saved_tensors
        min_level = levels[0].item()
        max_level = levels[1].item()
        target_level = levels[2].item()

        grad_feat_out = grad_feat_out.contiguous()
        grad_feat = torch.zeros_like(feat)
        _core.accumulate_to_level_backward(
            min_level, max_level, target_level, grad_feat, grad_feat_out
        )
        return None, None, None, grad_feat


def accumulate_to_level(
    min_level: int,
    max_level: int,
    feat: torch.Tensor,
    target_level: int | None = None,
) -> torch.Tensor:
    """Accumulate multi-resolution features down to a single target level.

    Combines the contributions of all levels in ``[min_level, max_level]``
    into the sampling pattern of ``target_level``.

    Args:
        min_level:    Lowest stored level (inclusive).
        max_level:    Highest stored level (inclusive).
        feat:         ``(T, S_total, C)`` float32 CUDA — concatenated features
                      across all stored levels.
        target_level: Level whose lattice is used for the output
                      (default: ``max_level``).

    Returns:
        ``(T, S_target, C)`` float32 CUDA — accumulated features.
    """
    assert min_level >= 0
    assert feat.ndim == 3 and feat.dtype == torch.float32
    assert feat.is_contiguous() and feat.is_cuda
    if target_level is None:
        target_level = max_level
    return _AccumulateToLevelFn.apply(min_level, max_level, target_level, feat)
