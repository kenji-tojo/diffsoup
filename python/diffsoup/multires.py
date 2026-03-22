import torch
from torch import nn
from typing import Tuple
from . import _core

from . import rasterize as _rz

def feats_at_level(level: int) -> int:
    assert level >= 0
    return 3 if level == 0 else ((1 << (level - 1)) + 1) * ((1 << level) + 1)

def build_multires_triangle_color(
    T: int,
    min_level: int,
    max_level: int,
    feat_dim: int,
) -> torch.Tensor:
    S = 0
    for level in range(min_level, max_level + 1):
        S += feats_at_level(level)
    return torch.zeros(T, S, feat_dim, dtype=torch.float32)

def rasterize_multires_triangle_alpha(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    tri: torch.Tensor,
    level: int,
    alpha_src: torch.Tensor,
    stochastic=True,
) -> torch.Tensor:
    """
    Perform full software rasterization with depth testing.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri (torch.Tensor): Triangle indices (T, 3), int32 CUDA.
        level: int (>=0).

    Returns:
        rast_out: Rasterized output (B, H, W, 4), float32 CUDA. Each pixel stores (bary0, bary1, z, triangle_id+1).
    """
    H, W = resolution
    B, V, _ = pos.shape
    dev = pos.device
    T, _ = tri.shape
    _, S, _ = alpha_src.shape

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert tri.shape == (T, 3) and tri.dtype == torch.int32 and tri.is_contiguous()
    assert alpha_src.shape == (T, S, 1) and alpha_src.dtype == torch.float32 and alpha_src.is_contiguous()
    assert S == feats_at_level(level)
    assert pos.is_cuda
    assert tri.device == dev
    assert alpha_src.device == dev

    frag_pix, frag_attrs = _rz._compute_fragments((H, W), pos, tri)
    num_frags = frag_pix.shape[0]

    min_level = max_level = level
    alpha_src = alpha_src.squeeze(-1)
    frag_alpha = torch.zeros(num_frags, dtype=torch.float32, device=dev)

    _core.multires_triangle_alpha(frag_attrs, min_level, max_level, alpha_src, frag_alpha)

    if stochastic: alpha_thresh = torch.rand(num_frags, dtype=torch.float32, device=dev)
    else:          alpha_thresh = torch.full((num_frags,), fill_value=0.5, dtype=torch.float32, device=dev)

    rast_out = _rz._depth_test((H, W), pos, frag_pix, frag_attrs, frag_alpha, alpha_thresh)
    return rast_out

# class ColorMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         hidden_dim=16,
#         n_layers=2,
#         zero_last=False,
#         use_skip=True,       # <- new
#     ):
#         super().__init__()

#         self.input_dim  = input_dim
#         self.output_dim = output_dim
#         self.use_skip   = use_skip

#         # -------- trunk (hidden layers) --------
#         layers = []
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())

#         for _ in range(n_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())

#         self.trunk = nn.Sequential(*layers)          # (.., hidden_dim)
#         self.out   = nn.Linear(hidden_dim, output_dim)

#         # skip proj: input_dim -> output_dim
#         if use_skip:
#             self.skip = nn.Linear(input_dim, output_dim)
#         else:
#             self.skip = None

#         self.final_activation = nn.Sigmoid()
#         self._init_weights(zero_last=zero_last)

#     def _init_weights(self, zero_last=False):
#         # Kaiming for trunk linears
#         trunk_linears = [m for m in self.trunk if isinstance(m, nn.Linear)]
#         for lin in trunk_linears:
#             nn.init.kaiming_uniform_(lin.weight, a=0.0, mode='fan_in', nonlinearity='relu')
#             nn.init.zeros_(lin.bias)

#         # Output head
#         if zero_last:
#             nn.init.zeros_(self.out.weight)
#             nn.init.zeros_(self.out.bias)
#         else:
#             nn.init.xavier_uniform_(self.out.weight, gain=1.0)
#             nn.init.zeros_(self.out.bias)

#         # Skip proj (if any)
#         if self.skip is not None:
#             if zero_last:
#                 nn.init.zeros_(self.skip.weight)
#                 nn.init.zeros_(self.skip.bias)
#             else:
#                 nn.init.xavier_uniform_(self.skip.weight, gain=1.0)
#                 nn.init.zeros_(self.skip.bias)

#     def _forward_flat(self, x_flat):
#         """
#         x_flat: (N, input_dim)
#         returns: (N, output_dim)
#         """
#         h = self.trunk(x_flat)          # (N, hidden_dim)
#         y = self.out(h)                 # (N, output_dim)

#         if self.use_skip:
#             y = y + self.skip(x_flat)   # residual from input

#         y = self.final_activation(y)    # sigmoid for colors in [0,1]
#         return y

#     def forward(self, x, mask=None):
#         B, H, W, _ = x.shape
#         assert x.shape == (B, H, W, self.input_dim)
#         assert mask is None or mask.shape == (B, H, W)

#         x_flat = x.view(-1, self.input_dim)

#         if mask is not None:
#             mask_flat = mask.view(-1)  # (B*H*W,)

#             output_flat = torch.zeros(
#                 B * H * W,
#                 self.output_dim,
#                 device=x.device,
#                 dtype=x.dtype,
#             )

#             if mask_flat.any():
#                 valid_input  = x_flat[mask_flat]              # (N_valid, D)
#                 valid_output = self._forward_flat(valid_input) # (N_valid, C)
#                 output_flat[mask_flat] = valid_output

#             output = output_flat.view(B, H, W, self.output_dim)
#         else:
#             y_flat = self._forward_flat(x_flat)               # (B*H*W, C)
#             output = y_flat.view(B, H, W, self.output_dim)

#         return output

class ColorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, n_layers=2, zero_last=False):
        super().__init__()

        layers = []
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

    def _init_weights(self, zero_last=False):
        # Collect all Linear layers
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        *hidden, last = linears

        # Hidden layers: Kaiming for ReLU
        for lin in hidden:
            nn.init.kaiming_uniform_(lin.weight, a=0.0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(lin.bias)

        # Final layer:
        if zero_last:
            # Residual-friendly start (network starts near zero output)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            # Xavier for linear output head
            nn.init.xavier_uniform_(last.weight, gain=1.0)
            nn.init.zeros_(last.bias)

    def forward(self, x, mask=None):
        B, H, W, _ = x.shape
        assert x.shape == (B, H, W, self.input_dim)
        assert mask is None or mask.shape == (B, H, W)

        rgb = x[..., :3]
        res = x[..., 3:4]

        if mask is not None:
            x_flat = x.view(-1, self.input_dim)
            mask_flat = mask.view(-1)

            output_flat = torch.zeros(B * H * W, self.output_dim, device=x.device, dtype=x.dtype)
            if mask_flat.any():
                valid_input = x_flat[mask_flat]
                valid_output = self.mlp(valid_input)

                valid_rgb = rgb[mask]
                valid_res = res[mask]
                valid_output = (1.0 - valid_res) * valid_rgb + valid_res * valid_output
                output_flat[mask_flat] = valid_output

            output = output_flat.view(B, H, W, self.output_dim)
        else:
            y = self.mlp(x.view(-1, self.input_dim))
            y = y.view(B, H, W, self.output_dim)
            output = (1.0 - res) * rgb + res * y

        return output

class _MultiresTriangleColorFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rast: torch.Tensor,
        min_level: int,
        max_level: int,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, _ = rast.shape
        _, _, feat_dim = feat.shape
        dev = rast.device

        out = torch.zeros((B, H, W, feat_dim), dtype=torch.float32, device=dev)
        _core.multires_triangle_color(rast, min_level, max_level, feat, out)

        ctx.save_for_backward(rast, torch.tensor([min_level, max_level], dtype=torch.int32), feat)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        rast, levels, feat = ctx.saved_tensors
        min_level = levels[0].item()
        max_level = levels[1].item()

        grad_out = grad_out.contiguous()
        grad_feat = torch.zeros_like(feat)

        _core.backward_multires_triangle_color(
            rast, min_level, max_level,
            grad_feat, grad_out
        )
        return None, None, None, grad_feat

def multires_triangle_color(
    rast: torch.Tensor,
    level: int,
    feat: torch.Tensor,
) -> torch.Tensor:
    B, H, W, _ = rast.shape
    dev = rast.device
    _, S, _ = feat.shape

    assert rast.shape == (B, H, W, 4) and rast.dtype == torch.float32 and rast.is_contiguous()
    assert feat.dim() == 3 and feat.dtype == torch.float32 and feat.is_contiguous()
    assert S == feats_at_level(level)
    assert rast.is_cuda
    assert feat.device == dev

    min_level = max_level = level
    return _MultiresTriangleColorFn.apply(rast, min_level, max_level, feat)

class _AccumulateToLevelFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        min_level: int,
        max_level: int,
        target_level: int,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        T, _, feat_dim = feat.shape
        dev = feat.device

        S_L = feats_at_level(target_level)
        feat_out = torch.zeros(T, S_L, feat_dim, dtype=torch.float32, device=dev)

        _core.accumulate_to_level_forward(
            min_level, max_level, target_level,
            feat, feat_out
        )

        ctx.save_for_backward(
            torch.tensor([min_level, max_level, target_level], dtype=torch.int32),
            feat
        )
        return feat_out

    @staticmethod
    def backward(ctx, grad_feat_out: torch.Tensor):
        levels, feat = ctx.saved_tensors
        min_level    = levels[0].item()
        max_level    = levels[1].item()
        target_level = levels[2].item()

        grad_feat_out = grad_feat_out.contiguous()
        grad_feat = torch.zeros_like(feat)

        _core.accumulate_to_level_backward(
            min_level, max_level, target_level,
            grad_feat, grad_feat_out
        )

        return None, None, None, grad_feat

def accumulate_to_level(
    min_level: int,
    max_level: int,
    feat: torch.Tensor,
    target_level: int = None,
) -> torch.Tensor:
    assert min_level >= 0
    assert feat.ndim == 3 and feat.dtype == torch.float32 and feat.is_contiguous() and feat.is_cuda
    if target_level is None: target_level = max_level
    return _AccumulateToLevelFn.apply(min_level, max_level, target_level, feat)
