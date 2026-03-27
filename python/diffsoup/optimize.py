# python/diffsoup/optimize.py
"""Custom optimisers for triangle-soup parameters."""

import torch


class VectorAdam(torch.optim.Optimizer):
    """Adam variant with isotropic second-moment estimation.

    For parameters whose last dimension represents a spatial vector (e.g. XYZ
    positions), the squared-gradient statistics are pooled across that
    dimension so that all components share a single adaptive learning rate.
    This prevents axis-aligned bias in the updates.

    Based on the VectorAdam optimiser introduced in:
        Ling, S. Z., Sharp, N., and Jacobson, A., "VectorAdam for Rotation
        Equivariant Geometry Optimization," NeurIPS 2022.

    Args:
        params: Iterable of parameters to optimise.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages of gradient and its
            squared norm (default: (0.9, 0.999)).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data[..., :1])

                state["step"] += 1
                g1 = state["g1"]
                g2 = state["g2"]

                # Exponential moving averages
                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(
                    grad.square().sum(dim=-1, keepdim=True), alpha=1 - b2
                )

                # Bias correction
                step = state["step"]
                m1 = g1 / (1 - b1**step)
                m2 = g2 / (1 - b2**step)

                # Isotropic normalisation
                p.data.sub_(m1 / (m2.sqrt() + 1e-8), alpha=lr)
