import torch

class VectorAdam(torch.optim.Optimizer):
    """
    VectorAdam

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for computing running averages of
                                     gradient and its square (default: (0.9, 0.999)).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(VectorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['g1'] = torch.zeros_like(p.data)
                    state['g2'] = torch.zeros_like(p.data[..., :1])

                state['step'] += 1
                g1 = state['g1']
                g2 = state['g2']

                # Update exponential moving averages
                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square().sum(dim=-1, keepdim=True), alpha=1 - b2)

                # Bias correction
                step = state['step']
                m1 = g1 / (1 - b1 ** step)
                m2 = g2 / (1 - b2 ** step)

                # Isotropic normalization
                denom = m2.sqrt() + 1e-8
                update = m1 / denom

                p.data.sub_(update, alpha=lr)
