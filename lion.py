import torch
from torch import nn
from typing import Tuple
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable, required


class Lion(Optimizer):
    """
    Pytorch implementation of Lion optimizer from https://arxiv.org/abs/2302.06675

    Args:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups

    lr (float, optional): learning rate (default: 1e-4)

    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))

    weight_decay (float, optional): weight decay (default: 0)
    """
    def __init__(self,
                params, 
                lr: float = 1e-4, 
                betas: Tuple[float, float] = (.9, .99),
                weight_decay: float = 0
                ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = self.betas
            for p in group['params']:
                if p.grad is not None:

                    state = self.state[p]
                    if "exp_avg" not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                    exp_avg = state['exp_avg']

                    grad = p.grad
                    update = torch.sign((beta1 * exp_avg + (1 - beta1) * grad))
                    update += self.weight_decay * p.data
                    p.add_(update, alpha= -group['lr'])
                    # Update EMA
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
    

