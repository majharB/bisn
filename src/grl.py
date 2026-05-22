import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# =============================================================================
# GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(Function):
    """Identity in the forward pass; negates and scales gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None


class GRL(nn.Module):
    def __init__(self, lambda_val: float = 0.0):
        super().__init__()
        self.lambda_val = torch.tensor(lambda_val, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, new_lambda: float):
        self.lambda_val = torch.tensor(
            new_lambda, requires_grad=False
        ).to(self.lambda_val.device)



# =============================================================================
# LAMBDA ANNEALING SCHEDULE (DANN)
# =============================================================================

def get_lambda(epoch: int, max_epochs: int, max_lambda: float = 1.0) -> float:
    """
    Anneals lambda from 0 to max_lambda using a sigmoid schedule.
    Matches the schedule in Ganin et al. (2016) with p in [0, 1].
    """
    p = epoch / max_epochs
    return max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
