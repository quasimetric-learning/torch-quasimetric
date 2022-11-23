from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidPow(torch.autograd.Function):
    # Computes `sigmoid(x)^y` and avoids NaN gradients when x << 0 and sigmoid(x) = 0 numerically.

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        # Compute sigmoid(x)^y = exp( logsigmoid(x) * y ),
        # where logsigmoid(x) = - softplus(-x).
        logsigmoid = F.softplus(-x).neg()
        out = logsigmoid.mul(y).exp()
        ctx.save_for_backward(out, logsigmoid, x, y)
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        # Formula is simple, obtained from mathematica here.
        out, logsigmoid, x, y = ctx.saved_tensors
        gx = ((y + 1) * logsigmoid - x).exp() * y * gout
        gy = logsigmoid * out * gout
        return gx, gy


def sigmoid_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Computes `sigmoid(x)^y` and avoids NaN gradients when x << 0 and sigmoid(x) = 0 numerically.
    return SigmoidPow.apply(x, y)


# https://stackoverflow.com/a/14267825
def ceilpow2(x: int):
    assert x > 0
    return 1 << (x - 1).bit_length()


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'multi_dot'):
    multidot = torch.linalg.multi_dot
else:
    def multidot(mats: List[torch.Tensor]):
        return torch.chain_matmul(*mats)


class DeepLinearNet(nn.Module):
    r"""
    Parametrize a vector/matrix as the matrix multiplication of a couple matrices (Sec. C.4.3).

    By default, this uses 3 hidden layers with dimension max(64, 1 + OutDim', 1 + InDim'),

    where XDim' is the smallest power of 2 that >= XDim.
    """

    bias: Optional[torch.Tensor]

    def __init__(self, input_dim, output_dim, *, hidden_dims=None, non_negative=False, bias=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dim = max(ceilpow2(input_dim), 64)
            hidden_dim = max(ceilpow2(output_dim), hidden_dim)
            hidden_dims = [hidden_dim, hidden_dim, hidden_dim]
        elif not bias and len(hidden_dims) > 0:
            assert min(hidden_dim) >= min(input_dim, output_dim), "cannot lose rank"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.non_negative = non_negative

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        mats = []
        layer_in_dim = dims[0]
        for nh in dims[1:]:
            mats.append(nn.Linear(layer_in_dim, nh, bias=False).weight)  # [linout, linin]
            nn.init.kaiming_normal_(mats[-1], a=1)  # a=1 for linear!
            layer_in_dim = nh
        self.mats = nn.ParameterList(mats[::-1])

        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(self.output_dim, self.input_dim), requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def collapse(self) -> torch.Tensor:
        # Returns A of Ax
        m: torch.Tensor = multidot(list(self.mats))
        if self.bias is not None:
            m = m + self.bias
        if self.non_negative:
            m = m.pow(2)
        return m

    def forward(self, *xs: torch.Tensor):
        m = self.collapse().T
        if len(xs) == 1:
            return xs[0] @ m
        else:
            return tuple(x @ m for x in xs)

    def __call__(self, *xs: torch.Tensor) -> torch.Tensor:
        return super().__call__(*xs)

    def extra_repr(self) -> str:
        return f"bias={self.bias is not None}, non_negative={self.non_negative}"

    def get_num_effective_nparameters(self):
        return self.input_dim * self.output_dim


def get_num_effective_parameters(module: nn.Module) -> int:
    total = 0

    def add(m: nn.Module):
        nonlocal total
        if isinstance(m, DeepLinearNet):
            total += m.get_num_effective_nparameters()
        else:
            total += sum(p.numel() for p in m.parameters(recurse=False))
            for c in m.children():
                add(c)

    add(module)
    return total
