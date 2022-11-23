from typing import *

import torch
import torch.nn as nn

from ..utils import DeepLinearNet


class MeasureBase(nn.Module):
    def __init__(self, shape: torch.Size):
        r"""
        `num_measures` defines how many measures (each of a different Poisson process
        space) this parameterizes.
        """
        super().__init__()
        self.shape = shape
        assert len(shape) == 2, "shape should be [num_quasipartition_mixtures, num_process_per_mixture]"


class LebesgueMeasure(MeasureBase):
    pass


class GaussianBasedMeasure(MeasureBase):
    def __init__(self, shape: torch.Size, *, init_sigma2=1):
        super().__init__(shape)
        self.log_sigma2 = nn.Parameter(torch.empty(shape).fill_(init_sigma2).log_().requires_grad_())
        self.scales_net = DeepLinearNet(shape.numel(), shape.numel(), non_negative=True, bias=True)  # Sec. C.4.3

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_multiple(x)[0]

    def scale_multiple(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        assert all(x.shape[-2:] == self.shape for x in xs)
        return tuple(scaled_x.unflatten(-1, self.shape) for scaled_x in self.scales_net(*[x.flatten(-2) for x in xs]))

    @property
    def sigma2(self):
        return self.log_sigma2.exp()

    @property
    def sigma(self):
        return self.log_sigma2.div(2).exp()
