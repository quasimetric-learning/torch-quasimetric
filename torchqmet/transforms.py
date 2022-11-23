from typing import *

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformBase(nn.Module, metaclass=abc.ABCMeta):
    input_num_components: int
    output_num_components: int

    def __init__(self, input_num_components: int, output_num_components: int) -> None:
        super().__init__()
        self.input_num_components = input_num_components
        self.output_num_components = output_num_components

    @abc.abstractmethod
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        # Manually define for typing
        # https://github.com/pytorch/pytorch/issues/45414
        return super().__call__(d)

    def extra_repr(self) -> str:
        return f"input_num_components={self.input_num_components}, output_num_components={self.output_num_components}"


@torch.jit.script
def apply_concave_activations(x: torch.Tensor, bs_first_constant: torch.Tensor, raw_bs_after_first: torch.Tensor,
                              raw_ms: torch.Tensor) -> torch.Tensor:
    bs = torch.cat([bs_first_constant, F.softplus(raw_bs_after_first)], dim=-1)
    ms = torch.sigmoid(raw_ms) * 2
    v = torch.addcmul(bs, x.unsqueeze(-1), ms)
    return v.min(-1).values


class ConcaveActivation(TransformBase):
    r'''
    Learned concave activatations used in neural norms (Deep Norm and Wide Norm):
    https://arxiv.org/abs/2002.05825

    Follows their official implementation:
    https://github.com/spitis/deepnorms/blob/6c8db1b1178eb92df23149c6d6bfb10782daac86/metrics_tf1.py#L30
    '''

    bs_first_constant: torch.Tensor

    def __init__(self, input_num_components: int, num_units_per_input: int = 5):
        super().__init__(input_num_components, input_num_components)
        self.num_units_per_input = num_units_per_input
        self.register_buffer('bs_first_constant', torch.zeros(input_num_components, 1))
        self.raw_bs_after_first = nn.Parameter(
            torch.randn(input_num_components, num_units_per_input - 1).mul_(1e-3).sub_(1).requires_grad_())
        self.raw_ms = nn.Parameter(
            torch.randn(input_num_components, num_units_per_input).mul_(1e-3).requires_grad_())

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return apply_concave_activations(d, self.bs_first_constant, self.raw_bs_after_first, self.raw_ms)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nnum_units_per_input={self.num_units_per_input}"


TRANSFORMS: Mapping[str, Type[TransformBase]] = dict(
    concave_activation=ConcaveActivation,
)

def make_transform(kind: str, input_num_components: int) -> TransformBase:
    return TRANSFORMS[kind](input_num_components)
