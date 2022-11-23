from typing import *

import abc
import math

import torch
import torch.nn as nn

from .utils import DeepLinearNet, multidot, sigmoid_pow


class ReductionBase(nn.Module, metaclass=abc.ABCMeta):
    input_num_components: int
    discount: Optional[float]

    def __init__(self, input_num_components: int, discount: Optional[float] = None) -> None:
        super().__init__()
        self.input_num_components = input_num_components
        self.discount = discount

    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        if self.discount is None:
            raise RuntimeError(f"{self} does not support non-discounted distances")
        return self.reduce_discounted_distance(d).log() / math.log(self.discount)

    def reduce_discounted_distance(self, d: torch.Tensor) -> torch.Tensor:
        return self.discount ** self.reduce_distance(d)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if self.discount is None:
            return self.reduce_distance(d)
        else:
            return self.reduce_discounted_distance(d)

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        # Manually define for typing
        # https://github.com/pytorch/pytorch/issues/45414
        return super().__call__(d)

    def extra_repr(self) -> str:
        if self.discount is None:
            return f"input_num_components={self.input_num_components}"
        else:
            return f"input_num_components={self.input_num_components}, discount={self.discount:g}"


class Max(ReductionBase):
    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        return d.max(dim=-1).values


class Sum(ReductionBase):
    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        return d.sum(dim=-1)


class Mean(ReductionBase):
    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        return d.mean(dim=-1)


class MaxMean(ReductionBase):
    r'''
    `maxmean` from Neural Norms paper:
    https://arxiv.org/abs/2002.05825

    Implementation follows the official implementation:
    https://github.com/spitis/deepnorms/blob/6c8db1b1178eb92df23149c6d6bfb10782daac86/metrics_tf1.py#L26
    '''

    def __init__(self, input_num_components: int, discount: Optional[float] = None) -> None:
        super().__init__(input_num_components=input_num_components, discount=discount)
        self.raw_alpha = nn.Parameter(torch.ones(()).neg_().requires_grad_())  # pre sigmoid

    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        alpha: torch.Tensor = self.raw_alpha.sigmoid()
        return torch.lerp(
            d.mean(dim=-1),        # * (1 - alpha)
            d.max(dim=-1).values,  # * alpha
            alpha,
        )


class DeepLinearNetWeightedSum(ReductionBase):
    r'''
    PQE-style aggregation by weighted sum from deep linear networks:
    https://arxiv.org/abs/2206.15478

    When using `discount`, we follow the original paper Sec. C.4.2 (and official PQE repository), and use a deep linear
    network to parametrize the input to a `sigmoid` function, whose output is used in an exponentiation.
    I.e., \prod_i sigmoid( deep_lienar_net_output[i] ) ** components[i].
    '''

    def __init__(self, input_num_components: int, discount: Optional[float] = None) -> None:
        super().__init__(input_num_components=input_num_components, discount=discount)
        if self.discount is None:
            self.alpha_net = DeepLinearNet(input_dim=input_num_components, output_dim=1, non_negative=True)
        else:
            self.beta_net = DeepLinearNet(input_dim=1, output_dim=input_num_components, non_negative=False)

            # Initialize logits so initial output is between 0.5 and 0.75. (Sec. C.4.3)
            #
            # Note that this is important since we are multiplying a bunch of things < 1 together,
            # and thus must take care to not make the result close to 0.
            #
            # Say the quasipartitions are 0.5. For output = y, with k quasipartitions,
            # we want the base to be roughly
            #   - log ( y^{-2/k} - 1).

            k = input_num_components
            low_out = 0.5
            high_out = 0.75
            low = -math.log(low_out ** (-2 / k) - 1)
            high = -math.log(high_out ** (-2 / k) - 1)
            # `DeepLinearNet` should initialize s.t. the collapsed vector roughly
            # has zero mean and 1 variance. This holds even for intermediate activations.
            # NB that we crucially used `in_dim=1` rather than `out_dim=1`, which will make
            # weights have variance O(1/n).

            ms: List[torch.Tensor] = list(self.beta_net.mats)

            with torch.no_grad():
                # collapse all but last
                out_before_last: torch.Tensor = multidot(ms[1:])
                norm_out_before_last: torch.Tensor = out_before_last.norm()
                unit_out_before_last: torch.Tensor = out_before_last/ out_before_last.norm()

                # now simply constrain the projection dimension
                ms[0].sub_((ms[0] @ unit_out_before_last) @ unit_out_before_last.T) \
                        .add_(torch.empty(k, 1).uniform_(low, high).div(norm_out_before_last) @ unit_out_before_last.T)  # noqa: E501
                q = self.beta_net.collapse().squeeze(1).sigmoid().pow(0.5).prod().item()
                assert low_out <= q <= high_out, q


    def reduce_distance(self, d: torch.Tensor) -> torch.Tensor:
        return self.alpha_net(d).squeeze(-1)

    def reduce_discounted_distance(self, d: torch.Tensor) -> torch.Tensor:
        logits = self.beta_net.collapse().squeeze(1)
        return sigmoid_pow(logits, d).prod(-1)


REDUCTIONS: Mapping[str, Type[ReductionBase]] = dict(
    sum=Sum,
    mean=Mean,
    maxmean=MaxMean,
    max=Max,
    deep_linear_net_weighted_sum=DeepLinearNetWeightedSum,
)


def make_reduction(kind: str, input_num_components: int, discount: Optional[float] = None) -> ReductionBase:
    return REDUCTIONS[kind](input_num_components, discount)
