r'''
Neural Norms:
https://arxiv.org/abs/2002.05825
'''

from typing import *
import re
import warnings

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import QuasimetricBase



class ConstrainedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool,
                 weight_constraint_: Optional[Callable[[torch.Tensor], None]] = None) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_constraint_ = weight_constraint_
        self.last_constrained_weight_version: int = -100  # any negative number works as init since real ones >= 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_constraint_ is not None and self.weight._version != self.last_constrained_weight_version:
            with torch.no_grad():
                self.weight_constraint_(self.weight)
                self.last_constrained_weight_version = self.weight._version
        return super().forward(input)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", weight_constraint_={self.weight_constraint_}"


@torch.jit.script
def max_relu(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    x = x.unflatten(-1, (x.shape[-1] // 2, 2))
    max_term = x.max(dim=-1).values
    relu_term = (x * weights).relu().mean(-1)
    return torch.cat([max_term, relu_term], dim=-1)


class MaxReLU(nn.Module):
    r'''
    Follows the official implementation:
    https://github.com/spitis/deepnorms/blob/6c8db1b1178eb92df23149c6d6bfb10782daac86/metrics_tf1.py#L15
    '''

    def __init__(self, num_features: int):
        super().__init__()
        assert num_features % 2 == 0, f"MaxReLU only supports *even* number of features, but num_features={num_features}"
        self.num_features = num_features
        self.raw_weights = nn.Parameter(torch.zeros(num_features // 2, 2, requires_grad=True))  # set init to be softplus(0) following original deepnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_relu(x, F.softplus(self.raw_weights))

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"



def make_activation(kind: str, num_features: int) -> nn.Module:
    if kind == 'relu':
        return nn.ReLU()
    elif kind == 'maxrelu':
        return MaxReLU(num_features)
    else:
        raise ValueError(f'Unknown activation {repr(kind)}')


class DeepNorm(QuasimetricBase):
    r'''
    Deep Norm:
    https://arxiv.org/abs/2002.05825

    Follows the official implementation:
    https://github.com/spitis/deepnorms/blob/6c8db1b1178eb92df23149c6d6bfb10782daac86/metrics_pytorch.py#L134

    One-line Usage:

        DeepNorm(input_size: int, num_components: int, num_layers: int = 3, ...)

    NOTE::
        When using `activation="maxrelu", final_activation=None`, this does not guarantee a true quasimetric, since
        using "maxrelu" as the final activation (whose output are the components) does not guarantee non-negative
        components. "maxrelu" is an activation function proposed in the original paper for better expressivity than
        "relu".

        Following a fix proposed in the IQE paper (https://arxiv.org/abs/2211.15120), we allow individually
        configuring `final_activation`.  When using `activation="maxrelu"`, setting `final_activation="relu"` will
        guarantee a quasimetric. Always using ReLU with `activation="relu", final_activation=None` also gurantees a
        quasimetric.

        Without the fix, while the negative values rarely appears in the final reduced quasimetric estimate, it does not
        inherently obey quasimetric constraints anymore, and thus does not have the correct inductive bias. Indeed, the
        IQE paper shows that adding the fix improves Deep Norm performance.

    Deep-Norm-Specific Args:
        input_size (int): Dimension of input latent vectors.
        num_components (int): Number of output components (also controlling hidden layer dimensions).
        num_layers (int): Number of layers.
                          Default: ``3``.
        activation (str): Activation functions.
                          Supported choices: "relu", "maxrelu".
                          Default: "relu"
        final_activation (Optional[str]): Activation function used at last to obtain the components. If not set, use the
                                          same one as `activation`.
                                          Supported choices: None, "relu", "maxrelu".
                                          Default: None
        symmetric (bool): Whether to enforce symmetry (i.e., metric).
                          Default: ``False``.

    Common Args (Exist for all quasimetrics, **Keyword-only**, Default values may be different for different quasimetrics):
        transforms (Collection[str]): A sequence of transforms to apply to the components, before reducing them to form
                                      the final latent quasimetric.
                                      Supported choices:
                                        + "concave_activation": Concave activation transform from Neural Norms paper.
                                      Default: ``("concave_activation",)``.
        reduction (str): Reduction method to aggregate components into final quasimetric value.
                         Supported choices:
                           + "sum": Sum of components.
                           + "max": Max of components.
                           + "mean": Average of components.
                           + "maxmean": Convex combination of max and mean. Used in original Deep Norm, Wide Norm, and IQE.
                           + "deep_linear_net_weighted_sum": Weighted sum with weights given by a deep linear net. Used in
                                                             original PQE, whose components have limited range [0, 1).
                         Default: ``"maxmean"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``, but recommended for PQEs (following original paper).
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  DeepNorms always obey quasimetric constraints if final activation
                                        applied is not "maxrelu".
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        symmetric (bool)
        num_components (int): Number of components to be combined to form the latent quasimetric. For MRN, this is always ``2``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> dn = torchqmet.DeepNorm(128, num_components=64, num_layers=3)
        >>> print(dn)
        DeepNorm(
          guaranteed_quasimetric=True
          input_size=128, num_components=64, discount=None
          symmetric=False
          (transforms): Sequential(
            (0): ConcaveActivation(
              input_num_components=64, output_num_components=64
              num_units_per_input=5
            )
          )
          (reduction): MaxMean(input_num_components=64)
          (u0): ConstrainedLinear(in_features=128, out_features=64, bias=False, weight_constraint_=None)
          (activations): ModuleList(
            (0): ReLU()
            (1): ReLU()
            (2): ReLU()
          )
          (ws_after_first): ModuleList(
            (0): ConstrainedLinear(in_features=64, out_features=64, bias=False, weight_constraint_=<built-in method relu_ of type object at 0x7fefde166060>)
            (1): ConstrainedLinear(in_features=64, out_features=64, bias=False, weight_constraint_=<built-in method relu_ of type object at 0x7fefde166060>)
          )
          (us_after_first): ModuleList(
            (0): ConstrainedLinear(in_features=128, out_features=64, bias=False, weight_constraint_=None)
            (1): ConstrainedLinear(in_features=128, out_features=64, bias=False, weight_constraint_=None)
          )
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(dn(x, y))
        tensor([1.9162, 2.3342, 1.4302, 2.4852, 2.0792], grad_fn=<LerpBackward1>)
        >>> print(dn(y, x))
        tensor([2.2290, 2.2457, 2.7775, 2.1579, 2.2385], grad_fn=<LerpBackward1>)
        >>> print(dn(x[:, None], x))  # pdist
        tensor([[0.0000, 2.0776, 2.4156, 1.9826, 2.5025],
                [2.5139, 0.0000, 2.8350, 2.8957, 2.6522],
                [1.5354, 1.6764, 0.0000, 2.0970, 1.8641],
                [1.9107, 2.4360, 3.0188, 0.0000, 2.3431],
                [2.1608, 1.8930, 2.5621, 2.0461, 0.0000]], grad_fn=<LerpBackward1>)
        >>>
        >>> # DeepNorm that may violate quasimetric constraints
        >>> dn = torchqmet.DeepNorm(128, num_components=64, num_layers=3, activation="maxrelu")
        .../torchqmet/neural_norms.py:105: UserWarning: MRN with final activation function maxrelu may not be a quasimetric (see IQE paper Sec
        . C.1). Use final_activation="relu" to guarantee a quasimetric.
    '''

    symmetric: bool

    def __init__(self, input_size: int, num_components: int, num_layers: int = 3, activation: str = 'relu',
                 final_activation: Optional[str] = None, symmetric: bool = False, *,
                 transforms: Collection[str] = ('concave_activation',), reduction: str = 'maxmean',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        if final_activation is None:
            final_activation = activation
        if final_activation != 'relu':
            guaranteed_quasimetric = False
            if warn_if_not_quasimetric:
                warnings.warn(
                    f'MRN with final activation function {final_activation} may not be a quasimetric (see IQE paper Sec. C.1). '
                    'Use final_activation="relu" to guarantee a quasimetric.')
        else:
            guaranteed_quasimetric = True
        super().__init__(input_size, num_components,
                         transforms=transforms, reduction=reduction, discount=discount,
                         guaranteed_quasimetric=guaranteed_quasimetric, warn_if_not_quasimetric=warn_if_not_quasimetric)
        self.symmetric = symmetric
        self.num_layers = num_layers

        hidden_size = num_components
        self.u0 = ConstrainedLinear(input_size, hidden_size, bias=False)
        self.activations = nn.ModuleList([make_activation(activation, hidden_size)])

        self.ws_after_first = nn.ModuleList()
        self.us_after_first = nn.ModuleList()
        layer_in_size = hidden_size
        for ii in range(1, num_layers):
            if ii == num_layers - 1:
                layer_output_size = num_components
                act = make_activation(final_activation, layer_output_size)
            else:
                layer_output_size = hidden_size
                act = make_activation(activation, layer_output_size)
            self.activations.append(act)
            self.ws_after_first.append(
                ConstrainedLinear(layer_in_size, layer_output_size, bias=False, weight_constraint_=torch.relu_))
            self.us_after_first.append(
                ConstrainedLinear(input_size, layer_output_size, bias=False))
            layer_in_size = layer_output_size

    def norm_asym(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.activations[0](self.u0(x))
        for ii in range(1, self.num_layers):
            h = self.ws_after_first[ii - 1](h) + self.us_after_first[ii - 1](x)
            h = self.activations[ii](h)
        return h

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.symmetric:
            x = torch.stack([x, -x], dim=0)
        h = self.norm_asym(x)
        if self.symmetric:
            h = h.sum(dim=0)
        return h

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x - y)

    def extra_repr(self) -> str:
        return super().extra_repr() + f'\nsymmetric={self.symmetric}'


class WideNorm(QuasimetricBase):
    r'''
    Wide Norm:
    https://arxiv.org/abs/2002.05825

    Follows the official implementation:
    https://github.com/spitis/deepnorms/blob/6c8db1b1178eb92df23149c6d6bfb10782daac86/metrics_pytorch.py#L101

    One-line Usage:

        WideNorm(input_size: int, num_components: int, output_component_size: int = 32, ...)


    Wide-Norm-Specific Args:
        input_size (int): Dimension of input latent vectors.
        num_components (int): Number of output components (also controlling hidden layer dimensions).
        output_component_size (int): In Wide Norm, each component is computed from a vector. This specifies the size of
                                     that vector. Using a large value induces **a lot of** training parameters.
                                     Default ``32``.
        symmetric (bool): Whether to enforce symmetry (i.e., metric).
                          Default: ``False``.

    Common Args (Exist for all quasimetrics, **Keyword-only**, Default values may be different for different quasimetrics):
        transforms (Collection[str]): A sequence of transforms to apply to the components, before reducing them to form
                                      the final latent quasimetric.
                                      Supported choices:
                                        + "concave_activation": Concave activation transform from Neural Norms paper.
                                      Default: ``("concave_activation",)``.
        reduction (str): Reduction method to aggregate components into final quasimetric value.
                         Supported choices:
                           + "sum": Sum of components.
                           + "max": Max of components.
                           + "mean": Average of components.
                           + "maxmean": Convex combination of max and mean. Used in original Deep Norm, Wide Norm, and IQE.
                           + "deep_linear_net_weighted_sum": Weighted sum with weights given by a deep linear net. Used in
                                                             original PQE, whose components have limited range [0, 1).
                         Default: ``"maxmean"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``, but recommended for PQEs (following original paper).
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  DeepNorms always obey quasimetric constraints if final activation
                                        applied is not "maxrelu".
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        symmetric (bool)
        num_components (int): Number of components to be combined to form the latent quasimetric. For MRN, this is always ``2``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> wn = torchqmet.WideNorm(128, num_components=64, output_component_size=32)
        >>> print(wn)
        WideNorm(
          guaranteed_quasimetric=True
          input_size=128, num_components=64, discount=None
          symmetric=False
          (transforms): Sequential(
            (0): ConcaveActivation(
              input_num_components=64, output_num_components=64
              num_units_per_input=5
            )
          )
          (reduction): MaxMean(input_num_components=64)
          (w): ConstrainedLinear(in_features=256, out_features=2048, bias=False, weight_constraint_=<built-in method relu_ of type object at 0x7fefde166060>)
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(wn(x, y))
        tensor([11.9579, 13.7906, 12.4837, 13.5018, 12.7114], grad_fn=<LerpBackward1>)
        >>> print(wn(y, x))
        tensor([11.9761, 13.7890, 12.4958, 13.4873, 12.5777], grad_fn=<LerpBackward1>)
        >>> print(wn(x[:, None], x))  # pdist
        tensor([[ 0.0000, 12.1332, 12.4422, 11.8869, 12.0364],
                [12.0220,  0.0000, 12.8111, 14.2183, 12.8810],
                [12.4647, 12.7598,  0.0000, 14.6667, 12.7702],
                [11.8936, 14.2685, 14.6700,  0.0000, 12.3801],
                [12.1089, 12.9846, 12.8418, 12.3906,  0.0000]], grad_fn=<LerpBackward1>)
    '''

    symmetric: bool

    def __init__(self, input_size: int, num_components: int, output_component_size: int = 32, symmetric: bool = False, *,
                 transforms: Collection[str] = ('concave_activation',), reduction: str = 'maxmean',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        super().__init__(input_size, num_components,
                         transforms=transforms, reduction=reduction, discount=discount,
                         guaranteed_quasimetric=True, warn_if_not_quasimetric=warn_if_not_quasimetric)
        self.symmetric = symmetric
        self.output_component_size = output_component_size
        if not symmetric:
            input_size *= 2
            weight_constraint_ = torch.relu_
        else:
            weight_constraint_ = None
        # one linear for efficiency :), fix init later
        self.w = ConstrainedLinear(input_size, num_components * output_component_size, bias=False,
                                   weight_constraint_=weight_constraint_)
        for ii in range(num_components):
            winit =  ConstrainedLinear(input_size, output_component_size, bias=False,
                                        weight_constraint_=weight_constraint_).weight.detach()
            with torch.no_grad():
                self.w.weight.reshape( num_components, output_component_size, input_size)[ii].copy_(winit)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            x = torch.cat([x, -x], dim=-1).relu()
        x = self.w(x)
        x = x.unflatten(-1, (self.num_components, self.output_component_size))
        return x.norm(dim=-1)

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x - y)

    def extra_repr(self) -> str:
        return super().extra_repr() + f'\nsymmetric={self.symmetric}'
