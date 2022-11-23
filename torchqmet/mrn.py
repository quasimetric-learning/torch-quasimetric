r"""
Metric Residual Network (MRN)
https://arxiv.org/abs/2208.08133
"""

from typing import *

import warnings

import torch
import torch.nn as nn

from . import QuasimetricBase


class MRNProjector(nn.Sequential):
    output_size: int

    def __init__(self, input_size: int, *, output_size: int = 16, hidden_sizes: List[int] = [176]):
        modules = []
        for hidden_size in hidden_sizes:
            modules.append(nn.Linear(input_size, hidden_size))
            modules.append(nn.ReLU())
            input_size = hidden_size
        modules.append(nn.Linear(input_size, output_size))
        super().__init__(*modules)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return super().__call__(z)


class MRN(QuasimetricBase):
    r"""
    Metric Residual Network (MRN):
    https://arxiv.org/abs/2208.08133

    One-line Usage:

        MRN(input_size: int, sym_p: float = 2, ...)

    Default arguments implement the MRN as described in the original MRN paper:

        d_z(x, y) = ( 1/d_sym * \sum_i (f_sym(x)[i] - f_sym(y))^2 )^(p/2) + \max_j ReLU( f_asym(x)[j] - f_asym(y)[j] ),

    where `f_sym` and `f_asym` are 2-layer MLPs, and `d_sym` is the output size of `f_sym`.

     + The first term is simply a (scaled) Euclidean distance raised to the `p`-th power, representing the symmetrical port.
     + The second term is the asymmetrical part.

    These two terms are used as two **components** of the quasimetric. With default arguments, a summation reduction
    combines them.

    NOTE::
        Default arguments does not guarantee a true quasimetric, since one of the component is the **squared** Euclidean
        distance, rather than regular Euclidean distance.

        Following a fix proposed in the IQE paper (https://arxiv.org/abs/2211.15120), we allow setting
        `sym_p=1`, which uses the regular Euclidean distance instead, and guarantees a quasimetric.

        Alternatively, simply use subclass :class:`MRNFixed`, which changes the default of `sym_p` to `1`.

    MRN-Specific Args:
        input_size (int): Dimension of input latent vectors.
        sym_p (float): Exponent applied to the symmetrical term of Euclidean distance.
                       Default: ``2``.
        proj_hidden_size (int): Hidden size of `f_sym` and `f_asym` MLPs.
                                Default: ``176``.
        proj_output_size (int): Output size of `f_sym` and `f_asym` MLPs.
                                Default: ``16``.

    Common Args (Exist for all quasimetrics, **Keyword-only**, Default values may be different for different quasimetrics):
        transforms (Collection[str]): A sequence of transforms to apply to the components, before reducing them to form
                                      the final latent quasimetric.
                                      Supported choices:
                                        + "concave_activation": Concave activation transform from Neural Norms paper.
                                      Default: ``()`` (no transforms).
        reduction (str): Reduction method to aggregate components into final quasimetric value.
                         Supported choices:
                           + "sum": Sum of components.
                           + "max": Max of components.
                           + "mean": Average of components.
                           + "maxmean": Convex combination of max and mean. Used in original Deep Norm, Wide Norm, and IQE.
                           + "deep_linear_net_weighted_sum": Weighted sum with weights given by a deep linear net. Used in
                                                             original PQE, whose components have limited range [0, 1).
                         Default: ``"sum"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``, but recommended for PQEs (following original paper).
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  MRNs always obey quasimetric constraints if `0 < sym_p <= 1`.
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        sym_p (float)
        num_components (int): Number of components to be combined to form the latent quasimetric. For MRN, this is always ``2``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> mrn = MRN(128)  # default MRN
        .../torchqmet/mrn.py:61: UserWarning: MRN with `sym_p=2` may not be a quasimetric (see IQE paper Sec. C.2). Use
        `torchqmet.MRNFixed` with default `sym_p=1` to guarantee a quasimetric.
        >>> print(mrn)
        MRN(
          guaranteed_quasimetric=False
          input_size=128, num_components=2, discount=None
          sym_p=2
          (transforms): Sequential()
          (reduction): Sum(input_num_components=2)
          (sym_proj): MRNProjector(
            (0): Linear(in_features=128, out_features=176, bias=True)
            (1): ReLU()
            (2): Linear(in_features=176, out_features=16, bias=True)
          )
          (asym_proj): MRNProjector(
            (0): Linear(in_features=128, out_features=176, bias=True)
            (1): ReLU()
            (2): Linear(in_features=176, out_features=16, bias=True)
          )
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(mrn(x, y))
        tensor([0.3584, 0.8246, 0.4646, 0.5300, 0.5409], grad_fn=<SumBackward1>)
        >>> print(mrn(y, x))
        tensor([0.5899, 0.5375, 0.7205, 0.4931, 0.5727], grad_fn=<SumBackward1>)
        >>> print(mrn(x[:, None], x))  # pdist
        tensor([[0.0000, 0.3609, 0.5478, 0.6326, 0.4724],
                [0.5219, 0.0000, 0.5700, 0.7597, 0.5657],
                [0.4636, 0.5970, 0.0000, 0.4545, 0.5955],
                [0.8028, 0.8550, 1.1630, 0.0000, 0.7704],
                [0.6520, 0.5160, 0.8666, 0.4677, 0.0000]], grad_fn=<SumBackward1>)
        >>>
        >>> # MRN with fix to guarantee quasimetric constraints
        >>> mrn = MRNFixed(128)  # or use MRN(..., sym_p=1)
        >>> print(mrn)
        MRNFixed(
          guaranteed_quasimetric=True
          input_size=128, num_components=2, discount=None
          sym_p=1
          (transforms): Sequential()
          (reduction): Sum(input_num_components=2)
          (sym_proj): MRNProjector(
            (0): Linear(in_features=128, out_features=176, bias=True)
            (1): ReLU()
            (2): Linear(in_features=176, out_features=16, bias=True)
          )
          (asym_proj): MRNProjector(
            (0): Linear(in_features=128, out_features=176, bias=True)
            (1): ReLU()
            (2): Linear(in_features=176, out_features=16, bias=True)
          )
        )
        >>> print(mrn(x[:, None], x))  # pdist
        tensor([[0.0000, 0.7640, 0.7091, 0.5985, 0.7392],
                [0.7220, 0.0000, 0.8448, 0.9160, 0.8006],
                [0.8715, 0.7199, 0.0000, 0.9072, 0.8582],
                [0.7666, 0.8370, 0.7094, 0.0000, 0.9459],
                [0.7773, 0.6895, 0.7869, 0.8662, 0.0000]], grad_fn=<SumBackward1>)
    """

    sym_p: float

    def __init__(self, input_size: int, sym_p: float = 2, proj_hidden_size: int = 176, proj_output_size: int = 16, *,
                 transforms: Collection[str] = (), reduction: str = 'sum',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        if sym_p > 1:
            guaranteed_quasimetric = False
            if warn_if_not_quasimetric:
                warnings.warn(
                    f'MRN with `sym_p={sym_p:g}` may not be a quasimetric (see IQE paper Sec. C.2). '
                    'Use `torchqmet.MRNFixed` with default `sym_p=1` to guarantee a quasimetric.')
        elif sym_p <= 0:
            raise ValueError(f"Expect positive `sym_p`, but `sym_p={sym_p:g}`")
        else:
            guaranteed_quasimetric = True
        super().__init__(input_size, num_components=2, guaranteed_quasimetric=guaranteed_quasimetric, warn_if_not_quasimetric=warn_if_not_quasimetric,
                         transforms=transforms, reduction=reduction, discount=discount)
        self.sym_p = sym_p
        self.sym_proj = MRNProjector(input_size, output_size=proj_output_size, hidden_sizes=[proj_hidden_size])
        self.asym_proj = MRNProjector(input_size, output_size=proj_output_size, hidden_sizes=[proj_hidden_size])

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.stack(torch.broadcast_tensors(x, y), dim=0)
        sym_projx, sym_projy = self.sym_proj(xy).unbind(0)
        sym_dist = (sym_projx - sym_projy).square().mean(dim=-1).pow(self.sym_p / 2)
        asym_projx, asym_projy = self.asym_proj(xy).unbind(0)
        asym_dist = (asym_projx - asym_projy).max(dim=-1).values.relu()
        return torch.stack([sym_dist, asym_dist], dim=-1)

    def extra_repr(self) -> str:
        return super().extra_repr() + f'\nsym_p={self.sym_p:g}'


class MRNFixed(MRN):
    r"""
    Metric Residual Network (MRN):
    https://arxiv.org/abs/2208.08133
    with fix proposed by the IQE paper (Sec. C.2):
    https://arxiv.org/abs/2211.15120

    One-line Usage:

        MRNFixed(input_size, sym_p=1, ...)

    Defaults to `sym_p=1`. This guarantees a quasimetric, unlike the original official MRN (where `sym_p=2`).

    See :class:`MRN` for details of other arguments.
    """

    def __init__(self, input_size: int, sym_p: float = 1, proj_hidden_size: int = 176, proj_output_size: int = 16, *,
                 transforms: Collection[str] = (), reduction: str = 'sum',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        super().__init__(input_size, sym_p, proj_hidden_size, proj_output_size, warn_if_not_quasimetric=warn_if_not_quasimetric,
                         transforms=transforms, reduction=reduction, discount=discount)
