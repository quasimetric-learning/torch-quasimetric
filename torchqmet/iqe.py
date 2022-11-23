r'''
Inteval Quasimetric Embedding (IQE)
https://arxiv.org/abs/2211.15120
'''

from typing import *

import torch

from . import QuasimetricBase


@torch.jit.script
def iqe(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]  # D: dim_per_component

    # ignore pairs that x >= y
    valid = x < y

    # sort to better count
    xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
    sxy, ixy = xy.sort(dim=-1)

    # f(c) = indic( c > 0 )
    # at each location `x` along the real line, get `c` the number of intervals covering `x`, and apply `f`:
    #     \int f(c(x)) dx

    # neg_inc_copies: the **negated** increment of **input** of f at sorted locations, in terms of **#copies of delta**
    neg_inc_copies = torch.gather(valid, dim=-1, index=ixy % D) * torch.where(ixy < D, -1, 1)

    # neg_incf: the **negated** increment of **output** of f at sorted locations
    neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)

    # delta = inf
    # f input: 0 -> 0, x -> -inf.
    # neg_inp = torch.where(neg_inp_copies == 0, 0., -delta)
    # f output: 0 -> 0, x -> 1.
    neg_f = (neg_inp_copies < 0) * (-1.)
    neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)

    # reduction
    return (sxy * neg_incf).sum(-1)


class IQE(QuasimetricBase):
    r'''
    Inteval Quasimetric Embedding (IQE):
    https://arxiv.org/abs/2211.15120

    One-line Usage:

        IQE(input_size: int, dim_per_component: int = 16, ...)


    Default arguments implements IQE-maxmean. Set `reduction="sum"` to create IQE-sum.

    IQE-Specific Args:
        input_size (int): Dimension of input latent vectors
        dim_per_component (int): IQE splits latent vectors into chunks, where ach chunk computes gives an IQE component.
                                 This is the number of latent dimensions assigned to each chunk. This number must
                                 perfectly divide ``input_size``. IQE paper recomments at least ``8``.
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
                         Default: ``"maxmean"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``.
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  IQEs always obey quasimetric constraints.
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        num_components (int): Number of components to be combined to form the latent quasimetric. For IQEs, this is
                              ``input_size // dim_per_component``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> iqe = IQE(128, dim_per_component=16)
        >>> print(iqe)
        IQE(
          guaranteed_quasimetric=True
          input_size=128, num_components=8, discount=None
          (transforms): Sequential()
          (reduction): MaxMean(input_num_components=8)
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(iqe(x, y))
        tensor([3.3045, 3.8072, 3.9671, 3.3521, 3.7831],, grad_fn=<LerpBackward1>)
        >>> print(iqe(y, x))
        tensor([3.3850, 3.8457, 4.0870, 3.1757, 3.9459], grad_fn=<LerpBackward1>)
        >>> print(iqe(x[:, None], x))  # pdist
        tensor([[0.0000, 3.8321, 3.7907, 3.5915, 3.3326],
                [3.9845, 0.0000, 4.0173, 3.8059, 3.7177],
                [3.7934, 4.3673, 0.0000, 4.0536, 3.6068],
                [3.1764, 3.4881, 3.5300, 0.0000, 2.9292],
                [3.7184, 3.8690, 3.8321, 3.5905, 0.0000]], grad_fn=<ReshapeAliasBackward0>)
    '''

    def __init__(self, input_size: int, dim_per_component: int = 16, *,
                 transforms: Collection[str] = (), reduction: str = 'maxmean',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        assert dim_per_component > 0, "dim_per_component must be positive"
        assert input_size % dim_per_component == 0, \
            f"input_size={input_size} is not divisible by dim_per_component={dim_per_component}"
        num_components = input_size // dim_per_component
        super().__init__(input_size, num_components, guaranteed_quasimetric=True, warn_if_not_quasimetric=warn_if_not_quasimetric,
                         transforms=transforms, reduction=reduction, discount=discount)
        self.latent_2d_shape = torch.Size([num_components, dim_per_component])

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return iqe(
            x=x.unflatten(-1, self.latent_2d_shape),
            y=y.unflatten(-1, self.latent_2d_shape),
        )
