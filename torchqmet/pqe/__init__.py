r'''
Poisson Quasimetric Embedding (PQE)
https://arxiv.org/abs/2206.15478
'''

from typing import *

import torch

from .measures import MeasureBase, LebesgueMeasure, GaussianBasedMeasure
from .shapes import ShapeBase, HalfLineShape, GaussianShape
from .. import QuasimetricBase


class PQE(QuasimetricBase):
    r'''
    Poisson Quasimetric Embedding (PQE):
    https://arxiv.org/abs/2206.15478

    One-line Usage:

        PQE(input_size, dim_per_component=16, measure="lebesgue", shape="halfline", ...)


    PQE requires a specification of "shape" and "measure" for defining the Poisson process counts. We support
      + Measure: Lebesgue measure, a Gaussian-based measure.
      + Shape: Half-line, a Gaussian shape.
    These choices are sufficient to implement PQE-LH (Lebesgue + Half-line) and PQE-GG (Gaussian-based measure + Gaussian shape),
    the two PQE variants used in the original PQE paper.

    Default arguments implements PQE-LH, which has a simple form and generally works well according the PQE paper.
    To use PQE-GG, PQE paper's other proposed variant, set `shape="gaussian", measure="gaussian"`, or simply use subclass
    :class:`PQEGG`. Similarly, subclass :class:`PQELH` is gauranteed to PQE-LH.

    PQE-Specific Args:
        input_size (int): Dimension of input latent vectors
        dim_per_component (int): IQE splits latent vectors into chunks, where ach chunk computes gives an IQE component.
                                 This is the number of latent dimensions assigned to each chunk. This number must
                                 perfectly divide ``input_size``.
                                 Default: ``4``.
        measure (str):  Measure used in the Poisson processes. Choices are ``"lebesgue"`` and ``"guassian"``.
                        Default: ``"lebesgue"``.
        shape (str):  Shape parametrizations used in the Poisson processes. Choices are ``"halfline"`` and ``"guassian"``.
                      ``"guassian"`` can only be used with ``"guassian"`` measure.
                      Default: ``"halfline"``.

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
                         Default: ``"deep_linear_net_weighted_sum"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``, but recommended for PQEs (following original paper).
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  PQEs always obey quasimetric constraints.
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        num_components (int): Number of components to be combined to form the latent quasimetric. For PQEs, this is
                              ``input_size // dim_per_component``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        measure (MeasureBase): Poisson process measure used.
        shape (ShapeBase): Poisson process shape parametrization used.
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> pqe = PQE(128, dim_per_component=16)  # default is PQE-LH, see `measure` and `shape` below
        >>> print(pqe)
        PQE(
          guaranteed_quasimetric=True
          input_size=128, num_components=8, discount=None
          (transforms): Sequential()
          (reduction): DeepLinearNetWeightedSum(
            input_num_components=8
            (alpha_net): DeepLinearNet(
              bias=True, non_negative=True
              (mats): ParameterList(
                  (0): Parameter containing: [torch.float32 of size 1x64]
                  (1): Parameter containing: [torch.float32 of size 64x64]
                  (2): Parameter containing: [torch.float32 of size 64x64]
                  (3): Parameter containing: [torch.float32 of size 64x8]
              )
            )
          )
          (measure): LebesgueMeasure()
          (shape): HalfLineShape()
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(pqe(x, y))
        tensor([0.5994, 0.7079, 0.6474, 0.7858, 0.6954], grad_fn=<SqueezeBackward1>)
        >>> print(pqe(y, x))
        tensor([0.5731, 0.7868, 0.9577, 0.5707, 0.7005], grad_fn=<SqueezeBackward1>)
        >>> print(pqe(x[:, None], x))  # pdist
        tensor([[0.0000, 0.8147, 0.9515, 0.6505, 0.8131],
                [0.6491, 0.0000, 0.8892, 0.4910, 0.7271],
                [0.5663, 0.6442, 0.0000, 0.4402, 0.6461],
                [0.6756, 0.7252, 0.9157, 0.0000, 0.7032],
                [0.6689, 0.7006, 0.8784, 0.4509, 0.0000]], grad_fn=<SqueezeBackward1>)
        >>>
        >>> # PQE-GG, modeling discounted distances
        >>> pqe = PQEGG(128, dim_per_component=16, discount=0.9)  # or use PQE(..., shape="guassian", measure="gaussian")
        >>> # PQE-GG requires the `cdf_ops` extension.  First usage of PQE-GG will trigger compile.
        >>> # See `PQE` docstring for details.
        >>> print(pqe(x, y))  # discounted distance
        tensor([0.9429, 0.9435, 0.9402, 0.9404, 0.9428], grad_fn=<ProdBackward1>)
        >>> print(pqe(x[:, None], x))  # discounted pdist
        tensor([[1.0000, 0.9423, 0.9313, 0.9473, 0.9470],
                [0.9452, 1.0000, 0.9400, 0.9520, 0.9517],
                [0.9395, 0.9456, 1.0000, 0.9489, 0.9531],
                [0.9380, 0.9397, 0.9313, 1.0000, 0.9484],
                [0.9395, 0.9412, 0.9371, 0.9502, 1.0000]], grad_fn=<ProdBackward1>)
    '''

    measure: MeasureBase
    shape: ShapeBase

    def __init__(self, input_size: int, dim_per_component: int = 4, measure: str = 'lebesgue', shape: str = 'halfline', *,
                 transforms: Collection[str] = (), reduction: str = 'deep_linear_net_weighted_sum',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        assert dim_per_component > 0, "dim_per_component must be positive"
        assert input_size % dim_per_component == 0, \
            f"input_size={input_size} is not divisible by dim_per_component={dim_per_component}"
        num_components = input_size // dim_per_component
        super().__init__(input_size, num_components, guaranteed_quasimetric=True, warn_if_not_quasimetric=warn_if_not_quasimetric,
                         transforms=transforms, reduction=reduction, discount=discount)
        # Will need to reshape the latents to be 2D so that
        #   - the last dim represents Poisson processes that parametrize a distribution of quasipartitions
        #   - the second last dim represents the number of mixtures of such quasipartition distributions
        self.latent_2d_shape = torch.Size([num_components, dim_per_component])
        if measure == 'lebesgue':
            self.measure = LebesgueMeasure(shape=self.latent_2d_shape)
        elif measure == 'gaussian':
            self.measure = GaussianBasedMeasure(shape=self.latent_2d_shape)
        else:
            raise ValueError(f'Unsupported measure={repr(measure)}')
        if shape == 'halfline':
            self.shape = HalfLineShape()
        elif shape == 'gaussian':
            self.shape = GaussianShape()
        else:
            raise ValueError(f'Unsupported shape={repr(shape)}')

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.shape.expected_quasipartiton(
            x.unflatten(-1, self.latent_2d_shape),
            y.unflatten(-1, self.latent_2d_shape),
            measure=self.measure,
        )


class PQELH(PQE):
    r"""
    PQE-LH variant of Poisson Quasimetric Embedding (PQE), using Lebesgue measure and Half-line shape:
    https://arxiv.org/abs/2206.15478

    One-line Usage:

        PQELH(input_size, dim_per_component=16, ...)

    Unlike :class:`PQE`, arguments `measure="lebesgue"` and `shape="halfline"` are fixed and not configurable.

    See :class:`PQE` for details of other arguments.
    """

    def __init__(self, input_size: int, dim_per_component: int = 4, *,
                 transforms: Collection[str] = (), reduction: str = 'deep_linear_net_weighted_sum',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        super().__init__(input_size, dim_per_component, measure='lebesgue', shape='halfline',
                         warn_if_not_quasimetric=warn_if_not_quasimetric, transforms=transforms, reduction=reduction, discount=discount)


class PQEGG(PQE):
    r"""
    PQE-GG variant of Poisson Quasimetric Embedding (PQE), using Gaussian-based measure and Gaussian-based shape:
    https://arxiv.org/abs/2206.15478

    One-line Usage:

        PQEGG(input_size, dim_per_component=16, ...)

    Unlike :class:`PQE`, arguments `measure="gaussian"` and `shape="gaussian"` are fixed and not configurable.

    See :class:`PQE` for details of other arguments.
    """

    def __init__(self, input_size: int, dim_per_component: int = 4, *,
                 transforms: Collection[str] = (), reduction: str = 'deep_linear_net_weighted_sum',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        super().__init__(input_size, dim_per_component, measure='gaussian', shape='gaussian',
                         warn_if_not_quasimetric=warn_if_not_quasimetric, transforms=transforms, reduction=reduction, discount=discount)

