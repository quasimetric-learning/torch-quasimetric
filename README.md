# `torchqmet`: PyTorch Package for Quasimetric Learning

**[Tongzhou Wang](https://www.tongzhouwang.info)**

This repository provides a PyTorch package for quasimetric learning --- Learning a **quasimetric** function from data.


It implements many recent quasimetric learning methods (in reverse chronological order):
+ [1] Interval Quasimetric Embeddings (IQEs) ([paper](https://arxiv.org/abs/2211.15120)) ([website](https://www.tongzhouwang.info/interval_quasimetric_embedding/)) <br/>
  Wang & Isola. NeurIPS 2022 NeurReps Workshop Proceedings Track.
+ [2] Metric Residual Networks (MRNs) ([paper](https://arxiv.org/abs/2208.08133)) <br/>
  Liu et al. arXiv 2022.
+ [3] Poisson Quasimetric Embeddings (PQEs) ([paper](https://arxiv.org/abs/2206.15478)) ([website](https://github.com/ssnl/poisson_quasimetric_embedding)) <br/>
  Wang and Isola. ICLR 2022. https://arxiv.org/abs/2206.15478
+ [4] Deep Norm and Wide Norm ([paper](https://arxiv.org/abs/2002.05825)) <br/>
  Pitis and Chan et al. ICLR 2020.

This is also the official implementation for IQE [1].

## Quasimetric Learning

In the core of quasimetric learning is a parametrized quasimetric function $d$. For data $x$ and $y$,
$$d(x, y) := \textsf{quasimetric distance from $x$ to $y$}.$$

Similar to metric learning, the best performing quasimetric learning methods use
1. a generic encoder function $f(x)$ mapping data $x$ to a generic latent space $f(x) \in \mathbb{R}^d$
2. a latent *quasimetric* function $d_z$ that defines a quasimetric on the $\mathbb{R}^d$ latent space.
$$d(x, y) := d_z(f(x), f(y))$$

Compared to other alternatives, such **latent quasimetric formulations** inherently have the correct geometric constraints and inductive bias (see [1,2,3,4] for comparisons). This package provides well-tested implementations of many recently-proposed choices of $d_z$.

While the encoder $f$ is usually taken to be a generic deep neural network, the choices of $d_z$ can significantly affect performance, as they may differ in expressivity (i.e., whether they can approximate a diverse family of quasimetrics), in easiness in gradient optimization, in overhead (e.g., number of parameters), etc.

In particular, the IQE paper [4] analyzes many desired properties for $d_z$ and propose IQE, which greatly improves over previous methods with a simple $d_z$ form. If you are not sure which one to use, we recommend first trying IQEs.

## Requirements:

+ Python >= 3.7
+ PyTorch >= 1.11.0


## Install

```py
python setup.py install
```

Alternatively, add the `torchqmet` folder to your project.

## Getting Started (Quickly)

Import package
```py
import torchqmet
```

Example with IQE [1]
```py
device = torch.device('cuda:0')  # works on both CPU and CUDA!

# Create IQE
d_iqe = torchqmet.IQE(
    input_size=128,            # 128-d latent vectors
    dim_per_component=16,      # split 128 dimensions into 16-dimenional chunks, where each chunk
                               #    gives an IQE component (IQE paper recommends `dim_per_component >= 8`)
).to(device)

# latents, usually from an encoder. use random vectors as example
x = torch.randn(2, 128, device=device)
y = torch.randn(2, 128, device=device)

# distance
print(d_iqe(x, y))
# tensor([22.5079, 29.4083], device='cuda:0')

# cdist
print(d_iqe(x[:, None], y))
# tensor([[22.5079, 25.2227],
#         [28.1123, 29.4083]], device='cuda:0')

# pdist
print(d_iqe(x[:, None], x))
# tensor([[ 0.0000, 22.9859],
#         [29.4122,  0.0000]], device='cuda:0')
```

Other latent quasimetrics $d_z$ can be similarly created (with different arguments) with
```py
torchqmet.PQE  # general PQE [3]
torchqmet.PQELH, qmet.PQEGG  # two specific PQE choices from the original paper [3]
torchqmet.MRN, torchqmet.MRNFixed  # MRN [2]
torchqmet.DeepNorm  # Deep Norm [4]
torchqmet.WideNorm  # Wide Norm [4]
```

See their docstrings (next section or via `help(torchqmet.XXX)`) for details!

## Documentation (Click to expand)

<details>
<summary>
Inteval Quasimetric Embedding (IQE) [1]<br/>
<code>torchqmet.IQE(input_size: int, dim_per_component: int = 16, ...)</code>
</summary>

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
</details>


<details>
<summary>
Metric Residual Network (MRN) [2]<br/>
<code>torchqmet.MRN(input_size: int, sym_p: float = 2, ...)</code>
</summary>
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
</details>


<details>
<summary>
Metric Residual Network (MRN) fixed to be guaranteed a quasimetric [1,2]<br/>
<code>torchqmet.MRNFixed(input_size: int, sym_p: float = 1, ...)</code>
</summary>

    Metric Residual Network (MRN):
    https://arxiv.org/abs/2208.08133
    with fix proposed by the IQE paper (Sec. C.2):
    https://arxiv.org/abs/2211.15120

    One-line Usage:

        MRNFixed(input_size, sym_p=1, ...)

    Defaults to `sym_p=1`. This guarantees a quasimetric, unlike the original official MRN (where `sym_p=2`).

    See :class:`MRN` for details of other arguments.
</details>

<details>
<summary>
Poisson Quasimetric Embedding (PQE) [3]<br/>
<code>torchqmet.PQE(input_size, dim_per_component=16, measure="lebesgue", shape="halfline", ...)</code>
</summary>
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
</details>


<details>
<summary>
Poisson Quasimetric Embedding with Lebesgue measure and Half-line shape (PQE-LH) [3]<br/>
<code>torchqmet.PQELH(input_size, dim_per_component=16, ...)</code>
</summary>
    PQE-LH variant of Poisson Quasimetric Embedding (PQE), using Lebesgue measure + Half-line shape:
    https://arxiv.org/abs/2206.15478

    One-line Usage:

        PQELH(input_size, dim_per_component=16, ...)

    Unlike :class:`PQE`, arguments `measure="lebesgue"` and `shape="halfline"` are fixed and not configurable.

    See :class:`PQE` for details of other arguments.
</details>


<details>
<summary>
Poisson Quasimetric Embedding with Gaussian-based measure and Gaussian-based shape (PQE-GG) [3]<br/>
<code>torchqmet.PQEGG(input_size, dim_per_component=16, ...)</code>
</summary>
    PQE-GG variant of Poisson Quasimetric Embedding (PQE), using Gaussian-based measure and Gaussian-based shape:
    https://arxiv.org/abs/2206.15478

    One-line Usage:

        PQEGG(input_size, dim_per_component=16, ...)

    Unlike :class:`PQE`, arguments `measure="gaussian"` and `shape="gaussian"` are fixed and not configurable.

    See :class:`PQE` for details of other arguments.
</details>


<details>
<summary>
Deep Norm [4]<br/>
<code>torchqmet.DeepNorm(input_size: int, num_components: int, num_layers: int = 3, ...)</code>
</summary>

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

</details>



<details>
<summary>
Wide Norm [4]<br/>
<code>torchqmet.WideNorm(input_size: int, num_components: int, output_component_size: int = 32, ...)</code>
</summary>
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
</details>

## Citation

This package:
```bibtex
@misc{wang2022torchquasimetric,
  author = {Tongzhou Wang},
  title = {torchqmet: {P}y{T}orch Package for Quasimetric Learning},
  year = {2022},
  howpublished = {\url{https://github.com/quasimetric-learning/torch-quasimetric}},
}
```

IQE [1]:
```bibtex
@inproceedings{wang2022iqe,
  title={Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings},
  author={Wang, Tongzhou and Isola, Phillip},
  note={Workshop on Symmetry and Geometry in Neural Representations at Conference on Neural Information Processing Systems (NeurIPS) 2022},
  booktitle={Proceedings of Machine Learning Research (PMLR)},
  volume={Volume on Symmetry and Geometry in Neural Representations},
  year={2022},
}
```

## Questions

For questions about the code provided in this repository, please open an GitHub issue.

For questions about the IQE [1] or PQE [3] paper, please contact Tongzhou Wang (`tongzhou _AT_ mit _DOT_ edu`).
