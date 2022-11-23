# `cdf_ops` Extension

This extension implements a couple functions for computing Bessel function, Poisson race probabilities, and Gaussian/non-central-chi-square CDFs.

## Documentations

The provided functions are listed below. They work on both CPU and CUDA PyTorch Tensors.

At first use of a function (if it is not found in the PyTorch installation), a compilation of the extension will trigger, which may take up to 10 minutes. Subsequent uses will use cached compilation results, as long as it is on the same GPU compute capabilities, CUDA and PyTorch versions.

```py
def chndtr(x: torch.Tensor, df: Union[torch.Tensor, float], nc: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the non-central Chi-square CDF.

    For a distribution with :attr:`df` degrees of freedom and :attr:`nc` non-centrality parameter,
    this evaluates the CDF at :attr:`x`.
    """
    ...


def i0(input: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

    .. math::
        \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
    """
    ...


def i0e(input: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
    for each element of :attr:`input`.

    .. math::
        \text{out}_{i} = \exp(-|x|) * i0(x) = \exp(-|x|) * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
    """
    ...


def i1(input: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the first order modified Bessel function of the first kind (as defined below)
    for each element of :attr:`input`.

    .. math::
        \text{out}_{i} = \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
    """
    ...


def i1e(input: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
    for each element of :attr:`input`.

    .. math::
        \text{out}_{i} = \exp(-|x|) * i1(x) =
            \exp(-|x|) * \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
    """
    ...


def prob_two_poisson_gt(mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the elementwise ``Prob[ Poisson(mu1) > Poisson(mu2) ]``.
    """
    ...


def prob_two_poisson_le(mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the elementwise ``Prob[ Poisson(mu1) <= Poisson(mu2) ]``.
    """
    ...


def ndtr(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the standard Gaussian CDF evaluated at :attr:`x`.
    """
    ...


def log_ndtr(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the log of the standard Gaussian CDF evaluated at :attr:`x`.

    This is numerically more stable than calling ``ndtr(x).log()``, in both forward and backward.
    """
    ...


def prod_ndtr(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    r"""
    Computes ``ndtr(x).prod(dim=dim)``.

    This is numerically more stable than calling ``ndtr(x).prod(dim=dim)``, in both forward and backward.
    """
    ...
```

## FAQ

**Q:** How to compile so that the compiled extension can be used for machines with GPUs of different compute capabilities (e.g., on a cluster with many types of GPUs)?

**A:** Specify a environment flag like `TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.5+PTX'`.

## License

Part of the code is modified from [`cephes`](https://www.netlib.org/cephes/) and [`CDFLIB`](https://people.sc.fsu.edu/~jburkardt/cpp_src/cdflib/cdflib.html).

`cephes` is available from [`scipy`](https://github.com/scipy/scipy) under 3-clause BSD. All derived code from `cephes` are located under [`./cpu/cephes`](./cpu/cephes) and  [`./cuda/cephes`](./cuda/cephes).

For `CDFLIB`, while it is website release its under LGPL. We derive the code from [`scipy`](https://github.com/scipy/scipy), which is under 3-clause BSD. All derived code from `CDFLIB` are located under [`./cpu/cdflib`](./cpu/cdflib) and  [`./cuda/cdflib`](./cuda/cdflib).
