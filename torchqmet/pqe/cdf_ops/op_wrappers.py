from typing import *

import torch
try:
    import torch.special as tsp
except ImportError:
    tsp = None


from .load_ext import load_extension_if_needed  # load lazily


def chndtr(x: torch.Tensor, df: Union[torch.Tensor, float], nc: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the non-central Chi-square CDF.

    For a distribution with :attr:`df` degrees of freedom and :attr:`nc` non-centrality parameter,
    this evaluates the CDF at :attr:`x`.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.chndtr(x, df, nc)


if not hasattr(torch, 'i0'):  # i0 was first added as torch.i0
    def i0(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
        """
        load_extension_if_needed()
        return torch.ops.cdf_ops.i0(input)
else:
    def i0(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
        """
        return torch.i0(input)


if not hasattr(tsp, 'i0e'):
    def i0e(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \exp(-|x|) * i0(x) = \exp(-|x|) * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
        """
        load_extension_if_needed()
        return torch.ops.cdf_ops.i0e(input)
else:
    def i0e(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \exp(-|x|) * i0(x) = \exp(-|x|) * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
        """
        return tsp.i0e(input)


if not hasattr(tsp, 'i1'):
    def i1(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the first order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
        """
        load_extension_if_needed()
        return torch.ops.cdf_ops.i1(input)
else:
    def i1(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the first order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
        """
        return tsp.i1(input)


if not hasattr(tsp, 'i1e'):
    def i1e(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \exp(-|x|) * i1(x) =
                \exp(-|x|) * \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
        """
        load_extension_if_needed()
        return torch.ops.cdf_ops.i1e(input)
else:
    def i1e(input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
        for each element of :attr:`input`.

        .. math::
            \text{out}_{i} = \exp(-|x|) * i1(x) =
                \exp(-|x|) * \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
        """
        return tsp.i1e(input)


def prob_two_poisson_gt(mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the elementwise ``Prob[ Poisson(mu1) > Poisson(mu2) ]``.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.prob_two_poisson_gt(mu1, mu2)


def prob_two_poisson_le(mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the elementwise ``Prob[ Poisson(mu1) <= Poisson(mu2) ]``.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.prob_two_poisson_le(mu1, mu2)


def ndtr(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the standard Gaussian CDF evaluated at :attr:`x`.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.ndtr(x)


def log_ndtr(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the log of the standard Gaussian CDF evaluated at :attr:`x`.

    This is numerically more stable than calling ``ndtr(x).log()``, in both forward and backward.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.log_ndtr(x)


def prod_ndtr(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    r"""
    Computes ``ndtr(x).prod(dim=dim)``.

    This is numerically more stable than calling ``ndtr(x).prod(dim=dim)``, in both forward and backward.
    """
    load_extension_if_needed()
    return torch.ops.cdf_ops.log_ndtr(x, dim=dim)


__all__ = [
    'chndtr', 'i0', 'i0e', 'i1', 'i1e',
    'prob_two_poisson_gt', 'prob_two_poisson_le',
    'ndtr', 'log_ndtr', 'prod_ndtr',
]
