import abc
import math

import torch
import torch.nn as nn

from . import cdf_ops

from .measures import MeasureBase, LebesgueMeasure, GaussianBasedMeasure


class ShapeBase(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def expected_quasipartiton(self: 'ShapeBase', u: torch.Tensor, v: torch.Tensor, *, measure: MeasureBase):
        r"""
        Computes the expected quasipartition as defined in Equation (13):

        For pi, a random quasipartition given by the Poisson processes defined with `measure` and shape parametrization
        `self`,

        E[ pi(u, v) ] = 1 - \prod_j Pr[ Count( Shape(u) ) <= Count( Shape(v) ) ]
                      = 1 - \prod_j Pr[ Poisson( Measure(Shape(u) \ Shape(v)) ) <= Poisson( Measure(Shape(v) \ Shape(u)) ) ],

        where `Count(*)` is the Poisson process count.
        """
        pass


class HalfLineShape(ShapeBase):
    def expected_quasipartiton(self, u: torch.Tensor, v: torch.Tensor, *, measure: MeasureBase):
        # Shapes are (-infty, u) and (-\infty, v)
        if isinstance(measure, LebesgueMeasure):
            # PQE-LH, Eqn. (9)
            #
            #    1 - \prod_j exp( -max(u_j - v_j, 0) )
            #  = 1 - exp( \sum_j min(v_j - u_j, 0) )
            #
            # Use mean instead of sum as a normalization for better stability at initialization (Sec. C.4.1).
            return -torch.expm1((v - u).clamp(max=0).mean(-1))
        elif isinstance(measure, GaussianBasedMeasure):
            measure_exceed = measure.scale((cdf_ops.ndtr(v / measure.sigma) - cdf_ops.ndtr(u / measure.sigma)).clamp(max=0))
            return -torch.expm1(measure_exceed.mean(-1))
        else:
            raise NotImplementedError(f"measure={measure} is not supported")


class GaussianShape(ShapeBase):
    sigma2: float = 1.
    sigma: float = 1.
    log_2pi: float = math.log(2 * math.pi)

    def expected_quasipartiton(self, u: torch.Tensor, v: torch.Tensor, *, measure: MeasureBase):
        # Shapes are areas under the unit Gaussian CDF centered at u or v
        if isinstance(measure, LebesgueMeasure):
            raise RuntimeError('Gaussian shape under lebesgue measure is always symmetrical')
        elif isinstance(measure, GaussianBasedMeasure):
            # See Sec. C.2 for details.

            # To obtain the rate of the Gaussian shape (centered at mu) over an interval,
            # we can view it as integrating density product of two independent Gaussian along a
            # line of the form Y = X + a.
            #
            # Following many algebraic manipulations, one can obtain, for the 1D case,
            #
            # 1 / \sqrt{2pi (shape_sig2 + measure_sig2)} * exp( - mu^2 / (2 (shape_sig2 + measure_sig2)) )
            #   * \Int GaussianDensity(mean=mu (measure_sig2 / (shape_sig2 + measure_sig2)),
            #                          sig2=shape_sig2 * measure_sig2 / (shape_sig2 + measure_sig2))
            #
            # which can be simply viewed as scaled density of another "induced" Gaussian.
            #
            # Total rate is thus
            #     lambda / \sqrt{2pi (shape_sig2 + measure_sig2)} * exp( - mu^2 / (2 (shape_sig2 + measure_sig2)) )

            measure_sigma2 = measure.sigma2
            shape_sigma2 = self.sigma2
            sum_sigma2 = shape_sigma2 + measure_sigma2
            log_sum_sigma2 = torch.log(sum_sigma2)

            log_total_base = - (self.log_2pi + log_sum_sigma2) / 2
            log_total_u = log_total_base - 0.5 * (u ** 2) / sum_sigma2
            log_total_v = log_total_base - 0.5 * (v ** 2) / sum_sigma2
            total_u = log_total_u.exp()
            total_v = log_total_v.exp()

            mid = (u + v) / 2

            # `mid` in the new mu Gaussian (the "induced" Gaussian in formula above)
            # after normalization would be
            #       mid - mu * measure_sig2 / (measure_sig2 + shape_sig2)
            #   --------------------------------------------------------
            #   \sqrt{measure_sig2 shape_sig2 / (measure_sig2 + shape_sig2)}

            new_mu_mult = measure_sigma2 / sum_sigma2
            new_sig2 = shape_sigma2 * new_mu_mult
            new_sig = torch.sqrt(new_sig2)
            u2mid_frac = cdf_ops.ndtr( (mid - u * new_mu_mult) / new_sig )  # noqa: E201, E202
            v2mid_frac = cdf_ops.ndtr( (mid - v * new_mu_mult) / new_sig )  # noqa: E201, E202

            intersection = torch.where(
                u < v,
                v2mid_frac * total_v + (1 - u2mid_frac) * total_u,
                u2mid_frac * total_u + (1 - v2mid_frac) * total_v,
            )

            u_only = total_u - intersection
            v_only = total_v - intersection

            # Mathematically, `intersection` is smaller than both. But numerically it sometimes is computed to be
            # slightly larger. So we fix here (w/o changing the computation graph for autograd).
            u_only.data.clamp_(min=0)
            v_only.data.clamp_(min=0)

            u_only, v_only = measure.scale_multiple(u_only, v_only)

            ple: torch.Tensor = cdf_ops.prob_two_poisson_le(u_only / u.shape[-1], v_only / v.shape[-1])
            return 1 - ple.prod(dim=-1)
        else:
            raise NotImplementedError(f"measure={measure} is not supported")
