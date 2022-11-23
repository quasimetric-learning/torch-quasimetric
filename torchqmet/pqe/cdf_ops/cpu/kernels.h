#pragma once

#include "../cdf_ops.h"
#include "cdflib/cdflib.hpp"
#include "cephes/i0.h"
#include "cephes/i1.h"
#include "cephes/ndtr.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <cmath>
#include <limits>
#include <torch/extension.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Math.h>
#include <c10/util/complex.h>


namespace cdf_ops {
namespace cpu {

namespace {

// https://github.com/scipy/scipy/blob/5f4c4d802e5a56708d86909af6e5685cd95e6e66/scipy/special/cdf_wrappers.c#L65-L98
template <bool return_bound>
static inline
double cdflib_get_result(const char* name, int status, double bound, double result) {
    TORCH_CHECK(status >= 0, name, ": Invalid input parameter ", -status, " is out of range");
    switch (status) {
    case 0:
      /* no error */
        return result;
    case 1:
        if (return_bound) {
            TORCH_WARN(name, ": Answer appears to be lower than lowest search bound (", bound, ")");
            return bound;
        } else {
            TORCH_CHECK(false, name, ": Answer appears to be lower than lowest search bound (", bound, ")");
        }
        break;
    case 2:
        if (return_bound) {
            TORCH_WARN(name, ": Answer appears to be higher than highest search bound (", bound, ")");
            return bound;
        } else {
            TORCH_CHECK(false, name, ": Answer appears to be higher than highest search bound (", bound, ")");
        }
        break;
    case 3:
    case 4:
        TORCH_CHECK(false, name, ": Two parameters that should sum to 1.0 do not");
        break;
    case 10:
        TORCH_CHECK(false, name, ": Computational error");
        break;
    default:
        TORCH_CHECK(false, name, ": Unknown error");
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}

template <typename scalar_t, bool use_bounds = true, bool clamp_01 = true>
static inline
scalar_t cdflib_chndtr(scalar_t _x, scalar_t _df, scalar_t _nc) {
    if (_x != _x || _df != _df || _nc != _nc) {
        // cdf_ops doesn't handle NaN well.
        return NAN;
    }

    // NOTE [ Non-central Chi-square CDF Bounds ]
    //
    // The CDFLIB algorithm is *really* slow when both `x` and `nc` are large
    // and can scale with `nc`.
    //
    // See documentation for CDFLIB CDFCHN, copied below:
    //
    //    The computation time required for this routine is proportional
    //    to the noncentrality parameter (PNONC).  Very large values of
    //    this parameter can consume immense computer resources.  This is
    //    why the search range is bounded by 1e9.
    //
    // Hence, we employ the CDF bounds from
    //   http://proceedings.mlr.press/v22/kolar12/kolar12Supple.pdf
    // Lemma8.

    if (use_bounds) {
        constexpr scalar_t neglogeps = -std::log(std::numeric_limits<scalar_t>::epsilon());
        constexpr scalar_t sqrt_neglogeps = std::sqrt(neglogeps);
        const scalar_t mean = _df + _nc;
        const scalar_t term1 = 2 * std::sqrt(_df + 2 * _nc) * sqrt_neglogeps;
        if (_x >= mean + term1 + 2 * neglogeps) {
            return 1;
        } else if (_x <= mean - term1) {
            return 0;
        }
    }

    int which = 1;
    double q = 0, p = 0, bound = 0;
    int status = 10;

    double x = (double) _x;
    double df = (double) _df;
    double nc = (double) _nc;

    cdflib::cpu::cdfchn(&which, &p, &q, &x, &df, &nc, &status, &bound);
    scalar_t cdf = cdflib_get_result<true>("cdfchn", status, bound, p);

    if (clamp_01) {
        if (cdf < 0) {
            return 0;
        } else if (cdf > 1) {
            return 1;
        }
    }
    return cdf;
}

static inline
void chndtr_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "chndtr_cpu", [&] {
            at::native::cpu_kernel(iter, &cdflib_chndtr<scalar_t>);
        }
    );
}


static inline
void chndtr_scalar_kernel_cpu(at::TensorIterator& iter, double df) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "chndtr_scalar_cpu", [&] {
            at::native::cpu_kernel(iter, [=](scalar_t x, scalar_t nc) -> scalar_t {
                return cdflib_chndtr<double>(x, df, nc);  // it is internally double anyways, so avoid precision lost on df...
            });
        }
    );
}


static inline
void i0e_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "i0e_cpu", [&] {
            at::native::cpu_kernel(iter, &cephes::cpu::i0e<scalar_t>);
        }
    );
}


static inline
void i1_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "i1_cpu", [&] {
            at::native::cpu_kernel(iter, &cephes::cpu::i1<scalar_t>);
        }
    );
}


static inline
void i1e_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "i1e_cpu", [&] {
            at::native::cpu_kernel(iter, &cephes::cpu::i1e<scalar_t>);
        }
    );
}


static inline
void ndtr_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "ndtr_cpu", [&] {
            at::native::cpu_kernel(iter, &cephes::cpu::ndtr<scalar_t>);
        }
    );
}


static inline
void log_ndtr_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "log_ndtr_cpu", [&] {
            at::native::cpu_kernel(iter, &cephes::cpu::log_ndtr<scalar_t>);
        }
    );
}


static inline
void ndtr_log_ndtr_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.input_dtype(), "ndtr_log_ndtr_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x) -> c10::complex<scalar_t> {
                auto result = cephes::cpu::ndtr_log_ndtr<scalar_t>(x);
                return c10::complex<scalar_t>(result.first, result.second);
            });
        }
    );
}


static inline
void ndtr_backward_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "ndtr_backward_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x, scalar_t gout) -> scalar_t {
                // Forward  ndtr(x) = 0.5 [ 1 + erf( x / sqrt(2) )]
                //
                // Erf backward:
                // - name: erf(Tensor self) -> Tensor
                // self: 2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad

                // Backward  grad * 0.5 * 2 / sqrt(pi) * exp( - (x / sqrt(2)).pow(2) ) / sqrt(2)
                //         = grad / sqrt(2 pi) * exp( - x * x / 2)

                // std::sqrt(2 * M_PI);  std::sqrt is not constexpr.
                constexpr scalar_t SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299235763422936546078419749466;
                return std::exp(- x * x / 2) * gout / SQRT_2PI;
            });
        }
    );
}


static inline
void log_ndtr_backward_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "log_ndtr_backward_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x, scalar_t gout) -> scalar_t {
                scalar_t ndtr_val = cephes::cpu::ndtr<scalar_t>(x);
                // std::sqrt(2 * M_PI);  std::sqrt is not constexpr.
                constexpr scalar_t SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299235763422936546078419749466;
                return std::exp(- x * x / 2) * gout / ndtr_val / SQRT_2PI;
            });
        }
    );
}


static inline
void log_ndtr_backward_with_ndtr_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "log_ndtr_backward_with_ndtr_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x, scalar_t ndtr_val, scalar_t gout) -> scalar_t {
                // std::sqrt(2 * M_PI);  std::sqrt is not constexpr.
                constexpr scalar_t SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299235763422936546078419749466;
                return std::exp(- x * x / 2) * gout / ndtr_val / SQRT_2PI;
            });
        }
    );
}


template<TwoPoissonComparisonProb comp, bool use_bounds = true>
static inline
void prob_two_poisson_kernel_cpu(at::TensorIterator& iter) {
    // NOTE [ Skellam CDF Bounds at 0 ]
    //
    // In `chndtr`, we used bounds for general noncentral chi-sq CDFs. Here,
    // we can use a slightly different one based on Poisson race.
    // https://www.wikiwand.com/en/Poisson_distribution#/Poisson_races
    //
    // Poisson race says that for independent X ~ Poisson(mu1), Y ~ Poisson(mu2),
    // mu1 > mu2,
    //     Pr[ X - Y >= 0 ] <= exp{ -(\sqrt{mu1} - \sqrt{mu2})^2 },
    // given by standard Chernoff bound.
    //
    // This implies that
    //
    // 1. if mu1 > mu2,
    //
    //        (\sqrt{mu1} - \sqrt{mu2})^2 >= t
    //     => Pr[ Y <= X ] > 1 - exp(-t).
    //
    // 2. if mu1 < mu2,
    //
    //   Pr[ Y <= X ]  = Pr[X = Y] + Pr[ Y < X ]
    //                 = Pr[X = Y] + 1 - Pr[Y >= X]
    //                >= 1 - Pr[Y >= X]
    //                >= 1 - exp{ -(\sqrt{mu1} - \sqrt{mu2})^2 }.
    //
    //   Side note: we have following fact, which maybe integrated to strengthen the bound:
    //      Pr[ X = Y ]  = Q_1 ( \sqrt{2 mu1}, \sqrt{2 mu2} )
    //                   = exp{ -(\sqrt{mu1} - \sqrt{mu2})^2 } I0e( 2\sqrt{mu1 mu2} ).
    //
    // Hence, if (\sqrt{mu1} - \sqrt{mu2})^2 is large enough, we can just act like an indicator
    // function!
    //
    // But how different is this from the bounds used in `chndtr`. When setting
    //   ` x = 2 mu1`
    //   `df = 2`
    //   `nc = 2 mu2`
    //
    // The `chndtr` bounds reverts to indicator when one of the following is true
    //   1. mu1 >= 1 + mu2 + \sqrt{ (2 + 4 mu2) t } + t
    //   2. mu1 <= 1 + mu2 - \sqrt{ (2 + 4 mu2) t }.
    //
    // The above bound reverts to indicator when one of the following is true
    //   1. mu1 >= mu2 + (\sqrt{mu1} + \sqrt{mu2}) \sqrt{t}
    //   2. mu1 <= mu2 - (\sqrt{mu1} + \sqrt{mu2}) \sqrt{t}.
    //
    // It is not really clear whether the above bound would lead to more improvement
    // frequently. So we do not implement it.

    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "prob_two_poisson_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t mu1, scalar_t mu2) -> scalar_t {
                // See NOTE [ Relation between Non-central Chi Square and Skellam ]
                //
                // Compute  Prob[ NCX2( 2, 2*mu2 ) < 2*mu1 ]
                //
                // df = 2.
                // nc = 2 * mu2.
                // x  = 2 * mu1

                double df = 2;
                double nc = 2 * mu2;
                double x = 2 * mu1;

                if (comp == TwoPoissonComparisonProb::GT) {
                    return cdflib_chndtr<scalar_t, /*use_bounds=*/use_bounds>(x, df, nc);
                } else if (comp == TwoPoissonComparisonProb::LE) {
                    return 1 - cdflib_chndtr<scalar_t, /*use_bounds=*/use_bounds>(x, df, nc);
                } else {
                    __builtin_unreachable();
                }
            });
        }
    );
}

template<TwoPoissonComparisonProb comp, bool use_besselIe = false>
static inline
void prob_two_poisson_grad_mu1_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "prob_two_poisson_grad_mu1_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t mu1, scalar_t mu2, scalar_t gout) -> scalar_t {
                // See NOTE [ Relation between Non-central Chi Square and Skellam ]
                //
                // Compute
                //      auto grad_mu1 = (-mu1 - mu2).exp() * torch::i0(2 * (mu1 * mu2).sqrt()) * gout;

                double g_gt;

                if (!use_besselIe) {
                    g_gt = std::exp(-mu1 - mu2) * ::calc_i0<scalar_t>(std::sqrt(mu1 * mu2) * 2) * gout;
                } else {
                    scalar_t twice_sqrtmu12 = std::sqrt(mu1 * mu2) * 2;
                    g_gt = std::exp(twice_sqrtmu12 - mu1 - mu2) * cephes::cpu::i0e<scalar_t>(twice_sqrtmu12) * gout;
                }

                if (comp == TwoPoissonComparisonProb::GT) {
                    return g_gt;
                } else if (comp == TwoPoissonComparisonProb::LE) {
                    return -g_gt;
                } else {
                    __builtin_unreachable();
                }

            });
        }
    );
}

template<TwoPoissonComparisonProb comp, bool use_bounds = true, bool use_besselIe = false>
static inline
void prob_two_poisson_grad_mu2_kernel_cpu(at::TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(), "prob_two_poisson_grad_mu2_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t mu1, scalar_t mu2, scalar_t out, scalar_t gout) -> scalar_t {
                // See NOTE [ Relation between Non-central Chi Square and Skellam ]
                //
                // Compute
                //      auto grad_mu2 = (chndtr_scalar(2 * mu1, 4, 2 * mu2) - out) * gout;
                if (mu1 == 0) {

                    if (comp == TwoPoissonComparisonProb::GT) {
                        return -out * gout;
                    } else if (comp == TwoPoissonComparisonProb::LE) {
                        return (1 - out) * gout;
                    } else {
                        __builtin_unreachable();
                    }

                } else if (!use_besselIe || mu2 == 0) {
                    // nc = 2 * mu2. When mu2 == 0, the chndtr code computes cdf for Chi-Square isntead.
                    // So let it handle.

                    if (comp == TwoPoissonComparisonProb::GT) {
                        return (cdflib_chndtr<scalar_t, /*use_bounds=*/use_bounds>(2 * mu1, 4, 2 * mu2) - out) * gout;
                    } else if (comp == TwoPoissonComparisonProb::LE) {
                        return (1 - out - cdflib_chndtr<scalar_t, /*use_bounds=*/use_bounds>(2 * mu1, 4, 2 * mu2)) * gout;
                    } else {
                        __builtin_unreachable();
                    }


                } else {
                    scalar_t twice_sqrtmu12 = std::sqrt(mu1 * mu2) * 2;
                    scalar_t log_mu1 = std::log(mu1);
                    scalar_t log_mu2 = std::log(mu2);

                    scalar_t g_le = std::exp(
                        (log_mu1 - log_mu2) / 2 + twice_sqrtmu12 - mu1 - mu2
                    ) * cephes::cpu::i1e<scalar_t>(twice_sqrtmu12) * gout;

                    if (comp == TwoPoissonComparisonProb::GT) {
                        return -g_le;
                    } else if (comp == TwoPoissonComparisonProb::LE) {
                        return g_le;
                    } else {
                        __builtin_unreachable();
                    }


                }
            });
        }
    );
}

} // namespace cpu
} // namespace cdf_ops
