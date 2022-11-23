#pragma once

#include "cumchn.cuh"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <limits>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>


namespace cdflib {
namespace cuda {

template<typename scalar_t, bool use_bounds = true, bool clamp_01 = true>
__host__ __device__ __forceinline__
scalar_t _chndtr(scalar_t x, scalar_t df, scalar_t pnonc) {
    
    static_assert(std::is_same<scalar_t, double>::value || std::is_same<scalar_t, float>::value ,"Unsupported scalar_t");

    // constexpr scalar_t tent9 = 1.0e9;
    // constexpr double tol = 1.0e-8;
    // constexpr double atol = 1.0e-50;
    // constexpr double zero = 1.0e-300;
    // constexpr double one = 1.0e0 - 1.0e-16;
    // constexpr scalar_t inf = std::is_same<scalar_t, double>::value ? 1.0e300 : 1.0e32;

    // if (x > inf) {
    //     x = inf;
    // }
    // if (df > inf) {
    //     df = inf;
    // }
    // if (pnonc > tent9){
    //     pnonc = tent9;
    // }

    // execute cumchn
    // cumchn(x,df,pnonc,p,q);
    //
    // Somehow the following commented code reproduces the CPU result (which always computes in float64)
    // fully with *float32*, yet the scalar_t version (uncommented code) does not with *float64*.
    // I'll just understand this as vectorization.
    // double p, q;
    // cumchn<double>((double) x, (double) df, (double) pnonc, &p, &q);
    scalar_t p, q;
    cumchn<scalar_t>(x, df, pnonc, &p, &q);

    if (clamp_01) {
        if (p > 1) {
            p = 1;
        } else if (p < 0) {
            p = 0;
        }
    }
    return p;
};


constexpr double chndtr_double_thresh = 1.0e3;

template<typename scalar_t, bool check_x = true, bool check_df = true, bool check_pnonc = true, bool use_bounds = true, bool clamp_01 = true>
__host__ __device__ __forceinline__
scalar_t chndtr(scalar_t x, scalar_t df, scalar_t pnonc) {

    if (x != x || df != df || pnonc != pnonc) {
        // cdf_ops doesn't handle NaN well.
        return NAN;
    }

    // bound checks
    CUDA_KERNEL_ASSERT(!(x < 0.0e0));
    CUDA_KERNEL_ASSERT(!(df <= 0.0e0));
    CUDA_KERNEL_ASSERT(!(pnonc < 0.0e0));

    // NOTE [ Non-central Chi-square CDF Bounds ]
    //
    // The following algorithm is *really* slow when both `x` and `nc` are large
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
    //   http://proceedings.mlr.press/v22/kolar12/klar12Supple.pdf
    // Lemma8.

    if (use_bounds) {  
        // CUDA doesn't like constexpr std::log and std::sqrt. Do it ourselves!
        // constexpr scalar_t neglogeps = -std::log(std::numeric_limits<scalar_t>::epsilon());
        // constexpr scalar_t sqrt_neglogeps = std::sqrt(neglogeps);
        static_assert(std::is_same<scalar_t, double>::value || std::is_same<scalar_t, float>::value ,"Unsupported scalar_t");
        static_assert(std::numeric_limits<float>::epsilon() < 1.193e-07 && 1.192e-07 < std::numeric_limits<float>::epsilon());
        static_assert(std::numeric_limits<double>::epsilon() < 2.221e-16 && 1.220e-16 < std::numeric_limits<double>::epsilon());

        constexpr scalar_t neglogeps = std::is_same<scalar_t, double>::value 
            ? 36.0436533891171535515240975655615329742431640625
            : 15.9423847198486328125;
        constexpr scalar_t sqrt_neglogeps = std::is_same<scalar_t, double>::value
            ? 6.00363668030612540604806781630031764507293701171875
            : 3.992791652679443359375;

        const scalar_t mean = df + pnonc;
        const scalar_t term1 = 2 * std::sqrt(df + 2 * pnonc) * sqrt_neglogeps;
        if (x >= mean + term1 + 2 * neglogeps) {
            return 1;
        } else if (x <= mean - term1) {
            return 0;
        }
    }

    if (!std::is_same<scalar_t, double>::value && (
            (check_x     && x     > chndtr_double_thresh) ||
            (check_df    && df    > chndtr_double_thresh) ||
            (check_pnonc && pnonc > chndtr_double_thresh))) {
        return static_cast<scalar_t>(_chndtr<double, use_bounds, clamp_01>(x, df, pnonc));
    } else {
        return _chndtr<scalar_t, use_bounds, clamp_01>(x, df, pnonc);
    }
}

} // namespace cuda
} // namespace cdflib
