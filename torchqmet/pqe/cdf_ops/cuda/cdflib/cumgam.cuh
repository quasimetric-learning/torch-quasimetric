#pragma once

#include "gamma_inc.cuh"

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

template<typename scalar_t>
__host__ __device__ __forceinline__
void cumgam ( scalar_t x, scalar_t a, scalar_t *cum, scalar_t *ccum )

//****************************************************************************80
//
//  Purpose:
//
//    CUMGAM evaluates the cumulative incomplete gamma distribution.
//
//  Discussion:
//
//    This routine computes the cumulative distribution function of the
//    incomplete gamma distribution, i.e., the integral from 0 to X of
//
//      (1/GAM(A))*EXP(-T)*T**(A-1) DT
//
//    where GAM(A) is the complete gamma function of A, i.e.,
//
//      GAM(A) = integral from 0 to infinity of EXP(-T)*T**(A-1) DT
//
//  Parameters:
//
//    Input, double *X, the upper limit of integration.
//
//    Input, double *A, the shape parameter of the incomplete
//    Gamma distribution.
//
//    Output, double *CUM, *CCUM, the incomplete Gamma CDF and
//    complementary CDF.
//
{
    if(!(x <= 0.0e0)) goto S10;
    *cum = 0.0e0;
    *ccum = 1.0e0;
    return;
S10:
    gamma_inc<scalar_t, 0> ( a, x, cum, ccum);
//
//     Call gratio routine
//
    return;
}

} // namespace cuda
} // namespace cdflib
