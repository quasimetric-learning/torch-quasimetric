#pragma once

#include "cumgam.cuh"

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
void cumchi ( scalar_t x, scalar_t df, scalar_t *cum, scalar_t *ccum )

//****************************************************************************80
//
//  Purpose:
//
//    CUMCHI evaluates the cumulative chi-square distribution.
//
//  Parameters:
//
//    Input, double *X, the upper limit of integration.
//
//    Input, double *DF, the degrees of freedom of the
//    chi-square distribution.
//
//    Output, double *CUM, the cumulative chi-square distribution.
//
//    Output, double *CCUM, the complement of the cumulative
//    chi-square distribution.
//
{
    scalar_t a = df * 0.5;
    scalar_t xx = x * 0.5;
    cumgam<scalar_t> ( xx, a, cum, ccum );
}

} // namespace cuda
} // namespace cdflib
