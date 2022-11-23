#pragma once

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

template<typename scalar_t, bool positive>
static constexpr scalar_t exparg()

//****************************************************************************80
//
//  Purpose:
//
//    EXPARG returns the largest or smallest legal argument for EXP.
//
//  Discussion:
//
//    Only an approximate limit for the argument of EXP is desired.
//
//  Modified:
//
//    09 December 1999
//
//  Parameters:
//
//    Input, int *L, indicates which limit is desired.
//    If L = 0, then the largest positive argument for EXP is desired.
//    Otherwise, the largest negative argument for EXP for which the
//    result is nonzero is desired.
//
//    Output, double EXPARG, the desired value.
//
{
    constexpr double lnb = .69314718055995e0;  // ln of base = 2
    int m = 0;

    if (positive) {
        m = std::numeric_limits<scalar_t>::max_exponent;  // the largest exponent E for double precision.
    } else {
        m = std::numeric_limits<scalar_t>::min_exponent - 1;  // the smallest exponent E for double precision, then - 1.
    }
    return static_cast<scalar_t>(0.99999e0*((double)m*lnb));
}

} // namespace cuda
} // namespace cdflib
