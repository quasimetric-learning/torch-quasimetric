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

template<typename scalar_t>
__host__ __device__ __forceinline__
scalar_t rexp ( scalar_t x )

//****************************************************************************80
//
//  Purpose:
//
//    REXP evaluates the function EXP(X) - 1.
//
//  Modified:
//
//    09 December 1999
//
//  Parameters:
//
//    Input, double *X, the argument of the function.
//
//    Output, double REXP, the value of EXP(X)-1.
//
{
  static const scalar_t p1 = .914041914819518e-09;
  static const scalar_t p2 = .238082361044469e-01;
  static const scalar_t q1 = -.499999999085958e+00;
  static const scalar_t q2 = .107141568980644e+00;
  static const scalar_t q3 = -.119041179760821e-01;
  static const scalar_t q4 = .595130811860248e-03;
  scalar_t rexp,w;

    if(std::abs(x) > 0.15e0) goto S10;
    rexp = x*(((p2*x+p1)*x+1.0e0)/((((q4*x+q3)*x+q2)*x+q1)*x+1.0e0));
    return rexp;
S10:
    w = std::exp(x);
    if(x > 0.0e0) goto S20;
    rexp = w-0.5e0-0.5e0;
    return rexp;
S20:
    rexp = w*(0.5e0+(0.5e0-1.0e0/w));
    return rexp;
}

} // namespace cuda
} // namespace cdflib
