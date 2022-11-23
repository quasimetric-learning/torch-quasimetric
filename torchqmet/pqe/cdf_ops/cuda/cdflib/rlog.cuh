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
scalar_t rlog ( scalar_t x )

//****************************************************************************80
//
//  Purpose:
//
//    RLOG computes  X - 1 - LN(X).
//
//  Modified:
//
//    09 December 1999
//
//  Parameters:
//
//    Input, double *X, the argument of the function.
//
//    Output, double RLOG, the value of the function.
//
{
  static const scalar_t a = .566749439387324e-01;
  static const scalar_t b = .456512608815524e-01;
  static const scalar_t p0 = .333333333333333e+00;
  static const scalar_t p1 = -.224696413112536e+00;
  static const scalar_t p2 = .620886815375787e-02;
  static const scalar_t q1 = -.127408923933623e+01;
  static const scalar_t q2 = .354508718369557e+00;
  scalar_t rlog,r,t,u,w,w1;

    if(x < 0.61e0 || x > 1.57e0) goto S40;
    if(x < 0.82e0) goto S10;
    if(x > 1.18e0) goto S20;
//
//              ARGUMENT REDUCTION
//
    u = x-0.5e0-0.5e0;
    w1 = 0.0e0;
    goto S30;
S10:
    u = x-0.7e0;
    u /= 0.7e0;
    w1 = a-u*0.3e0;
    goto S30;
S20:
    u = 0.75e0*x-1.e0;
    w1 = b+u/3.0e0;
S30:
//
//               SERIES EXPANSION
//
    r = u/(u+2.0e0);
    t = r*r;
    w = ((p2*t+p1)*t+p0)/((q2*t+q1)*t+1.0e0);
    rlog = 2.0e0*t*(1.0e0/(1.0e0-r)-r*w)+w1;
    return rlog;
S40:
    r = x-0.5e0-0.5e0;
    rlog = r-std::log(x);
    return rlog;
}
//****************************************************************************80

} // namespace cuda
} // namespace cdflib
