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
__device__ __host__ __forceinline__
static long fifidint(scalar_t val) {
    return val < 1.0 ? 0 : static_cast<long>(val);
}

} // namespace cuda
} // namespace cdflib
