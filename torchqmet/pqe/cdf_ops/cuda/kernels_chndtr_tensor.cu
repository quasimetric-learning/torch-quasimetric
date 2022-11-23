#include "cdflib/chndtr.cuh"
#include "kernels.cuh"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>


namespace cdf_ops {
namespace cuda {

void chndtr_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "chndtr_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x, scalar_t df, scalar_t pnonc) -> scalar_t {
            return cdflib::cuda::chndtr<scalar_t>(x, df, pnonc);
        });
    });
}

} // namespace cuda
} // namespace cdf_ops
