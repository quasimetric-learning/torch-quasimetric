#include "cephes/i0.cuh"
#include "cephes/i1.cuh"
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


void i0e_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "i0e_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
            return cephes::cuda::i0e<scalar_t>(x);
        });
    });
}


void i1_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "i1_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
            return cephes::cuda::i1<scalar_t>(x);
        });
    });
}


void i1e_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "i1e_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
            return cephes::cuda::i1e<scalar_t>(x);
        });
    });
}

} // namespace cuda
} // namespace cdf_ops
