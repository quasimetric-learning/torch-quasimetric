#include "cephes/ndtr.cuh"
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


void ndtr_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ndtr_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
            return cephes::cuda::ndtr<scalar_t>(x);
        });
    });
}


void log_ndtr_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_ndtr_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
            return cephes::cuda::log_ndtr<scalar_t>(x);
        });
    });
}


void ndtr_log_ndtr_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "ndtr_log_ndtr_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> c10::complex<scalar_t> {
            auto result = cephes::cuda::ndtr_log_ndtr<scalar_t>(x);
            return c10::complex<scalar_t>(result.first, result.second);
        });
    });
}


void ndtr_backward_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ndtr_backward_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x, scalar_t gout) -> scalar_t {
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
    });
}


void log_ndtr_backward_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_ndtr_backward_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x, scalar_t gout) -> scalar_t {
            scalar_t ndtr_val = cephes::cuda::ndtr<scalar_t>(x);
            // std::sqrt(2 * M_PI);  std::sqrt is not constexpr.
            constexpr scalar_t SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299235763422936546078419749466;
            return std::exp(- x * x / 2) * gout / ndtr_val / SQRT_2PI;
        });
    });
}


void log_ndtr_backward_with_ndtr_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_ndtr_backward_with_ndtr_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x, scalar_t ndtr_val, scalar_t gout) -> scalar_t {
            // std::sqrt(2 * M_PI);  std::sqrt is not constexpr.
            constexpr scalar_t SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299235763422936546078419749466;
            return std::exp(- x * x / 2) * gout / ndtr_val / SQRT_2PI;
        });
    });
}

} // namespace cuda
} // namespace cdf_ops
