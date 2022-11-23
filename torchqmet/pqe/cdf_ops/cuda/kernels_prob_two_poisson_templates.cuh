#include "cdflib/chndtr.cuh"
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

template<TwoPoissonComparisonProb comp, bool use_bounds>
void prob_two_poisson_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "prob_two_poisson_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t mu1, scalar_t mu2) -> scalar_t {
            // See NOTE [ Relation between Non-central Chi Square and Skellam ]
            //
            // Compute  Prob[ NCX2( 2, 2*mu2 ) < 2*mu1 ]
            //
            // df = 2.
            // nc = 2 * mu2.
            // x  = 2 * mu1
            scalar_t df = 2;
            scalar_t nc = 2 * mu2;
            scalar_t x = 2 * mu1;

            if (comp == TwoPoissonComparisonProb::GT) {
                return cdflib::cuda::chndtr<scalar_t, /*check_x=*/true, /*check_df=*/false, /*check_pnonc=*/true, /*use_bounds=*/use_bounds>(x, df, nc);
            } else if (comp == TwoPoissonComparisonProb::LE) {
                return 1 - cdflib::cuda::chndtr<scalar_t, /*check_x=*/true, /*check_df=*/false, /*check_pnonc=*/true, /*use_bounds=*/use_bounds>(x, df, nc);
            } else {
                // __builtin_unreachable();
                // Until CUDA 11.3, can't use above.
                // https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
                return NAN;
            }

        });
    });
}

template<TwoPoissonComparisonProb comp, bool use_besselIe>
void prob_two_poisson_grad_mu1_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "prob_two_poisson_grad_mu1_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t mu1, scalar_t mu2, scalar_t gout) -> scalar_t {
            // See NOTE [ Relation between Non-central Chi Square and Skellam ]
            //
            // Compute
            //      auto grad_mu1 = (-mu1 - mu2).exp() * torch::i0(2 * (mu1 * mu2).sqrt()) * gout;

            scalar_t g_gt;

            if (!use_besselIe) {
                g_gt = std::exp(-mu1 - mu2) * cephes::cuda::i0<scalar_t>(std::sqrt(mu1 * mu2) * 2) * gout;
            } else {
                scalar_t twice_sqrtmu12 = std::sqrt(mu1 * mu2) * 2;
                g_gt = std::exp(twice_sqrtmu12 - mu1 - mu2) * cephes::cuda::i0e<scalar_t>(twice_sqrtmu12) * gout;
            }

            if (comp == TwoPoissonComparisonProb::GT) {
                return g_gt;
            } else if (comp == TwoPoissonComparisonProb::LE) {
                return -g_gt;
            } else {
                // __builtin_unreachable();
                // Until CUDA 11.3, can't use above.
                // https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
                return NAN;
            }

        });
    });
}

template<TwoPoissonComparisonProb comp, bool use_bounds, bool use_besselIe>
void prob_two_poisson_grad_mu2_kernel_cuda(at::TensorIterator& iter) {
    const at::cuda::OptionalCUDAGuard device_guard(iter.device());
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "prob_two_poisson_gt_grad_mu2_cuda", [&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t mu1, scalar_t mu2, scalar_t out, scalar_t gout) -> scalar_t {
            // See NOTE [ Relation between Non-central Chi Square and Skellam ]
            //
            // Compute
            //      auto grad_mu2 = (chndtr_scalar(2 * mu1, 4, 2 * mu2) - out) * gout;
            if (mu1 == 0) {

                if (comp == TwoPoissonComparisonProb::GT) {
                    return -out * gout;
                } else if (comp == TwoPoissonComparisonProb::LE) {
                    return (1 - out) * gout;
                } else {
                    // __builtin_unreachable();
                    // Until CUDA 11.3, can't use above.
                    // https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
                    return NAN;
                }

            } else if (!use_besselIe || mu2 == 0) {
                // nc = 2 * mu2. When mu2 == 0, the chndtr code computes cdf for Chi-Square isntead.
                // So let it handle.

                if (comp == TwoPoissonComparisonProb::GT) {
                    return (cdflib::cuda::chndtr<scalar_t, /*use_bounds=*/use_bounds>(2 * mu1, 4, 2 * mu2) - out) * gout;
                } else if (comp == TwoPoissonComparisonProb::LE) {
                    return (1 - out - cdflib::cuda::chndtr<scalar_t, /*use_bounds=*/use_bounds>(2 * mu1, 4, 2 * mu2)) * gout;
                } else {
                    // __builtin_unreachable();
                    // Until CUDA 11.3, can't use above.
                    // https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
                    return NAN;
                }

            } else {
                scalar_t twice_sqrtmu12 = std::sqrt(mu1 * mu2) * 2;
                scalar_t log_mu1 = std::log(mu1);
                scalar_t log_mu2 = std::log(mu2);

                scalar_t g_le = std::exp(
                    (log_mu1 - log_mu2) / 2 + twice_sqrtmu12 - mu1 - mu2
                ) * cephes::cuda::i1e<scalar_t>(twice_sqrtmu12) * gout;

                if (comp == TwoPoissonComparisonProb::GT) {
                    return -g_le;
                } else if (comp == TwoPoissonComparisonProb::LE) {
                    return g_le;
                } else {
                    // __builtin_unreachable();
                    // Until CUDA 11.3, can't use above.
                    // https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
                    return NAN;
                }
            }
        });
    });
}

} // namespace cuda
} // namespace cdf_ops
