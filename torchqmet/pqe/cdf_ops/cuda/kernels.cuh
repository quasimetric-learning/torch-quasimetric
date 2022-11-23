#include "../cdf_ops.h"

#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

namespace cdf_ops {
namespace cuda {

void chndtr_kernel_cuda(at::TensorIterator& iter);

void chndtr_scalar_double_kernel_cuda(at::TensorIterator& iter, double df);
void chndtr_scalar_scalar_t_kernel_cuda(at::TensorIterator& iter, double df);
void chndtr_scalar_kernel_cuda(at::TensorIterator& iter, double df);

void i0e_kernel_cuda(at::TensorIterator& iter);
void i1_kernel_cuda(at::TensorIterator& iter);
void i1e_kernel_cuda(at::TensorIterator& iter);

template<TwoPoissonComparisonProb comp, bool use_bounds = true>
void prob_two_poisson_kernel_cuda(at::TensorIterator& iter);
template<TwoPoissonComparisonProb comp, bool use_besselIe = true>
void prob_two_poisson_grad_mu1_kernel_cuda(at::TensorIterator& iter);
template<TwoPoissonComparisonProb comp, bool use_bounds = true, bool use_besselIe = true>
void prob_two_poisson_grad_mu2_kernel_cuda(at::TensorIterator& iter);

void ndtr_kernel_cuda(at::TensorIterator& iter);
void log_ndtr_kernel_cuda(at::TensorIterator& iter);
void ndtr_log_ndtr_kernel_cuda(at::TensorIterator& iter);
void ndtr_backward_kernel_cuda(at::TensorIterator& iter);
void log_ndtr_backward_kernel_cuda(at::TensorIterator& iter);
void log_ndtr_backward_with_ndtr_kernel_cuda(at::TensorIterator& iter);

} // namespace cdf_ops
} // namespace cuda
