#include "kernels_prob_two_poisson_templates.cuh"


namespace cdf_ops {
namespace cuda {

// Explicit instantiations
template void prob_two_poisson_grad_mu2_kernel_cuda<TwoPoissonComparisonProb::LE, true, true>(at::TensorIterator& iter);
// template void prob_two_poisson_grad_mu2_kernel_cuda<TwoPoissonComparisonProb::LE, true, false>(at::TensorIterator& iter);
// template void prob_two_poisson_grad_mu2_kernel_cuda<TwoPoissonComparisonProb::LE, false, true>(at::TensorIterator& iter);
// template void prob_two_poisson_grad_mu2_kernel_cuda<TwoPoissonComparisonProb::LE, false, false>(at::TensorIterator& iter);

} // namespace cuda
} // namespace cdf_ops
