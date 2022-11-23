#include "cdflib/chndtr.cuh"
#include "kernels.cuh"

namespace cdf_ops {
namespace cuda {


void chndtr_scalar_kernel_cuda(at::TensorIterator& iter, double df) {
    // Do dispatch our selves :)
    if (df > cdflib::cuda::chndtr_double_thresh) {
        chndtr_scalar_double_kernel_cuda(iter, df);
    } else {
        chndtr_scalar_scalar_t_kernel_cuda(iter, df);
    }
}

} // namespace cuda
} // namespace cdf_ops
