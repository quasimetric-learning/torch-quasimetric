#include "cdf_ops.h"
#include "cpu/kernels.h"
#include "cuda/kernels.cuh"

#include <cmath>
#include <limits>
#include <torch/script.h>
#include <torch/extension.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Math.h>



namespace cdf_ops {


torch::Tensor chndtr(const torch::Tensor& _x, const torch::Tensor& _df, const torch::Tensor& _nc) {
    // Non-central chi square cumulative distribution function
    // Reference https://github.com/scipy/scipy/blob/5f4c4d802e5a56708d86909af6e5685cd95e6e66/scipy/special/cdf_wrappers.c#L157-L166

    auto binps = torch::broadcast_tensors({_x, _df, _nc});
    torch::Tensor& x = binps[0];
    torch::Tensor& df = binps[1];
    torch::Tensor& nc = binps[2];
    torch::Tensor out = torch::empty_like(x);

    at::TensorIteratorConfig iter_config;
    iter_config
        .add_output(out)
        .add_input(x)
        .add_input(df)
        .add_input(nc)
        .promote_inputs_to_common_dtype(true);

    auto iter = iter_config.build();
    auto device = iter.device();

    if (device.is_cpu()) {
        cdf_ops::cpu::chndtr_kernel_cpu(iter);
    } else {
        cdf_ops::cuda::chndtr_kernel_cuda(iter);
    }
    return out;
}


torch::Tensor chndtr_scalar(const torch::Tensor& _x, double df, const torch::Tensor& _nc) {
    // Non-central chi square cumulative distribution function
    // Reference https://github.com/scipy/scipy/blob/5f4c4d802e5a56708d86909af6e5685cd95e6e66/scipy/special/cdf_wrappers.c#L157-L166

    auto binps = torch::broadcast_tensors({_x, _nc});
    torch::Tensor& x = binps[0];
    torch::Tensor& nc = binps[1];
    torch::Tensor out = torch::empty_like(x);

    at::TensorIteratorConfig iter_config;
    iter_config
        .add_output(out)
        .add_input(x)
        .add_input(nc)
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true);

    auto iter = iter_config.build();
    auto device = iter.device();

    if (device.is_cpu()) {
        cdf_ops::cpu::chndtr_scalar_kernel_cpu(iter, df);
    } else {
        cdf_ops::cuda::chndtr_scalar_kernel_cuda(iter, df);
    }
    return out;
}


torch::Tensor i0e(const torch::Tensor& x) {
    torch::Tensor out = torch::empty_like(x);

    at::TensorIteratorConfig iter_config;
    iter_config
        .add_output(out)
        .add_input(x)
        .allow_cpu_scalars(true);

    auto iter = iter_config.build();
    auto device = iter.device();

    if (device.is_cpu()) {
        cdf_ops::cpu::i0e_kernel_cpu(iter);
    } else {
        cdf_ops::cuda::i0e_kernel_cuda(iter);
    }
    return out;
}


torch::Tensor i1(const torch::Tensor& x) {
    torch::Tensor out = torch::empty_like(x);

    at::TensorIteratorConfig iter_config;
    iter_config
        .add_output(out)
        .add_input(x)
        .allow_cpu_scalars(true);

    auto iter = iter_config.build();
    auto device = iter.device();

    if (device.is_cpu()) {
        cdf_ops::cpu::i1_kernel_cpu(iter);
    } else {
        cdf_ops::cuda::i1_kernel_cuda(iter);
    }
    return out;
}


torch::Tensor i1e(const torch::Tensor& x) {
    torch::Tensor out = torch::empty_like(x);

    at::TensorIteratorConfig iter_config;
    iter_config
        .add_output(out)
        .add_input(x)
        .allow_cpu_scalars(true);

    auto iter = iter_config.build();
    auto device = iter.device();

    if (device.is_cpu()) {
        cdf_ops::cpu::i1e_kernel_cpu(iter);
    } else {
        cdf_ops::cuda::i1e_kernel_cuda(iter);
    }
    return out;
}


template<TwoPoissonComparisonProb comp>
class ProbTwoPoissonCompFunction : public torch::autograd::Function<ProbTwoPoissonCompFunction<comp>> {
    // NOTE [ Relation between Non-central Chi Square and Skellam ]
    //
    // Implementation based on SciPy: https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/_discrete_distns.py
    //
    // Refer to Eqn (4) in On an Extension of the Connetion Between Poisson and chi-sq2 Distributions, N.L Johnson(1959)
    // Vol 46, No 3/4, doi:10.2307/2333532
    // https://www.jstor.org/stable/2333532
    //
    // It relates the Skellam and Non-central chisquare PDFs, which is very similar to their CDFs computation as well.
    // Specifically, for the non-central chisquare with degree of freedom `df` (> 0) and non-centrality parameter `nc` (>= 0)
    // we have
    //
    // For x >= 0,
    //      Prob[ NCX2( df, nc ) < x ] = Prob[ Poi(x/2) - Poi(nc/2) >= df/2 ]
    //                                 = Prob[ Skellam(x/2, nc/2) >= df/2 ]
    //                                 = 1 - Prob[ Skellam(x/2, nc/2) < df/2 ]
    //                              OR = 1 - Prob[ Skellam(x/2, nc/2) <= df/2 - 1 ]
    //                              OR = Prob[ Skellam(nc/2, x/2) <= -df/2 ]
    //                              OR = Prob[ Skellam(nc/2, x/2) < -df/2 - 1 ]
    //
    // Thus, we have
    //      Prob[ Poi(mu1) > Poi(mu2) ] = 1 - Prob[ Skellam(mu1, mu2) <= 0]
    //                                  = Prob[ NCX2( 2, 2*mu2 ) < 2*mu1 ]
    //                                  = 1 - MarcumQ( 1, sqrt{2*mu2}, sqrt{2*mu1} ),
    //
    // And generally, for Skellam(mu1, mu2), integral x,
    //
    //     Prob[ Skellam(mu1, mu2) <= x ] =
    //                         (if x < 0) = Prob[ NCX2( -2x, 2*mu1 ) < 2*mu2 ]
    //                                    = 1 - MarcumQ( -x, sqrt{2*mu1}, sqrt{2*mu2} )
    //                        (if x >= 0) = 1 - Prob[ NCX2( 2(x+1), 2*mu2 ) < 2*mu1 ]
    //                                    = MarcumQ( x+1, sqrt{2*mu2}, sqrt{2*mu1} )
    //
    // (Consistent with the SciPy implementation
    //  https://github.com/scipy/scipy/blob/5f4c4d802e5a56708d86909af6e5685cd95e6e66/scipy/stats/_discrete_distns.py#L1160-L1165)
    //
    // where the last relation with Marcum Q function is from the NCX2 CDF. From this, we can derive
    // the backward formula for mu1 and mu2. C.f. On some properties of the Marcum Q function
    // https://www.researchgate.net/publication/233152406_On_some_properties_of_the_Marcum_Q_function,
    // Formula (16) and (19), we have
    //
    //       d/(d mu1) 1 - MarcumQ( 1, sqrt{2*mu2}, sqrt{2*mu1} )
    //     = exp(-mu1 -mu2) * I_0( 2 sqrt{mu1*mu2} )
    //
    //       d/(d mu2) 1 - MarcumQ( 1, sqrt{2*mu2}, sqrt{2*mu1} )
    //     = MarcumQ( 1, sqrt{2*mu2}, sqrt{2*mu1} ) - MarcumQ( 2, sqrt{2*mu2}, sqrt{2*mu1} )
    //     = 1 - OUTPUT - (1 - Prob[ NCX2( 4, 2*mu2 ) < 2*mu1 ])
    //     = Prob[ NCX2( 4, 2*mu2 ) < 2*mu1 ] - OUTPUT
    //
    // See also ./_test_scripts/poisson_diff_le_zero_prob.py for more identities and the backward formula.

    public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor& _mu1, const torch::Tensor& _mu2,
        bool backward_use_besselIe,
        bool chndtr_use_bounds
    ) {
        auto binps = torch::broadcast_tensors({_mu1, _mu2});
        torch::Tensor& mu1 = binps[0];
        torch::Tensor& mu2 = binps[1];
        torch::Tensor out = torch::empty_like(mu1);

        at::TensorIteratorConfig iter_config;
        iter_config
            .add_output(out)
            .add_input(mu1)
            .add_input(mu2)
            .allow_cpu_scalars(true)
            .promote_inputs_to_common_dtype(true);

        auto iter = iter_config.build();
        auto device = iter.device();

        if (device.is_cpu()) {
            if (chndtr_use_bounds) {
                cdf_ops::cpu::prob_two_poisson_kernel_cpu<comp, /*use_bounds=*/true>(iter);
            } else {
                throw std::runtime_error("chndtr_use_bounds=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it.");
                // cdf_ops::cpu::prob_two_poisson_kernel_cpu<comp, /*use_bounds=*/false>(iter);
            }
        } else {
            if (chndtr_use_bounds) {
                cdf_ops::cuda::prob_two_poisson_kernel_cuda<comp, /*use_bounds=*/true>(iter);
            } else {
                throw std::runtime_error("chndtr_use_bounds=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it.");
                // cdf_ops::cuda::prob_two_poisson_kernel_cuda<comp, /*use_bounds=*/false>(iter);
            }
        }

        ctx->save_for_backward({mu1, mu2, out});
        ctx->saved_data["backward_use_besselIe"] = backward_use_besselIe;
        ctx->saved_data["chndtr_use_bounds"] = chndtr_use_bounds;
        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs
    ) {
        bool backward_use_besselIe = ctx->saved_data["backward_use_besselIe"].toBool();
        bool chndtr_use_bounds = ctx->saved_data["chndtr_use_bounds"].toBool();
        auto saved = ctx->get_saved_variables();
        auto mu1 = saved[0];
        auto mu2 = saved[1];
        auto out = saved[2];
        auto gout = grad_outputs[0];

        auto device = mu1.device();

        // grad_mu1
        torch::Tensor grad_mu1 = torch::empty_like(mu1);
        {
            at::TensorIteratorConfig iter_config;
            iter_config
                .add_output(grad_mu1)
                .add_input(mu1)
                .add_input(mu2)
                .add_input(gout)
                .allow_cpu_scalars(true)
                .promote_inputs_to_common_dtype(true);

            auto iter = iter_config.build();

            if (device.is_cpu()) {
                if (backward_use_besselIe) {
                    cdf_ops::cpu::prob_two_poisson_grad_mu1_kernel_cpu<comp, /*use_besselIe=*/true>(iter);
                } else {
                    throw std::runtime_error("backward_use_besselIe=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it.");
                    // cdf_ops::cpu::prob_two_poisson_grad_mu1_kernel_cpu<comp, /*use_besselIe=*/false>(iter);
                }
            } else {
                if (backward_use_besselIe) {
                    cdf_ops::cuda::prob_two_poisson_grad_mu1_kernel_cuda<comp, /*use_besselIe=*/true>(iter);
                } else {
                    throw std::runtime_error("backward_use_besselIe=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it.");
                    // cdf_ops::cuda::prob_two_poisson_grad_mu1_kernel_cuda<comp, /*use_besselIe=*/false>(iter);
                }
            }
        }

        // grad_mu2
        torch::Tensor grad_mu2 = torch::empty_like(mu1);
        {
            at::TensorIteratorConfig iter_config;
            iter_config
                .add_output(grad_mu2)
                .add_input(mu1)
                .add_input(mu2)
                .add_input(out)
                .add_input(gout)
                .allow_cpu_scalars(true)
                .promote_inputs_to_common_dtype(true);

            auto iter = iter_config.build();

#define MU2_HANDLE_CHNDTR_USE_BOUNDS(kernel, backward_use_besselIe)                                       \
            do {                                                                                          \
                if (chndtr_use_bounds) {                                                                  \
                    kernel<comp, /*use_bounds=*/true, /*use_besselIe=*/backward_use_besselIe>(iter);      \
                } else {                                                                                  \
                    throw std::runtime_error("chndtr_use_bounds=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it."); \
                }                                                                                         \
            } while (0)

#define MU2_HANDLE_BACKWARD_USE_BESSELIE_CHNDTR_USE_BOUNDS(kernel)                                        \
            do {                                                                                          \
                if (backward_use_besselIe) {                                                              \
                    MU2_HANDLE_CHNDTR_USE_BOUNDS(kernel, /*backward_use_besselIe=*/true);                 \
                } else {                                                                                  \
                    throw std::runtime_error("backward_use_besselIe=False is disabled for compilation speed. Modify this file and the instantiations in CUDA kernels if you need it."); \
                }                                                                                         \
            } while (0)


            if (device.is_cpu()) {
                MU2_HANDLE_BACKWARD_USE_BESSELIE_CHNDTR_USE_BOUNDS(cdf_ops::cpu::prob_two_poisson_grad_mu2_kernel_cpu);
            } else {
                MU2_HANDLE_BACKWARD_USE_BESSELIE_CHNDTR_USE_BOUNDS(cdf_ops::cuda::prob_two_poisson_grad_mu2_kernel_cuda);
            }

#undef MU2_HANDLE_CHNDTR_USE_BOUNDS
#undef MU2_HANDLE_BACKWARD_USE_BESSELE_CHNDTR_USE_BOUNDS

        }

        return {grad_mu1, grad_mu2, torch::Tensor(), torch::Tensor()};
    }
};


torch::Tensor prob_two_poisson_gt(const torch::Tensor& mu1, const torch::Tensor& mu2,
                                  bool backward_use_besselIe = true, bool chndtr_use_bounds = true) {
    return ProbTwoPoissonCompFunction<TwoPoissonComparisonProb::GT>::apply(mu1, mu2, backward_use_besselIe, chndtr_use_bounds);
}


torch::Tensor prob_two_poisson_le(const torch::Tensor& mu1, const torch::Tensor& mu2,
                                  bool backward_use_besselIe = true, bool chndtr_use_bounds = true) {
    return ProbTwoPoissonCompFunction<TwoPoissonComparisonProb::LE>::apply(mu1, mu2, backward_use_besselIe, chndtr_use_bounds);
}


class NdtrFunction : public torch::autograd::Function<NdtrFunction> {
    public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx, const torch::Tensor& x) {

        torch::Tensor out = torch::empty_like(x);

        at::TensorIteratorConfig iter_config;
        iter_config
            .add_output(out)
            .add_input(x);

        auto iter = iter_config.build();
        auto device = iter.device();

        if (device.is_cpu()) {
            cdf_ops::cpu::ndtr_kernel_cpu(iter);
        } else {
            cdf_ops::cuda::ndtr_kernel_cuda(iter);
        }

        ctx->save_for_backward({x});
        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto gout = grad_outputs[0];

        auto device = x.device();

        // grad_x
        torch::Tensor grad_x = torch::empty_like(x);
        at::TensorIteratorConfig iter_config;
        iter_config
            .add_output(grad_x)
            .add_input(x)
            .add_input(gout);

        auto iter = iter_config.build();

        if (device.is_cpu()) {
            cdf_ops::cpu::ndtr_backward_kernel_cpu(iter);
        } else {
            cdf_ops::cuda::ndtr_backward_kernel_cuda(iter);
        }
        return {grad_x};
    }
};


torch::Tensor ndtr(const torch::Tensor& x) {
    return NdtrFunction::apply(x);
}


class LogNdtrFunction : public torch::autograd::Function<LogNdtrFunction> {
    public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx, const torch::Tensor& x
    ) {

        if (!x.requires_grad()) {

            torch::Tensor out = torch::empty_like(x);

            at::TensorIteratorConfig iter_config;
            iter_config
                .add_output(out)
                .add_input(x);

            auto iter = iter_config.build();
            auto device = iter.device();

            if (device.is_cpu()) {
                cdf_ops::cpu::log_ndtr_kernel_cpu(iter);
            } else {
                cdf_ops::cuda::log_ndtr_kernel_cuda(iter);
            }

            return out;
        } else {

            std::vector<int64_t> x_shape = x.sizes().vec();
            x_shape.push_back(2);

            torch::Tensor ndtr_log_ndtr = torch::empty(x_shape, x.options());

            at::TensorIteratorConfig iter_config;
            auto ndtr_log_ndtr_as_complex = torch::view_as_complex(ndtr_log_ndtr);  // Really the kernel will put ndtr in real and logndtr in imag
            iter_config
                .add_output(ndtr_log_ndtr_as_complex)
                .add_input(x)
                .check_all_same_dtype(false);

            auto iter = iter_config.build();
            auto device = iter.device();

            if (device.is_cpu()) {
                cdf_ops::cpu::ndtr_log_ndtr_kernel_cpu(iter);
            } else {
                cdf_ops::cuda::ndtr_log_ndtr_kernel_cuda(iter);
            }

            auto split_ndtr_log_ndtr = ndtr_log_ndtr.unbind(-1);
            auto ndtrs = split_ndtr_log_ndtr[0];
            auto log_ndtrs = split_ndtr_log_ndtr[1];

            ctx->save_for_backward({x, ndtrs});
            return log_ndtrs;
        }
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto ndtrs = saved[1];
        auto gout = grad_outputs[0];

        auto device = x.device();

        // grad_x
        torch::Tensor grad_x = torch::empty_like(x);
        at::TensorIteratorConfig iter_config;
        iter_config
            .add_output(grad_x)
            .add_input(x)
            .add_input(ndtrs)
            .add_input(gout);

        auto iter = iter_config.build();

        if (device.is_cpu()) {
            cdf_ops::cpu::log_ndtr_backward_with_ndtr_kernel_cpu(iter);
        } else {
            cdf_ops::cuda::log_ndtr_backward_with_ndtr_kernel_cuda(iter);
        }
        return {grad_x};
    }
};


torch::Tensor log_ndtr(const torch::Tensor& x) {
    return LogNdtrFunction::apply(x);
}


class ProdNdtrFunction : public torch::autograd::Function<ProdNdtrFunction> {
    // Why BP through log_ndtr is unstable?
    //
    // Think x really small \approx -\infty.
    //
    // d / dx log ndtr(x) = ndtr'(x) / ndtr(x) = ndtr'(x) / 0.
    //
    // So, to BP through \prod ndtr but want numerical stability from log, we
    // shall use a customized BP rule.
    public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor& x,
        int64_t dim
    ) {
        int64_t wrapped_dim = at::maybe_wrap_dim(dim, x.sizes().size());

        torch::Tensor log_ndtrs = torch::empty_like(x);

        at::TensorIteratorConfig log_ndtr_iter_config;
        log_ndtr_iter_config
            .add_output(log_ndtrs)
            .add_input(x);

        auto log_ndtr_iter = log_ndtr_iter_config.build();
        auto device = log_ndtr_iter.device();

        if (device.is_cpu()) {
            cdf_ops::cpu::log_ndtr_kernel_cpu(log_ndtr_iter);
        } else {
            cdf_ops::cuda::log_ndtr_kernel_cuda(log_ndtr_iter);
        }

        auto sum_log_ndtrs = log_ndtrs.sum(wrapped_dim);
        if (x.requires_grad()) {
            ctx->save_for_backward({x, log_ndtrs, sum_log_ndtrs});
            ctx->saved_data["wrapped_dim"] = wrapped_dim;
            return sum_log_ndtrs.exp();
        } else {
            return sum_log_ndtrs.exp_();
        }
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs
    ) {
        int64_t wrapped_dim = ctx->saved_data["wrapped_dim"].toInt();

        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto log_ndtrs = saved[1];
        auto sum_log_ndtrs = saved[2];
        auto gout = grad_outputs[0];

        auto device = x.device();

        torch::Tensor log_ndtr_sum_others;

        if (torch::isfinite(sum_log_ndtrs).all().item<bool>()) {
            log_ndtr_sum_others = torch::sub_out(
                /*out=*/log_ndtrs,
                /*self=*/sum_log_ndtrs.unsqueeze(wrapped_dim),
                /*other=*/log_ndtrs);
        } else {
            // Compute the gradient for the ndtrs. This is essentially bp through exp sum log_ndtrs.
            // NB that the naive approach of exp(log_ndtrs.sum() - log_ndtrs) have issues if
            // log_ndtrs contains -\infty's, leading to NaN's in the subtraction results.
            // Instead, we need to use the same idea as in here
            // https://github.com/pytorch/pytorch/blob/61b074581ce1ccf0fb1bf4f1b73f4b99f93fa70c/torch/csrc/autograd/FunctionsManual.cpp#L472-L478
            // and compute
            //      log_ndtrs.cumsum(exclusive, normal) + log_ndtrs.cumsum(exclusive, reverse).

            auto zeros_size = log_ndtrs.sizes().vec();
            zeros_size[wrapped_dim] = 1;
            auto zeros = torch::zeros(zeros_size, log_ndtrs.options());
            log_ndtr_sum_others = torch::cumsum(
                torch::cat({zeros, log_ndtrs.narrow(wrapped_dim, 0, log_ndtrs.size(wrapped_dim) - 1)}, wrapped_dim),
                wrapped_dim
            );
            auto reverse_idx = torch::arange(log_ndtrs.size(wrapped_dim) - 1, -1, -1, log_ndtrs.options().dtype(torch::kLong));
            log_ndtr_sum_others.add_(
                torch::cumsum(
                    torch::cat(
                        {
                            zeros,
                            log_ndtrs.index_select(wrapped_dim, reverse_idx.narrow(0, 0, log_ndtrs.size(wrapped_dim) - 1))
                        },
                        wrapped_dim
                    ),
                    wrapped_dim
                ).index_select(wrapped_dim, reverse_idx)
            );
        }

        auto gndtrs = log_ndtr_sum_others.exp_().mul_(gout.unsqueeze(wrapped_dim));

        // grad_x
        torch::Tensor grad_x = torch::empty_like(x);
        at::TensorIteratorConfig iter_config;
        iter_config
            .add_output(grad_x)
            .add_input(x)
            .add_input(gndtrs);

        auto iter = iter_config.build();

        if (device.is_cpu()) {
            cdf_ops::cpu::ndtr_backward_kernel_cpu(iter);
        } else {
            cdf_ops::cuda::ndtr_backward_kernel_cuda(iter);
        }
        return {grad_x, torch::Tensor()};
    }
};


torch::Tensor prod_ndtr(const torch::Tensor& x, int64_t dim = -1) {
    if (x.size(dim) == 1) {
        return NdtrFunction::apply(x).squeeze(dim);
    } else {
        return ProdNdtrFunction::apply(x, dim);
    }
}


TORCH_LIBRARY(cdf_ops, m) {
    m.def("chndtr.tensor(Tensor x, Tensor df, Tensor nc) -> Tensor", &chndtr);
    m.def("chndtr.scalar(Tensor x, float df, Tensor nc) -> Tensor", &chndtr_scalar);
    m.def("i0e(Tensor x) -> Tensor", &i0e);
    m.def("i1(Tensor x) -> Tensor", &i1);
    m.def("i1e(Tensor x) -> Tensor", &i1e);
    m.def("prob_two_poisson_gt(Tensor mu1, Tensor mu2, *, bool backward_use_besselIe=True, bool chndtr_use_bounds=True) -> Tensor", &prob_two_poisson_gt);
    m.def("prob_two_poisson_le(Tensor mu1, Tensor mu2, *, bool backward_use_besselIe=True, bool chndtr_use_bounds=True) -> Tensor", &prob_two_poisson_le);
    m.def("ndtr(Tensor x) -> Tensor", &ndtr);
    m.def("prod_ndtr(Tensor x, int dim=-1) -> Tensor", &prod_ndtr);
    m.def("log_ndtr(Tensor x) -> Tensor", &log_ndtr);
}

} // namespace cdf_ops
