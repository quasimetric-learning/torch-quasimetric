import os
import glob
import warnings
import packaging.version

import torch
import torch.version


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


DEBUG_FLAG = check_env_flag('DEBUG', default='0')


def get_source_files():
    files = [
        # Entry
        os.path.join(os.path.dirname(__file__), "cdf_ops.cpp"),
        # CPU
        os.path.join(os.path.dirname(__file__), "cpu", "cdflib", "cdflib.cpp"),
    ]
    # CUDA
    if torch.cuda.is_available():
        files.extend(glob.glob(os.path.join(
            glob.escape(os.path.join(os.path.dirname(__file__), "cuda")),
            "kernels_*.cu",
        )))
    return tuple(files)


def get_extra_cflags():
    if DEBUG_FLAG:
        return ['-O0', '-fopenmp', '-march=native', '-g']
    else:
        return ['-O3', '-fopenmp', '-march=native', '-funroll-loops']


def get_extra_cuda_cflags():
    if not torch.cuda.is_available():
        return []
    if DEBUG_FLAG:
        return ['--expt-relaxed-constexpr', '--expt-extended-lambda', '-O0', '-Xcicc', '-O0', '-Xptxas', '-O0', '-g']
    else:
        return ['--expt-relaxed-constexpr', '--expt-extended-lambda', '-O3']


_extension_loaded: bool = False
_warn_first_load: bool = True


def disable_load_extension_warning():
    global _warn_first_load
    _warn_first_load = False


def load_extension_if_needed():
    global _extension_loaded
    if _extension_loaded:
        return

    if _warn_first_load:
        warnings.warn(
            'Loading `cdf_ops` extension. If this is the first compilation on this machine, '
            'up to 10 minutes is needed. Subsequent loading will use cached results. '
            'Use `pqe.cdf_ops.disable_load_extension_warning()` to suppress this warning.')


    if torch.cuda.is_available() and torch.version.cuda is not None:
        if packaging.version.parse(torch.version.cuda) == packaging.version.parse('11.3'):
            raise RuntimeError(
                'cdf_ops: CUDA 11.3 has a compiler bug that causes compiling `cdf_ops` to hang. '
                'Please use anewer CUDA version.')

    # JIT load
    from torch.utils.cpp_extension import load
    load(
        name="cdf_ops",
        sources=get_source_files(),
        extra_cflags=get_extra_cflags(),
        extra_cuda_cflags=get_extra_cuda_cflags(),
        is_python_module=False,
        with_cuda=torch.cuda.is_available(),
    )
    _extension_loaded = True
