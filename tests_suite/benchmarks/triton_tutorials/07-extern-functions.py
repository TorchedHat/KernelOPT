"""
Libdevice (`tl.extra.libdevice`) function
==============================
Triton can invoke a custom function from an external library.
In this example, we will use the `libdevice` library to apply `asin` on a tensor.

Please refer to `CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_ and/or `HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_ regarding the semantics of all available libdevice functions.

In `libdevice.py`, we try to aggregate functions with the same computation but different data types together.
For example, both `__nv_asin` and `__nv_asinf` calculate the principal value of the arc sine of the input, but `__nv_asin` operates on `double` and `__nv_asinf` operates on `float`.
Triton automatically selects the correct underlying device function to invoke based on input and output types.
"""

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# ===== kpt interface =====

def reference_function(x):
    return torch.asin(x)


def kernel_function(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    asin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def get_inputs():
    return [torch.rand(98432, device=DEVICE)]


if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    if "--profile-only" in sys.argv:
        kernel_function(*inputs)
        torch.cuda.synchronize()
    else:
        kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
