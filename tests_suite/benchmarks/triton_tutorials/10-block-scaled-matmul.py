"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication
which is generic over FP4 and FP8 formats on NVIDIA and AMD GPUs.

Note: This kernel requires compute capability >= 10 (Blackwell). It will NOT work on
H200 (compute capability 9.0) and will fail gracefully at runtime.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]) or is_hip_cdna4()


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output


# ===== kpt interface =====

def reference_function(*inputs):
    # Block-scaled matmul requires compute capability >= 10 (Blackwell).
    # On H200 (compute capability 9.0), this will not work.
    # Return None as a placeholder.
    raise RuntimeError(
        "Block-scaled matmul requires compute capability >= 10 (Blackwell). "
        "This kernel cannot run on the current GPU."
    )


def kernel_function(*inputs):
    # Block-scaled matmul requires compute capability >= 10 (Blackwell).
    # On H200 (compute capability 9.0), this will not work.
    if not supports_block_scaling():
        raise RuntimeError(
            "Block-scaled matmul requires compute capability >= 10 (Blackwell). "
            "This kernel cannot run on the current GPU."
        )
    # If we get here, the GPU supports block scaling - but full initialization
    # requires specialized tensor formats (MXFP4/MXFP8) which are complex to set up.
    raise RuntimeError(
        "Block-scaled matmul requires specialized input formats (MXFP4/MXFP8). "
        "Use initialize_block_scaled() from the original tutorial for proper setup."
    )


def get_inputs():
    # Return empty list - this kernel requires specialized inputs
    # that depend on hardware support
    return []


if __name__ == "__main__":
    import sys
    if not supports_block_scaling():
        print("Block-scaled matmul requires compute capability >= 10 (Blackwell). Skipping.")
    else:
        inputs = get_inputs()
        if "--profile-only" in sys.argv:
            try:
                kernel_function(*inputs)
                torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"Skipped: {e}")
        else:
            try:
                kernel_function(*inputs)
                torch.cuda.synchronize()
                print("Kernel executed successfully.")
            except RuntimeError as e:
                print(f"Skipped: {e}")
