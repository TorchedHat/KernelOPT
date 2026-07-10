import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # 64x64 tiles: accumulator = 64*64*4 / 128 threads = 128 bytes = 32 float regs/thread
        # smem(stages=4, BLOCK_K=32): 4*(64*32+32*64)*4 = 65536 bytes = 64KB
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        # 64x128 tiles: accumulator = 64*128*4 / 128 threads = 256 bytes = 64 float regs/thread
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        # 128x64 tiles: same accumulator budget as 64x128
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sigmoid_scale_residual_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly ordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Row/col offsets (still needed for bias load)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # TMA block_ptr for X: shape [M, K], row-major (K contiguous)
    X_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # TMA block_ptr for W: stored as [N, K], accessed as transposed [K, N]
    # shape=(K, N), strides=(stride_wk, stride_wn)
    W_block_ptr = tl.make_block_ptr(
        base=W_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    # Accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero')
        w_tile = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc = tl.dot(x_tile, w_tile, acc, input_precision='tf32x3')
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_K, 0))

    # Load bias (keep scalar 1D load — bias is a vector, not a 2D tile)
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Add bias -> this is original_x
    original_x = acc + bias[None, :]

    # Sigmoid
    sig = tl.sigmoid(original_x.to(tl.float32))

    # Scale
    scaled = sig * scaling_factor

    # Residual add
    result = scaled + original_x

    # Store via block_ptr (Out is [M, N] row-major, stride N, 1)
    Out_block_ptr = tl.make_block_ptr(
        base=Out_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(Out_block_ptr, result.to(tl.float32), boundary_check=(0, 1))

def kernel_function(p_gemm_weight, p_gemm_bias, x):
    # p_gemm_weight: [N, K] = [8192, 8192]
    # p_gemm_bias:   [N]    = [8192]
    # x:             [M, K] = [1024, 8192]
    M, K = x.shape
    N = p_gemm_weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Weight is [N, K], stride_wn = K, stride_wk = 1
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_wn, stride_wk = p_gemm_weight.stride(0), p_gemm_weight.stride(1)

    scaling_factor = 2.0

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    gemm_sigmoid_scale_residual_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, out,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        scaling_factor,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_gemm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, x = inputs

        # Reference computation
        linear_out = x @ p_gemm_weight.T + p_gemm_bias
        original_x = linear_out
        sig = torch.sigmoid(linear_out)
        scaled = sig * 2.0
        ref = scaled + original_x

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")