import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_div_gelu_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache optimization: group programs
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Row/col offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Masks for boundary
    mask_m = rm < M
    mask_n = rn < N

    # GEMM: X (M, K) x W^T (K, N) -> (M, N)
    # W is stored as (N, K), so W[n, k] = W_ptr + n * stride_wn + k * stride_wk
    for k_off in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_off * BLOCK_K
        offs_k = k_start + rk
        mask_k = offs_k < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + rm[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N] (W is N x K, we want K x N for dot)
        # W_ptr[n, k] -> we need [k, n] layout for tl.dot
        w_ptrs = W_ptr + rn[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Divide by divisor
    inv_divisor = 1.0 / divisor
    acc = acc * inv_divisor

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476
    x_half = acc * 0.5
    x_erf = libdevice.erf(acc * sqrt2_inv)
    result = x_half * (1.0 + x_erf)

    # Store output
    out_ptrs = OUT_ptr + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, result.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])

def kernel_function(weight, bias, x):
    # weight: (N, K) = (8192, 8192)
    # bias: (N,) = (8192,)
    # x: (M, K) = (1024, 8192)
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    divisor = 10.0

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_div_gelu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        divisor,
    )

    return out


def get_inputs():
    weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        weight, bias, x = inputs
        # Reference: linear -> div -> gelu
        linear_out = x @ weight.t() + bias
        div_out = linear_out / 10.0
        ref = torch.nn.functional.gelu(div_out)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")