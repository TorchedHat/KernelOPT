import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_div_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly swizzling
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # TMA block_ptr for X: shape (M, K), tile (BLOCK_M, BLOCK_K)
    X_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # TMA block_ptr for W: W is stored as (N, K), presented as (K, N) view
    # strides=(stride_wk, stride_wn) gives us a (K, N) shaped tile (BLOCK_K, BLOCK_N)
    W_block_ptr = tl.make_block_ptr(
        base=W_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero')
        w_tile = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option='zero')

        acc = tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16), acc, allow_tf32=True)

        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_K, 0))

    # Load bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Add bias
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Divide
    acc = acc * (1.0 / divisor)

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    Out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(Out_ptrs, acc.to(tl.float32), mask=out_mask)

def kernel_function(weight, bias, x):
    # weight: (N, K) = (8192, 8192)
    # bias:   (N,)   = (8192,)
    # x:      (M, K) = (1024, 8192)
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    divisor = 2.0

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_linear_relu_div_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        divisor,
    )

    return out


def get_inputs():
    weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, bias, x = inputs

    output = kernel_function(weight, bias, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        # Reference: linear + relu + div
        ref_linear = F.linear(x, weight, bias)
        ref_relu = torch.relu(ref_linear)
        ref_div = ref_relu / 2.0

        assert torch.allclose(output, ref_div, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref_div).abs().max().item()):.4e}"
        print("Kernel executed successfully.")