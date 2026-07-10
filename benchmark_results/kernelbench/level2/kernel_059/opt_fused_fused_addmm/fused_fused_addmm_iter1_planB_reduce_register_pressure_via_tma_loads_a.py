import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_swish_scale_kernel(
    # Pointers
    x_ptr, w_ptr, bias_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Scaling factor
    scaling_factor,
    # Tile sizes (constexpr)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    # Row/col offsets (used for bias load and output store)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Block pointers for TMA-style loads on Hopper (sm_90)
    # x has shape (M, K), row-major: K is contiguous
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # Weight w has shape (N, K); access as (K, N) for the dot product
    # strides=(stride_wk, stride_wn) with shape=(K, N) correctly transposes
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # Accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load x tile [BLOCK_M, BLOCK_K] via TMA block_ptr
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # Load w tile [BLOCK_K, BLOCK_N] via TMA block_ptr
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))

    # Load bias and add (scalar pointer, unchanged)
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Swish: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig

    # Scale
    acc = acc * scaling_factor

    # Store output (scalar pointer, unchanged)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)

def kernel_function(w, bias, x):
    # x: [M, K], w: [N, K] (Linear weight), bias: [N]
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    scaling_factor = 2.0

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_swish_scale_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor,
    )

    return out


def get_inputs():
    w = torch.rand(32768, 32768, device='cuda', dtype=torch.float32)
    bias = torch.rand(32768, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 32768, device='cuda', dtype=torch.float32)
    return [w, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    w, bias, x = inputs

    output = kernel_function(w, bias, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference computation
        import torch.nn.functional as F
        linear_out = F.linear(x, w, bias)
        ref = linear_out * torch.sigmoid(linear_out) * 2.0

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")