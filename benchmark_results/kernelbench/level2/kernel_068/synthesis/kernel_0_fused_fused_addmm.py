import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_min_sub_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    constant_val,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to x and w tiles
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is (N, K) = (out_features, in_features), we want w[n, k]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # Load x tile [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile [BLOCK_K, BLOCK_N] - weight is (out, in), access w[n, k] -> transpose
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Load bias [BLOCK_N]
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)

    # Add bias
    acc = acc + bias[None, :]

    # Apply min(acc, constant)
    acc = tl.minimum(acc, constant_val)

    # Subtract constant
    acc = acc - constant_val

    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_linear_weight, p_linear_bias, p_constant, x):
    # p_linear_weight: [out_features, in_features] = [16384, 16384]
    # p_linear_bias: [out_features] = [16384]
    # p_constant: scalar tensor []
    # x: [batch, in_features] = [128, 16384]

    assert p_linear_weight.ndim == 2, f"Weight must be 2D, got shape {p_linear_weight.shape}"
    
    M = x.shape[0]          # batch size = 128
    K = x.shape[1]          # in_features = 16384
    N = p_linear_weight.shape[0]  # out_features = 16384

    # Ensure contiguous
    x = x.contiguous()
    w = p_linear_weight.contiguous()  # shape [N, K]
    bias = p_linear_bias.contiguous()

    constant_val = p_constant.item()

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_min_sub_kernel[grid](
        x, w, bias, out,
        constant_val,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
    )

    return out


def get_inputs():
    p_linear_weight = torch.randn(16384, 16384, dtype=torch.float32, device='cuda')
    p_linear_bias = torch.randn(16384, dtype=torch.float32, device='cuda')
    p_constant = torch.tensor(2.0, dtype=torch.float32, device='cuda')
    x = torch.rand(128, 16384, dtype=torch.float32, device='cuda')
    return [p_linear_weight, p_linear_bias, p_constant, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_linear_weight, p_linear_bias, p_constant, x = inputs
        # Reference computation
        import torch.nn.functional as F
        linear_out = F.linear(x, p_linear_weight, p_linear_bias)
        min_out = torch.min(linear_out, p_constant)
        ref = min_out - p_constant

        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Ref range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")