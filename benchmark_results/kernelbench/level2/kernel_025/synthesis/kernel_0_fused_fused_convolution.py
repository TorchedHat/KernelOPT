import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    stride_on: tl.constexpr, stride_oc: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] + kh[None, :]
        iw = out_w[:, None] + kw[None, :]

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_n = offs_n < OUT_C

        valid_x = (
            valid_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        w_ptrs = (
        )
        valid_w = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    valid_out = (
        (offs_m < M_TOTAL)[:, None]
    )
    out_ptrs = (
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_stages=1, num_warps=4),
        triton.Config({'XBLOCK': 2}, num_stages=1, num_warps=4),
        triton.Config({'XBLOCK': 4}, num_stages=1, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_stages=1, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def fused_min_bias_tanh_kernel(
    conv_ptr, bias_ptr, out_ptr,
    xnumel, r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    r0_index = tl.arange(0, R0_BLOCK)[None, :]

    # conv_ptr layout: (BATCH, OUT_C, OUT_H, OUT_W) with strides (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    # xindex = batch * OUT_H * OUT_W + oh * OUT_W + ow  (spatial position)
    # We reduce over OUT_C (r0_numel = 64)
    tmp0 = tl.load(conv_ptr + (r0_index + r0_numel * xindex), xmask, other=0.0)
    tmp1 = tl.load(bias_ptr + r0_index, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1

    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp3, float('inf'))
    tmp5 = tl.min(tmp4, axis=1)[:, None].to(tl.float32)

    tmp6 = libdevice.tanh(tmp5)
    tmp7 = libdevice.tanh(tmp6)

    tl.store(out_ptr + xindex, tmp7, mask=xmask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = p_conv_weight.shape
    OUT_H = IN_H - KERNEL_H + 1
    OUT_W = IN_W - KERNEL_W + 1

    # Allocate conv output: (BATCH, OUT_C, OUT_H, OUT_W) in standard NCHW
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    def cdiv(a, b):
        return (a + b - 1) // b

    # Launch conv kernel - we'll use fixed BLOCK_M/N for grid calculation
    # The autotune will pick the best config
    BLOCK_M_FIXED = 256
    BLOCK_N_FIXED = 64

    grid_conv = (cdiv(M_TOTAL, BLOCK_M_FIXED), cdiv(OUT_C, BLOCK_N_FIXED))

    conv2d_kernel[grid_conv](
        x, p_conv_weight, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2), p_conv_weight.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
    )

    # Fused min + bias + tanh + tanh
    # conv_out: (BATCH, OUT_C, OUT_H, OUT_W)
    # We need to reduce over OUT_C dimension
    # Reshape to (BATCH*OUT_H*OUT_W, OUT_C) for the reduction
    # conv_out is contiguous in NCHW: stride = (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    # For the fused kernel, xindex = n*OUT_H*OUT_W + h*OUT_W + w, r0 = c
    # ptr offset = c * OUT_H*OUT_W + xindex  -- but NCHW layout means:
    # offset = n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w
    #        = c*OUT_H*OUT_W + (n*OUT_H*OUT_W + h*OUT_W + w)
    #        = r0 * (OUT_H*OUT_W) + xindex  -- NO, this doesn't work for n>0

    # Let's use a transposed view: (BATCH*OUT_H*OUT_W, OUT_C)
    # conv_out NCHW: (N, C, H, W) -> permute to (N, H, W, C) -> reshape to (N*H*W, C)
    conv_nhwc = conv_out.permute(0, 2, 3, 1).contiguous()  # (BATCH, OUT_H, OUT_W, OUT_C)
    # Now conv_nhwc is (BATCH*OUT_H*OUT_W, OUT_C) effectively

    xnumel = BATCH * OUT_H * OUT_W
    r0_numel = OUT_C  # 64

    out = torch.empty((BATCH, 1, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    out_flat = out.view(xnumel)

    R0_BLOCK = 64  # OUT_C = 64, must be power of 2

    XBLOCK_FIXED = 1
    grid_min = (cdiv(xnumel, XBLOCK_FIXED),)

    fused_min_bias_tanh_kernel[grid_min](
        conv_nhwc, p_conv_bias, out_flat,
        xnumel, r0_numel,
        XBLOCK=XBLOCK_FIXED,
        R0_BLOCK=R0_BLOCK,
    )

    return out


def get_inputs():
    p_conv_weight = torch.randn(64, 16, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 16, 256, 256, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_conv_weight, p_conv_bias, x = inputs
        import torch.nn.functional as F
        conv_ref = F.conv2d(x, p_conv_weight, p_conv_bias)
        min_ref = torch.min(conv_ref, dim=1, keepdim=True)[0]
        tanh1_ref = torch.tanh(min_ref)
        ref = torch.tanh(tanh1_ref)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")