import torch
import triton
import triton.language as tl

# Implicit GEMM conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KH * KW
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial
        
        x_ptrs = X_ptr + batch_idx[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        valid_k2 = offs_k[:, None] < TOTAL_K
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k2 & valid_n
        
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    valid_m_out = offs_m[:, None] < M
    valid_n_out = offs_n[None, :] < OUT_C
    mask_out = valid_m_out & valid_n_out
    
    out_ptrs = Out_ptr + batch_idx[:, None] * stride_on + offs_n[None, :] * stride_oc + out_h_idx[:, None] * stride_oh + out_w_idx[:, None] * stride_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 2}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 4}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_stages=2, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def fused_bias_scale_min_kernel(
    conv_ptr, bias_ptr, out_ptr,
    xnumel, r0_numel,
    scale: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    # xnumel = BATCH * OUT_H * OUT_W (spatial positions)
    # r0_numel = OUT_C (channels to reduce over)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    
    # Load conv output: shape [BATCH, OUT_C, OUT_H, OUT_W] stored as [xnumel, r0_numel]
    # conv_ptr layout: [BATCH, OUT_C, OUT_H*OUT_W] -> index = x * OUT_C + r
    tmp = tl.load(conv_ptr + xindex * r0_numel + r0_index, mask=xmask & r0_mask, other=float('inf'))
    bias = tl.load(bias_ptr + r0_index, mask=r0_mask, other=0.0)
    tmp = (tmp + bias) * scale
    
    tmp = tl.where(xmask & r0_mask, tmp, float('inf'))
    min_val = tl.min(tmp, axis=1)[:, None]
    
    tl.store(out_ptr + xindex, min_val.to(tl.float32), mask=xmask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: [64, 64, 256, 256]
    # weight: [128, 64, 3, 3]
    # bias: [128]
    
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    M = BATCH * OUT_H * OUT_W
    
    # Conv output in layout [BATCH, OUT_C, OUT_H, OUT_W] but stored as [M, OUT_C] for easy reduction
    # Actually store as contiguous [BATCH, OUT_C, OUT_H, OUT_W]
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x, p_conv_weight, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2), p_conv_weight.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
    )
    
    # Now fuse bias + scale + channel min
    # conv_out: [BATCH, OUT_C, OUT_H, OUT_W]
    # We need to reduce over dim=1 (OUT_C)
    # Reshape to [BATCH * OUT_H * OUT_W, OUT_C] - but conv_out is not contiguous in this layout
    # conv_out stride: (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    # For spatial index x = n*OUT_H*OUT_W + h*OUT_W + w, channel c:
    # offset = n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w
    #        = x*OUT_C + ... no, not contiguous
    
    # Need to transpose to [BATCH, OUT_H, OUT_W, OUT_C] for contiguous spatial access
    # Or use a different kernel that handles the strided access
    
    # Let's use a reduction kernel that handles the NCHW layout
    xnumel = BATCH * OUT_H * OUT_W
    r0_numel = OUT_C
    
    out = torch.empty((BATCH, 1, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    # Use a simpler approach: reshape conv_out to access channels
    # conv_out[n, c, h, w] -> linear index: n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w
    # For fixed (n, h, w), varying c: stride is OUT_H*OUT_W
    # spatial index: n*OUT_H*OUT_W + h*OUT_W + w
    # So conv_out[spatial, c] = conv_out_ptr + (n*OUT_C + c)*OUT_H*OUT_W + h*OUT_W + w
    # This is NOT a simple 2D layout
    
    # Better: use a custom kernel with proper strides
    fused_min_kernel_strided[lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)](
        conv_out, p_conv_bias, out,
        xnumel, r0_numel,
        OUT_H, OUT_W, OUT_C,
        2.0,
    )
    
    return out


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_stages=2, num_warps=8),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 128}, num_stages=2, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def fused_min_kernel_strided(
    conv_ptr, bias_ptr, out_ptr,
    xnumel, r0_numel,
    OUT_H, OUT_W, OUT_C,
    scale: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    
    # xindex = n * OUT_H * OUT_W + h * OUT_W + w
    spatial = xindex  # [XBLOCK, 1]
    w_idx = spatial % OUT_W
    tmp2 = spatial // OUT_W
    h_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H
    
    # conv_out[n, c, h, w] = conv_ptr + n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w
    conv_offset = n_idx * (OUT_C * OUT_H * OUT_W) + r0_index * (OUT_H * OUT_W) + h_idx * OUT_W + w_idx
    
    vals = tl.load(conv_ptr + conv_offset, mask=xmask & r0_mask, other=float('inf'))
    bias = tl.load(bias_ptr + r0_index, mask=r0_mask, other=0.0)
    vals = (vals + bias) * scale
    
    vals = tl.where(xmask & r0_mask, vals, float('inf'))
    min_val = tl.min(vals, axis=1)[:, None]
    
    tl.store(out_ptr + xindex, min_val.to(tl.float32), mask=xmask)


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda')
    p_conv_bias = torch.randn(128, device='cuda')
    x = torch.rand(64, 64, 256, 256, device='cuda')
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
        import torch.nn as nn
        p_conv_weight, p_conv_bias, x = inputs
        # Reference
        conv = nn.Conv2d(64, 128, 3, bias=True)
        conv.weight.data = p_conv_weight
        conv.bias.data = p_conv_bias
        conv = conv.cuda()
        with torch.no_grad():
            ref = conv(x)
            ref = ref * 2.0
            ref = torch.min(ref, dim=1, keepdim=True)[0]
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")