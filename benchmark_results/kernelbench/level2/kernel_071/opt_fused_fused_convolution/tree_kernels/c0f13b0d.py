import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=2, maxnreg=128),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8, maxnreg=128),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4, maxnreg=128),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_div_leaky_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X is in NHWC layout: (BATCH, IN_H, IN_W, IN_C)
    # W is in OHWI layout: (OUT_C, KH, KW, IN_C) -> but we use (OUT_C, KH*KW*IN_C) effectively
    # Actually W in channels-last: (OUT_C, IN_C, KH, KW) reordered to (OUT_C, KH, KW, IN_C)
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    # X strides for NHWC: stride_n=IN_H*IN_W*IN_C, stride_h=IN_W*IN_C, stride_w=IN_C, stride_c=1
    stride_xn = IN_H * IN_W * IN_C
    stride_xh = IN_W * IN_C
    stride_xw = IN_C
    # W strides for OHWI (OUT_C, KH, KW, IN_C): stride_o=KH*KW*IN_C, stride_kh=KW*IN_C, stride_kw=IN_C, stride_i=1
    stride_wo = KH * KW * IN_C
    stride_wkh = KW * IN_C
    stride_wkw = IN_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K
    
    for ijk in range(KH * KW * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        kh_idx = ij // KW
        kw_idx = ij % KW
        
        offs_k = k + tl.arange(0, BLOCK_K)
        # offs_k indexes into IN_C dimension (since we iterate kh, kw in outer loop)
        in_c = offs_k  # within this kh, kw block
        
        ih = out_h * STRIDE_H + kh_idx - PAD_H
        iw = out_w * STRIDE_W + kw_idx - PAD_W
        
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (batch < BATCH)
        valid_k = offs_k < IN_C
        
        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + batch[:, None] * stride_xn + ih[:, None] * stride_xh + iw[:, None] * stride_xw + in_c[None, :]
        x_mask = valid_spatial[:, None] & valid_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W tile [BLOCK_K, BLOCK_N] - W in OHWI layout
        # W[out_c, kh_idx, kw_idx, in_c]
        w_ptrs = W_ptr + offs_n[None, :] * stride_wo + kh_idx * stride_wkh + kw_idx * stride_wkw + offs_k[:, None]
        w_mask = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias, divide, leaky relu
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * (1.0 / DIVISOR)
    acc = tl.where(acc > 0.0, acc, acc * 0.01)
    
    # Store output in NCHW layout: (BATCH, OUT_C, OUT_H, OUT_W)
    # stride_n=OUT_C*OUT_H*OUT_W, stride_c=OUT_H*OUT_W, stride_h=OUT_W, stride_w=1
    stride_yn = OUT_C * OUT_H * OUT_W
    stride_yc = OUT_H * OUT_W
    stride_yh = OUT_W
    
    out_mask = (batch[:, None] < BATCH) & (out_h[:, None] < OUT_H) & (out_w[:, None] < OUT_W) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * stride_yn + offs_n[None, :] * stride_yc + out_h[:, None] * stride_yh + out_w[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)

def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 8, 128, 128) NCHW
    # weight: (64, 8, 3, 3)
    # bias: (64,)
    
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    DIVISOR = 2.0
    
    # Convert X to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    
    # Convert W to OHWI (OUT_C, KH, KW, IN_C) for coalesced access
    # Original: (OUT_C, IN_C, KH, KW) -> permute to (OUT_C, KH, KW, IN_C)
    w_ohwi = p_conv_weight.permute(0, 2, 3, 1).contiguous()
    
    # Output in NCHW
    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    conv2d_div_leaky_relu_kernel[grid](
        x_nhwc, w_ohwi, p_conv_bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        DIVISOR,
    )
    
    return y


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 8, 128, 128, device='cuda', dtype=torch.float32)
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
        import torch.nn.functional as F
        p_conv_weight, p_conv_bias, x = inputs
        # Reference computation
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        ref = conv_out / 2.0
        ref = F.leaky_relu(ref, negative_slope=0.01)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")