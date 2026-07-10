import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_relu_bias_kernel(
    X_ptr, W_ptr, conv_bias_ptr, bias_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides (NCHW)
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # Weight strides (OIHW)
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # Output strides (NCHW)
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W (output spatial positions)
    # N = OUT_C (output channels)
    # K = IN_C * KH * KW (reduction dimension)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KH * KW
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K index -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        in_c_idx = tmp_k // KH
        
        # Compute input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial
        
        x_ptrs = (X_ptr 
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load weight tile [BLOCK_K, BLOCK_N]
        valid_k_w = offs_k[:, None] < TOTAL_K
        valid_n_w = offs_n[None, :] < OUT_C
        mask_w = valid_k_w & valid_n_w
        
        w_ptrs = (W_ptr 
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Epilogue: add conv bias, relu, add extra bias
    valid_m_out = offs_m[:, None] < M_TOTAL
    valid_n_out = offs_n[None, :] < OUT_C
    mask_out = valid_m_out & valid_n_out
    
    # Load conv bias (shape: OUT_C)
    conv_bias = tl.load(conv_bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + conv_bias[None, :]
    
    # ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Load extra bias (shape: OUT_C, broadcast over spatial)
    extra_bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + extra_bias[None, :]
    
    # Store output in NCHW layout
    # Output index: batch_idx * s_yn + offs_n * s_yc + out_h_idx * s_yh + out_w_idx * s_yw
    y_ptrs = (Y_ptr 
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, p_bias, x):
    # x: (128, 64, 128, 128) NCHW
    # p_conv_weight: (128, 64, 3, 3) OIHW
    # p_conv_bias: (128,)
    # p_bias: (128, 1, 1)
    
    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_H = x.shape[2]
    IN_W = x.shape[3]
    
    OUT_C = p_conv_weight.shape[0]
    KH = p_conv_weight.shape[2]
    KW = p_conv_weight.shape[3]
    
    STRIDE_H = 1
    STRIDE_W = 1
    PAD_H = 0
    PAD_W = 0
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    # Ensure contiguous
    x = x.contiguous()
    w = p_conv_weight.contiguous()
    conv_bias = p_conv_bias.contiguous()
    # p_bias is (128, 1, 1), flatten to (128,)
    bias = p_bias.reshape(OUT_C).contiguous()
    
    # Output tensor in NCHW
    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    # Input strides (NCHW)
    s_xn = x.stride(0)
    s_xc = x.stride(1)
    s_xh = x.stride(2)
    s_xw = x.stride(3)
    
    # Weight strides (OIHW)
    s_wo = w.stride(0)
    s_wi = w.stride(1)
    s_wkh = w.stride(2)
    s_wkw = w.stride(3)
    
    # Output strides (NCHW)
    s_yn = y.stride(0)
    s_yc = y.stride(1)
    s_yh = y.stride(2)
    s_yw = y.stride(3)
    
    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
    
    conv2d_relu_bias_kernel[grid](
        x, w, conv_bias, bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wo, s_wi, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )
    
    return y


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(128, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 64, 128, 128, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    if not args.profile_only:
        # Verify correctness
        p_conv_weight, p_conv_bias, p_bias, x = inputs
        
        # Reference computation
        import torch.nn.functional as F
        ref_out = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        ref_out = torch.relu(ref_out)
        ref_out = ref_out + p_bias
        
        # Triton computation
        triton_out = kernel_function(p_conv_weight, p_conv_bias, p_bias, x)
        
        max_diff = (triton_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Correctness check passed!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")