import torch
import triton
import triton.language as tl

# Kernel 1: squeeze conv 1x1 (3->6) + ReLU
# M = B*H*W, K=3, N=6
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 8, 'BLOCK_K': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & mask_k[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & mask_k[:, None]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x, w, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if DO_RELU:
        acc = tl.maximum(acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# Kernel 2: expand3x3 conv (6->64, 3x3, pad=1) + ReLU
# Treat as implicit GEMM: M=B*H*W, K=6*3*3=54, N=64
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M_total', 'N_out', 'K_total'],
)
@triton.jit
def conv3x3_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    PAD_H, PAD_W,
    M_total, N_out, K_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K_total, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total
        
        # Decode k -> (in_c, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        in_c = offs_k // (KW * KH)
        
        # Input coordinates
        ih = out_h[:, None] + kh[None, :] - PAD_H
        iw = out_w[:, None] + kw[None, :] - PAD_W
        
        valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_m[:, None] < M_total) & mask_k[None, :]
        
        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
        
        # Weight: [OUT_C, IN_C, KH, KW] -> index [n, in_c, kh, kw]
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KH * KW) + in_c[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        w_mask = (offs_n[None, :] < N_out) & mask_k[:, None]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x, w, allow_tf32=False)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < N_out)
    tl.store(Y_ptr + offs_m[:, None] * N_out + offs_n[None, :], acc, mask=out_mask)


def kernel_function(p_squeeze_weight, p_squeeze_bias, p_expand1x1_weight, p_expand1x1_bias,
    B, C, H, W = x.shape  # 128, 3, 256, 256
    squeeze_out_c = p_squeeze_weight.shape[0]  # 6
    expand1x1_out_c = p_expand1x1_weight.shape[0]  # 64
    expand3x3_out_c = p_expand3x3_weight.shape[0]  # 64
    
    # Step 1: squeeze conv 1x1 (3->6) + ReLU
    # Input x: [B, C, H, W] -> treat as [B*H*W, C] (need NHWC layout)
    # x is NCHW: [128, 3, 256, 256]
    # For 1x1 conv: output[b,oc,h,w] = sum_ic x[b,ic,h,w] * w[oc,ic]
    # Reshape x to [B*H*W, C] by permuting to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C] -> [B*H*W, C]
    M_sq = B * H * W  # 8388608
    K_sq = C  # 3
    N_sq = squeeze_out_c  # 6
    
    # Weight for squeeze: [6, 3, 1, 1] -> [6, 3] -> treat as [N, K], need [K, N] for matmul
    # We want Y[m, n] = sum_k X[m, k] * W[n, k]
    # W stored as [N, K]: stride_wn=K, stride_wk=1
    sq_w = p_squeeze_weight.reshape(N_sq, K_sq)  # [6, 3]
    
    squeeze_out = torch.empty(M_sq, N_sq, device=x.device, dtype=x.dtype)
    
    grid_sq = lambda meta: (triton.cdiv(M_sq, meta['BLOCK_M']), triton.cdiv(N_sq, meta['BLOCK_N']))
    conv1x1_relu_kernel[grid_sq](
        x_nhwc, sq_w, p_squeeze_bias, squeeze_out,
        M_sq, N_sq, K_sq,
        K_sq, 1,   # stride_wn, stride_wk (W is [N, K], so stride_wn=K, stride_wk=1)
    )
    # squeeze_out: [B*H*W, 6] in NHWC order
    
    # Step 2: expand1x1 conv (6->64) + ReLU
    M_e1 = B * H * W
    K_e1 = squeeze_out_c  # 6
    N_e1 = expand1x1_out_c  # 64
    
    e1_w = p_expand1x1_weight.reshape(N_e1, K_e1)  # [64, 6]
    expand1x1_out = torch.empty(M_e1, N_e1, device=x.device, dtype=x.dtype)
    
    grid_e1 = lambda meta: (triton.cdiv(M_e1, meta['BLOCK_M']), triton.cdiv(N_e1, meta['BLOCK_N']))
    conv1x1_relu_kernel[grid_e1](
        squeeze_out, e1_w, p_expand1x1_bias, expand1x1_out,
        M_e1, N_e1, K_e1,
        K_e1, 1,
        K_e1, 1,
        True,
    )
    # expand1x1_out: [B*H*W, 64]
    
    # Step 3: expand3x3 conv (6->64, 3x3, pad=1) + ReLU
    # squeeze_out is [B*H*W, 6] in NHWC, need to reshape to [B, H, W, 6] for 3x3 conv
    # Actually for 3x3 conv we need NCHW access pattern
    # Let's reshape squeeze_out to [B, H, W, 6] and use it as NHWC
    # But our conv3x3 kernel expects NCHW input
    # squeeze_out is [B*H*W, 6] = NHWC flattened
    # We need to convert back to NCHW for the 3x3 conv
    squeeze_nchw = squeeze_out.reshape(B, H, W, squeeze_out_c).permute(0, 3, 1, 2).contiguous()
    # squeeze_nchw: [128, 6, 256, 256]
    
    M_e3 = B * H * W
    K_e3 = squeeze_out_c * 3 * 3  # 54
    N_e3 = expand3x3_out_c  # 64
    
    expand3x3_out = torch.empty(M_e3, N_e3, device=x.device, dtype=x.dtype)
    
    grid_e3 = lambda meta: (triton.cdiv(M_e3, meta['BLOCK_M']), triton.cdiv(N_e3, meta['BLOCK_N']))
    conv3x3_relu_kernel[grid_e3](
        squeeze_nchw, p_expand3x3_weight, p_expand3x3_bias, expand3x3_out,
        B, squeeze_out_c, H, W,
        expand3x3_out_c, H, W,
        3, 3,
        1, 1,
        M_e3, N_e3, K_e3,
    )
    # expand3x3_out: [B*H*W, 64]
    
    # Step 4: cat along channel dim
    # expand1x1_out: [B*H*W, 64] in NHWC order
    # expand3x3_out: [B*H*W, 64] in NHWC order
    # cat -> [B*H*W, 128] in NHWC order
    # Then reshape to [B, H, W, 128] and permute to [B, 128, H, W]
    cat_out = torch.cat([expand1x1_out, expand3x3_out], dim=1)  # [B*H*W, 128]
    result = cat_out.reshape(B, H, W, expand1x1_out_c + expand3x3_out_c).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    p_squeeze_weight = torch.randn(6, 3, 1, 1)
    p_squeeze_bias = torch.randn(6)
    p_expand1x1_weight = torch.randn(64, 6, 1, 1)
    p_expand1x1_bias = torch.randn(64)
    p_expand3x3_weight = torch.randn(64, 6, 3, 3)
    p_expand3x3_bias = torch.randn(64)
    x = torch.randn(128, 3, 256, 256)
    return [p_squeeze_weight, p_squeeze_bias, p_expand1x1_weight, p_expand1x1_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]
    
    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        p_squeeze_weight, p_squeeze_bias, p_expand1x1_weight, p_expand1x1_bias, \
            p_expand3x3_weight, p_expand3x3_bias, x = inputs_cuda
        
        # Reference computation
        squeeze_ref = torch.nn.functional.conv2d(x, p_squeeze_weight, p_squeeze_bias)
        squeeze_ref = torch.relu(squeeze_ref)
        expand1x1_ref = torch.nn.functional.conv2d(squeeze_ref, p_expand1x1_weight, p_expand1x1_bias)
        expand1x1_ref = torch.relu(expand1x1_ref)
        expand3x3_ref = torch.nn.functional.conv2d(squeeze_ref, p_expand3x3_weight, p_expand3x3_bias, padding=1)
        expand3x3_ref = torch.relu(expand3x3_ref)
        ref = torch.cat([expand1x1_ref, expand3x3_ref], dim=1)
        
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")