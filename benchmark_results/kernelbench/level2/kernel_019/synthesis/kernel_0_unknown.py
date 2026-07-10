import torch
import triton
import triton.language as tl
import math

# ConvTranspose2d kernel: output[n, oc, oh, ow] = sum over ic, kh, kw of
#   weight[ic, oc, kh, kw] * input[n, ic, oh - kh, ow - kw]
# where oh - kh must be in [0, IH) and ow - kw in [0, IW)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M: n, oh, ow
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    n  = offs_m // (OW * OH)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IC * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: ic, kh, kw
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ic_idx = offs_k // (KW * KH)
        
        # For conv_transpose: output[n, oc, oh, ow] += input[n, ic, oh-kh, ow-kw]
        # ih = oh - kh, iw = ow - kw
        ih = oh[:, None] - kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] - kw_idx[None, :]  # [BLOCK_M, BLOCK_K]
        
        valid_x = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & \
                  (offs_k[None, :] < K) & (n[:, None] < N)
        
        x_ptrs = X_ptr + n[:, None] * stride_xn + ic_idx[None, :] * stride_xc + \
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # Weight: [IC, OC, KH, KW] -> for K=(ic,kh,kw), N=oc
        # w[ic, oc, kh, kw]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OC)
        w_ptrs = W_ptr + ic_idx[:, None] * stride_wic + offs_n[None, :] * stride_woc + \
                 kh_idx[:, None] * stride_wkh + kw_idx[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias[None, :]
    
    # Store
    mask_m = (offs_m < M) & (n < N)
    mask_n = offs_n < OC
    y_ptrs = Y_ptr + n[:, None] * stride_yn + offs_n[None, :] * stride_yc + \
             oh[:, None] * stride_yh + ow[:, None] * stride_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['group_size'],
)
@triton.jit
def gelu_groupnorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C, H, W_dim,
    num_groups, group_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    
    # group_size = C // num_groups, spatial = H * W_dim
    # Elements in this group: group_size * H * W_dim
    spatial = H * W_dim
    total = group_size * spatial
    
    # Compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    c_start = g * group_size
    
    for off in range(0, total, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        # offs encodes (c_local, h, w) flattened
        c_local = offs // spatial
        hw = offs % spatial
        h_idx = hw // W_dim
        w_idx = hw % W_dim
        c_idx = c_start + c_local
        
        mask = (offs < total) & (c_idx < C)
        
        # Load and apply GELU
        x_ptrs = X_ptr + n * (C * H * W_dim) + c_idx * (H * W_dim) + h_idx * W_dim + w_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x3)
        gelu_x = x * 0.5 * (1.0 + tl.extra.libdevice.tanh(inner))
        
        mean += tl.sum(tl.where(mask, gelu_x, 0.0), axis=0)
        var  += tl.sum(tl.where(mask, gelu_x * gelu_x, 0.0), axis=0)
    
    mean = mean / total
    var = var / total - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply scale/bias
    for off in range(0, total, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        c_local = offs // spatial
        hw = offs % spatial
        h_idx = hw // W_dim
        w_idx = hw % W_dim
        c_idx = c_start + c_local
        
        mask = (offs < total) & (c_idx < C)
        
        x_ptrs = X_ptr + n * (C * H * W_dim) + c_idx * (H * W_dim) + h_idx * W_dim + w_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # GELU again
        x3 = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x3)
        gelu_x = x * 0.5 * (1.0 + tl.extra.libdevice.tanh(inner))
        
        # Normalize
        gn_w = tl.load(W_ptr + c_idx, mask=mask & (c_idx < C), other=1.0).to(tl.float32)
        gn_b = tl.load(B_ptr + c_idx, mask=mask & (c_idx < C), other=0.0).to(tl.float32)
        
        out = (gelu_x - mean) * inv_std * gn_w + gn_b
        
        y_ptrs = Y_ptr + n * (C * H * W_dim) + c_idx * (H * W_dim) + h_idx * W_dim + w_idx
        tl.store(y_ptrs, out.to(tl.float32), mask=mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias,
    # x: (N, IC, IH, IW)
    # weight: (IC, OC, KH, KW) for ConvTranspose2d
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = p_conv_transpose_weight.shape
    
    stride_val = 1
    pad = 0
    OH = (IH - 1) * stride_val - 2 * pad + KH  # 258
    OW = (IW - 1) * stride_val - 2 * pad + KW  # 258
    
    # Allocate output for conv_transpose
    conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    M = N * OH * OW
    N_out = OC
    K = IC * KH * KW
    
    grid_conv = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    
    conv_transpose2d_kernel[grid_conv](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        M, N_out, K,
    )
    
    # Fused GELU + GroupNorm
    num_groups = 8
    group_size = OC // num_groups
    eps = 1e-5
    
    final_out = torch.empty_like(conv_out)
    
    grid_gn = (N * num_groups,)
    
    gelu_groupnorm_kernel[grid_gn](
        conv_out, p_group_norm_weight, p_group_norm_bias, final_out,
        N, OC, OH, OW,
        num_groups, group_size,
        eps,
    )
    
    return final_out


def get_inputs():
    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(64, 64, 3, 3, dtype=torch.float32, device='cuda')
    p_conv_transpose_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    p_group_norm_weight = torch.randn(64, dtype=torch.float32, device='cuda')
    p_group_norm_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    x = torch.rand(128, 64, 256, 256, dtype=torch.float32, device='cuda')
    return [p_conv_transpose_weight, p_conv_transpose_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    if not args.profile_only:
        # Verify correctness
        import torch.nn as nn
        weight, bias, gn_w, gn_b, x = inputs
        
        # Reference
        conv_t = nn.ConvTranspose2d(64, 64, 3, stride=1, bias=True).cuda()
        conv_t.weight.data = weight
        conv_t.bias.data = bias
        gn = nn.GroupNorm(8, 64).cuda()
        gn.weight.data = gn_w
        gn.bias.data = gn_b
        
        with torch.no_grad():
            ref = gn(torch.nn.functional.gelu(conv_t(x)))
        
        out = kernel_function(*inputs)
        
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("PASSED")
        else:
            print("FAILED")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")