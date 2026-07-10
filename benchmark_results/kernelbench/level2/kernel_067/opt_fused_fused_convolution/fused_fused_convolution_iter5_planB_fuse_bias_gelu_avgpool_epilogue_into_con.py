import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Conv kernel: implicit GEMM with NHWC layout

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8, maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8, maxnreg=128),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4, maxnreg=128),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_nhwc_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_on, stride_oh, stride_ow, stride_oc,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # stride_on  = Bias_ptr (pointer value passed as int64)
    # stride_oh  = N_SPATIAL (integer)
    # stride_ow  = unused
    # stride_oc  = unused
    # Out_ptr    = fused output [BATCH, OUT_C] (zero-initialized)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C

    # Precompute M-side base pointers outside all loops
    base_x_m = X_ptr + batch * stride_xn  # shape [BLOCK_M]

    for kh_idx in range(KERNEL_H):   # dynamic range — NO unrolling
        for kw_idx in range(KERNEL_W):  # dynamic range — NO unrolling
            # Hoist spatial offset outside in_c loop
            ih = out_h + kh_idx   # shape [BLOCK_M], scalar addition
            iw = out_w + kw_idx   # shape [BLOCK_M], scalar addition
            valid_spatial = valid_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            # Spatial pointer base for this (kh, kw)
            x_spatial = base_x_m + ih * stride_xh + iw * stride_xw  # [BLOCK_M]
            # Weight pointer base for this (kh, kw)
            w_spatial = W_ptr + offs_n * stride_wco + kh_idx * stride_wkh + kw_idx * stride_wkw  # [BLOCK_N]
            for ic_off in range(0, IN_C, BLOCK_K):  # innermost: contiguous in_c
                offs_ic = ic_off + tl.arange(0, BLOCK_K)
                valid_ic = offs_ic < IN_C
                # x_ptrs: [BLOCK_M, BLOCK_K] stride_xc=1 in NHWC, fully coalesced
                x_ptrs = x_spatial[:, None] + offs_ic[None, :] * stride_xc
                valid_x = valid_spatial[:, None] & valid_ic[None, :]
                x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
                # w_ptrs: [BLOCK_K, BLOCK_N] contiguous in_c across BLOCK_K
                w_ptrs = w_spatial[None, :] + offs_ic[:, None] * stride_wci
                valid_w = valid_ic[:, None] & valid_n[None, :]
                w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
                acc = tl.dot(x_tile, w_tile, acc=acc)

    # Fused epilogue: bias + GELU + partial avgpool via atomic_add
    # Reconstruct Bias_ptr from stride_on (passed as int64 pointer value)
    # stride_oh holds N_SPATIAL
    N_SPATIAL = stride_oh
    Bias_ptr = Out_ptr + stride_on  # stride_on = bias_offset_from_out_ptr

    # Load bias: [BLOCK_N]
    bias = tl.load(Bias_ptr + offs_n, mask=valid_n, other=0.0)

    # Add bias and apply GELU: [BLOCK_M, BLOCK_N]
    acc_biased = acc + bias[None, :]
    acc_half = acc_biased * 0.5
    acc_erf = libdevice.erf(acc_biased * 0.7071067811865476)
    gelu_out = acc_half * (1.0 + acc_erf)

    # Zero out invalid entries
    gelu_out = tl.where(valid_m[:, None] & valid_n[None, :], gelu_out, 0.0)

    # Accumulate into [BATCH, OUT_C] output via per-row atomic_add
    # Each row m contributes gelu_out[m, :] / N_SPATIAL to Out_ptr[batch[m], :]
    n_spatial_f = N_SPATIAL.to(tl.float32)
    for m_idx in tl.static_range(0, BLOCK_M):
        b = batch[m_idx]
        row_valid = valid_m[m_idx]
        row_vals = gelu_out[m_idx, :] / n_spatial_f  # [BLOCK_N]
        tl.atomic_add(Out_ptr + b * OUT_C + offs_n, row_vals, mask=valid_n & row_valid)

def gelu_avgpool_kernel(
    Conv_ptr, Bias_ptr, Out_ptr,
    BATCH, OUT_C, N_SPATIAL,
    stride_cn, stride_cc, stride_ch, stride_cw,
    OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch = pid // OUT_C
    channel = pid % OUT_C
    
    bias = tl.load(Bias_ptr + channel)
    
    acc = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, N_SPATIAL, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_SPATIAL
        
        oh = offs // OUT_W
        ow = offs % OUT_W
        
        ptrs = Conv_ptr + batch * stride_cn + channel * stride_cc + oh * stride_ch + ow * stride_cw
        val = tl.load(ptrs, mask=mask, other=0.0)
        val = val + bias
        
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        val_half = val * 0.5
        val_erf = libdevice.erf(val * 0.7071067811865476)
        gelu_val = val_half * (1.0 + val_erf)
        
        gelu_val = tl.where(mask, gelu_val, 0.0)
        acc = acc + tl.sum(gelu_val, axis=0)
    
    result = acc / N_SPATIAL
    tl.store(Out_ptr + pid, result)


def kernel_function(p_conv_weight, p_conv_bias, x):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = p_conv_weight.shape
    OUT_H = IN_H - KERNEL_H + 1
    OUT_W = IN_W - KERNEL_W + 1
    
    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to NHWC-friendly: [OUT_C, IN_C, KH, KW] -> keep as is but use strides
    w = p_conv_weight.contiguous()
    
    # Allocate conv output in NCHW
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    # Strides for NHWC input
    stride_xn = x_nhwc.stride(0)
    stride_xh = x_nhwc.stride(1)
    stride_xw = x_nhwc.stride(2)
    stride_xc = x_nhwc.stride(3)
    
    # Strides for weight [OUT_C, IN_C, KH, KW]
    stride_wco = w.stride(0)
    stride_wci = w.stride(1)
    stride_wkh = w.stride(2)
    stride_wkw = w.stride(3)
    
    # Strides for output [BATCH, OUT_C, OUT_H, OUT_W]
    stride_on = conv_out.stride(0)
    stride_oc = conv_out.stride(1)
    stride_oh = conv_out.stride(2)
    stride_ow = conv_out.stride(3)
    
    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
    
    conv2d_nhwc_kernel[grid](
        x_nhwc, w, conv_out,
        BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride_xn, stride_xh, stride_xw, stride_xc,
        stride_wco, stride_wci, stride_wkh, stride_wkw,
        stride_on, stride_oh, stride_ow, stride_oc,
        M_TOTAL,
    )
    
    # GELU + global average pooling
    N_SPATIAL = OUT_H * OUT_W
    out = torch.empty((BATCH, OUT_C), device=x.device, dtype=x.dtype)
    
    total_programs = BATCH * OUT_C
    
    gelu_avgpool_kernel[(total_programs,)](
        conv_out, p_conv_bias, out,
        BATCH, OUT_C, N_SPATIAL,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        OUT_H, OUT_W,
    )
    
    return out


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, device='cuda')
    p_conv_bias = torch.randn(64, device='cuda')
    x = torch.rand(128, 8, 256, 256, device='cuda')
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
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias)
        gelu_out = F.gelu(conv_out)
        pool_out = F.adaptive_avg_pool2d(gelu_out, 1)
        ref = pool_out.squeeze(-1).squeeze(-1)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")