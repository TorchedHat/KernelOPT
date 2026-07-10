import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Conv kernel: implicit GEMM with NHWC layout

@triton.autotune(
    configs=[
        # Small tiles: low register pressure, high occupancy
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        # Medium tiles: balance between tile efficiency and occupancy
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        # Original large tiles retained for completeness
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_nhwc_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_on, stride_oh, stride_ow, stride_oc,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Parameter repurposing for fused epilogue:
    #   Out_ptr      -> PartialSum_ptr [BATCH, OUT_C], zero-initialized float32
    #   stride_on    -> Bias_ptr (integer address of bias tensor [OUT_C])
    #   stride_oh    -> N_SPATIAL (= OUT_H * OUT_W, integer)
    #   stride_ow, stride_oc are unused in fused mode

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] + kh[None, :]  # no padding, stride=1
        iw = out_w[:, None] + kw[None, :]

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_x = valid_m[:, None] & valid_k[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)

        x_ptrs = X_ptr + batch[:, None] * stride_xn + ih * stride_xh + iw * stride_xw + in_c[None, :] * stride_xc
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Fused epilogue: bias + GELU + partial sum accumulation for average pooling
    # stride_on holds the integer address of the bias tensor
    # stride_oh holds N_SPATIAL
    Bias_ptr = stride_on.to(tl.pointer_type(tl.float32), bitcast=True)
    N_SPATIAL = stride_oh

    # Load bias for this tile's output channels
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)

    # Add bias: acc [BLOCK_M, BLOCK_N], bias [BLOCK_N]
    acc_biased = acc + bias[None, :]

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_vals = (acc_biased * 0.5) * (1.0 + libdevice.erf(acc_biased * 0.7071067811865476))

    # Mask out invalid entries (out-of-bounds M or N)
    valid_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    gelu_vals = tl.where(valid_mask, gelu_vals, 0.0)

    # Accumulate partial sums into Out_ptr [BATCH, OUT_C] via atomic_add
    # Handle tiles that may span multiple batch indices
    batch_first = batch[0]
    batch_last = batch[BLOCK_M - 1]

    if batch_first == batch_last:
        # All rows belong to the same batch — simple reduction
        partial = tl.sum(gelu_vals, axis=0)  # [BLOCK_N]
        out_ptrs = Out_ptr + batch_first * OUT_C + offs_n
        tl.atomic_add(out_ptrs, partial, mask=(offs_n < OUT_C))
    else:
        # Tile spans multiple batches — accumulate per batch
        for b in tl.range(batch_first, batch_last + 1):
            batch_mask = (batch == b)  # [BLOCK_M]
            b_gelu = tl.where(batch_mask[:, None] & (offs_n[None, :] < OUT_C), gelu_vals, 0.0)
            partial_b = tl.sum(b_gelu, axis=0)  # [BLOCK_N]
            out_ptrs_b = Out_ptr + b * OUT_C + offs_n
            tl.atomic_add(out_ptrs_b, partial_b, mask=(offs_n < OUT_C))


@triton.jit

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