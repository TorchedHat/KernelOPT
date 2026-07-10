import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv_epilogue_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    stride_on: tl.constexpr, stride_oc: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K
    
    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (kw, kh, in_c): kw fastest-changing to match weight layout (OUT_C, IN_C, KH, KW)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        in_c = offs_k // (KH * KW)
        
        # Input coordinates
        ih = out_h[:, None] + kh[None, :]  # no padding, stride=1
        iw = out_w[:, None] + kw[None, :]
        
        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid = valid_m[:, None] & valid_k[None, :] & (batch[:, None] < BATCH)
        
        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        # KEY CHANGE: fresh zero internal acc per sub-tile so TF32 error is bounded to BLOCK_K elements.
        # No acc= argument means tl.dot initializes its internal accumulator to zero.
        # The FP32 partial is then added to the running acc in full FP32 outside the dot.
        partial = tl.dot(x_tile, w_tile, allow_tf32=True)
        acc = acc + partial
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    # sub 0.5, tanh, sub 0.2
    acc = acc - SUB1
    acc = libdevice.tanh(acc)
    acc = acc - SUB2
    
    # Store output: shape (BATCH, OUT_C, OUT_H, OUT_W)
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_ptrs = OUT_ptr + batch[:, None] * stride_on + offs_n[None, :] * stride_oc + out_h[:, None] * stride_oh + out_w[:, None] * stride_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['N_TOTAL'],
)
@triton.jit

def avgpool2d_kernel(
    IN_ptr, OUT_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_in_n: tl.constexpr, stride_in_c: tl.constexpr,
    stride_in_h: tl.constexpr, stride_in_w: tl.constexpr,
    stride_out_n: tl.constexpr, stride_out_c: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_w: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL
    
    # Decode flat index -> (batch, c, out_h, out_w)
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    batch = tmp2 // C
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * KH + kh
            iw = out_w * KW + kw
            in_ptrs = IN_ptr + batch * stride_in_n + c * stride_in_c + ih * stride_in_h + iw * stride_in_w
            val = tl.load(in_ptrs, mask=mask & (batch < BATCH) & (c < C) & (ih < IN_H) & (iw < IN_W), other=0.0)
            acc = acc + val
    
    acc = acc / (KH * KW)
    
    out_ptrs = OUT_ptr + batch * stride_out_n + c * stride_out_c + out_h * stride_out_h + out_w * stride_out_w
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 64, 128, 128) NCHW
    # weight: (128, 64, 3, 3)
    # bias: (128,)
    
    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_H = x.shape[2]
    IN_W = x.shape[3]
    OUT_C = p_conv_weight.shape[0]
    KH = p_conv_weight.shape[2]
    KW = p_conv_weight.shape[3]
    OUT_H = IN_H - KH + 1  # no padding, stride=1
    OUT_W = IN_W - KW + 1
    
    # Intermediate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    def grid_conv(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
    
    conv_epilogue_kernel[grid_conv](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2), p_conv_weight.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        0.5, 0.2,
    )
    
    # AvgPool2d with kernel_size=2, stride=2
    POOL_KH = 2
    POOL_KW = 2
    POOL_OUT_H = OUT_H // POOL_KH
    POOL_OUT_W = OUT_W // POOL_KW
    
    pool_out = torch.empty((BATCH, OUT_C, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=x.dtype)
    
    N_TOTAL = BATCH * OUT_C * POOL_OUT_H * POOL_OUT_W
    
    def grid_pool(meta):
        return (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    
    avgpool2d_kernel[grid_pool](
        conv_out, pool_out,
        BATCH, OUT_C,
        OUT_H, OUT_W,
        POOL_OUT_H, POOL_OUT_W,
        POOL_KH, POOL_KW,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        pool_out.stride(0), pool_out.stride(1), pool_out.stride(2), pool_out.stride(3),
        N_TOTAL,
    )
    
    return pool_out


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 64, 128, 128, device='cuda', dtype=torch.float32)
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
        
        # Reference computation
        conv = nn.Conv2d(64, 128, 3, bias=True)
        conv.weight.data = p_conv_weight
        conv.bias.data = p_conv_bias
        conv = conv.cuda()
        avgpool = nn.AvgPool2d(2)
        
        with torch.no_grad():
            ref = conv(x)
            ref = ref - 0.5
            ref = torch.tanh(ref)
            ref = ref - 0.2
            ref = avgpool(ref)
        
        print(f"Output shape: {output.shape}, ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.6f}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")