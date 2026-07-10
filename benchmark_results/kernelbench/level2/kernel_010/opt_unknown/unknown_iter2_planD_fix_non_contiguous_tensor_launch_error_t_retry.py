import torch
import triton
import triton.language as tl

# Fused kernel: MaxPool2d(2,2) + Hardtanh(-1,1) + Mean(H,W) + Tanh
# Input: conv_out [B, C, H, W] (H=W=256), output: [B, C, 1, 1]
# Pool output: [B, C, H//2, W//2] = [B, C, 128, 128]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['pool_h', 'pool_w'],
)
@triton.jit
def fused_pool_hardtanh_mean_tanh_kernel(
    conv_ptr,
    out_ptr,
    B, C,
    in_h, in_w,
    pool_h, pool_w,
    pool_k: tl.constexpr,
    pool_s: tl.constexpr,
    ht_min: tl.constexpr,
    ht_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    bc = tl.program_id(0)
    b = bc // C
    c = bc % C

    # Base pointer for this (b, c) in conv output
    base = b * (C * in_h * in_w) + c * (in_h * in_w)

    total_pool = pool_h * pool_w
    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over pooled positions in blocks
    for start in tl.range(0, total_pool, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_pool

        # Decode pool position
        ph = offs // pool_w
        pw = offs % pool_w

        # For each pool position, compute max over 2x2 window
        # ih_base = ph * pool_s, iw_base = pw * pool_s
        ih0 = ph * pool_s
        iw0 = pw * pool_s

        # Load 4 values (2x2 pool)
        ih0_ = ih0
        iw0_ = iw0
        ih1_ = ih0
        iw1_ = iw0 + 1
        ih2_ = ih0 + 1
        iw2_ = iw0
        ih3_ = ih0 + 1
        iw3_ = iw0 + 1

        v0 = tl.load(conv_ptr + base + ih0_ * in_w + iw0_, mask=mask & (ih0_ < in_h) & (iw0_ < in_w), other=-1e9)
        v1 = tl.load(conv_ptr + base + ih1_ * in_w + iw1_, mask=mask & (ih1_ < in_h) & (iw1_ < in_w), other=-1e9)
        v2 = tl.load(conv_ptr + base + ih2_ * in_w + iw2_, mask=mask & (ih2_ < in_h) & (iw2_ < in_w), other=-1e9)
        v3 = tl.load(conv_ptr + base + ih3_ * in_w + iw3_, mask=mask & (ih3_ < in_h) & (iw3_ < in_w), other=-1e9)

        pool_val = tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3))

        # Hardtanh
        ht_val = tl.minimum(tl.maximum(pool_val, ht_min), ht_max)

        # Accumulate (mask out invalid)
        acc = acc + tl.sum(tl.where(mask, ht_val, 0.0), axis=0)

    # Mean
    mean_val = acc / (pool_h * pool_w)

    # Tanh
    tanh_val = tl.extra.libdevice.tanh(mean_val)

    # Store
    out_idx = b * C + c
    tl.store(out_ptr + out_idx, tanh_val)


# ConvTranspose2d kernel using implicit GEMM
# weight shape: [in_channels, out_channels, kH, kW] (PyTorch ConvTranspose2d weight)
# For conv_transpose2d: output[b,oc,oh,ow] = sum_{ic,kh,kw} x[b,ic,ih,iw] * w[ic,oc,kh,kw]
# where ih = oh - kh + padding, iw = ow - kw + padding (for stride=1)
# Equivalently: for each output (oh,ow), sum over (ic,kh,kw) where ih=oh+pad-kh, iw=ow+pad-kw is valid

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, OC,
    IN_H, IN_W,
    OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = B * OUT_H * OUT_W, N = OC, K = IC * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (b, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    b  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (ic, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        ic = offs_k // (KH * KW)

        # For conv_transpose2d: input position
        # oh = ih + kh - pad => ih = oh - kh + pad
        ih = oh[:, None] - kh[None, :] + PAD_H
        iw = ow[:, None] - kw[None, :] + PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = x_ptr + b[:, None] * (IC * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: [IC, OC, KH, KW] -> index [ic, oc, kh, kw]
        # w_ptrs shape: [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + ic[:, None] * (OC * KH * KW) + offs_n[None, :] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = out_ptr + b[:, None] * (OC * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, x):
    B, IC, IN_H, IN_W = x.shape
    # weight for ConvTranspose2d: [IC, OC, KH, KW]
    IC_w, OC, KH, KW = p_conv_transpose_weight.shape
    PAD_H, PAD_W = 1, 1
    STRIDE_H, STRIDE_W = 1, 1

    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW

    M = B * OUT_H * OUT_W
    N = OC
    K = IC * KH * KW

    conv_out = torch.empty(B, OC, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    grid_conv = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv_transpose2d_kernel[grid_conv](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        B, IC, OC,
        IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        PAD_H, PAD_W,
        STRIDE_H, STRIDE_W,
        M, N, K,
    )

    # Fused: MaxPool2d(2,2) + Hardtanh(-1,1) + Mean(H,W) + Tanh
    pool_k = 2
    pool_s = 2
    pool_h = OUT_H // pool_s
    pool_w = OUT_W // pool_s

    out = torch.empty(B, OC, device=x.device, dtype=x.dtype)

    grid_fused = (B * OC,)

    fused_pool_hardtanh_mean_tanh_kernel[grid_fused](
        conv_out, out,
        B, OC,
        OUT_H, OUT_W,
        pool_h, pool_w,
        pool_k, pool_s,
        -1.0, 1.0,
    )

    return out.view(B, OC, 1, 1)


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 64, 3, 3)
    p_conv_transpose_bias = torch.randn(64)
    x = torch.rand(128, 64, 256, 256)
    return [p_conv_transpose_weight, p_conv_transpose_bias, x]


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
        w, b, x = inputs
        w_cuda = w.cuda()
        b_cuda = b.cuda()
        x_cuda = x.cuda()

        # Reference
        conv_t = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=True)
        conv_t.weight.data = w_cuda
        conv_t.bias.data = b_cuda
        conv_t = conv_t.cuda()

        with torch.no_grad():
            ref = conv_t(x_cuda)
            ref = torch.nn.functional.max_pool2d(ref, kernel_size=2, stride=2)
            ref = torch.clamp(ref, -1.0, 1.0)
            ref = torch.mean(ref, dim=(2, 3), keepdim=True)
            ref = torch.tanh(ref)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")