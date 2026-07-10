import torch
import triton
import triton.language as tl

# Epilogue kernel: fuse tanh + scale + bias + max_pool2d
# Input: conv output in NCHW format (128, 64, 254, 254)
# Output: max pooled (128, 64, 63, 63)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv_epilogue_pool_kernel(
    X_ptr, W_ptr, bias_ptr, pbias_ptr, out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    POOL_OUT_H: tl.constexpr, POOL_OUT_W: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of output positions (M) x output channels (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index to (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over kernel positions and input channels
    TOTAL_K = IN_C * KH * KW
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K index to (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid = (valid_m[:, None] & valid_k[None, :] & 
                 (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) &
                 (batch_idx[:, None] < BATCH))
        
        # Load input: X is NCHW
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * (IN_C * IN_H * IN_W) +
                  in_c[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Load weight: W is (OUT_C, IN_C, KH, KW)
        valid_n = offs_n < OUT_C
        w_ptrs = (W_ptr +
                  offs_n[None, :] * (IN_C * KH * KW) +
                  in_c[:, None] * (KH * KW) +
                  kh[:, None] * KW +
                  kw[:, None])
        w_mask = (valid_k[:, None] & valid_n[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add conv bias
    valid_n = offs_n < OUT_C
    bias = tl.load(bias_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]
    
    # Tanh
    acc = tl.extra.cuda.libdevice.tanh(acc)
    
    # Scale
    acc = acc * SCALE
    
    # Add per-channel bias (pbias shape: OUT_C)
    pbias = tl.load(pbias_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + pbias[None, :]
    
    # Store result - we need to store to intermediate buffer for max pooling
    # Store in NCHW layout: (batch, out_c, out_h, out_w)
    valid_m = offs_m < M_TOTAL
    mask = valid_m[:, None] & valid_n[None, :]
    
    out_ptrs = (out_ptr +
                batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def max_pool2d_kernel(
    in_ptr, out_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL
    
    # Decode linear index to (batch, c, out_h, out_w)
    out_w_idx = offs % OUT_W
    tmp = offs // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c_idx = tmp2 % C
    batch_idx = tmp2 // C
    
    # Compute max over pool window
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for ph in range(POOL_H):
        for pw in range(POOL_W):
            ih = out_h_idx * POOL_H + ph
            iw = out_w_idx * POOL_W + pw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            ptrs = (in_ptr +
            val = tl.load(ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    out_ptrs = (out_ptr +
    tl.store(out_ptrs, max_val, mask=mask)


def kernel_function(p_conv_weight, p_conv_bias, p_bias, x):
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
    POOL_H = 4
    POOL_W = 4
    POOL_OUT_H = OUT_H // POOL_H
    POOL_OUT_W = OUT_W // POOL_W
    SCALE = 2.0
    
    # Intermediate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    # Flatten pbias
    pbias_flat = p_bias.view(OUT_C)
    
    # Make weight contiguous
    w = p_conv_weight.contiguous()
    x_cont = x.contiguous()
    
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 16
    
    grid = (triton.cdiv(M_TOTAL, BLOCK_M), triton.cdiv(OUT_C, BLOCK_N))
    
    conv_epilogue_pool_kernel[grid](
        x_cont, w, p_conv_bias, pbias_flat, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        POOL_H, POOL_W,
        POOL_OUT_H, POOL_OUT_W,
        SCALE,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Max pooling
    pool_out = torch.empty((BATCH, OUT_C, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=x.dtype)
    N_TOTAL = BATCH * OUT_C * POOL_OUT_H * POOL_OUT_W
    
    BLOCK_SIZE = 256
    grid2 = (triton.cdiv(N_TOTAL, BLOCK_SIZE),)
    
    max_pool2d_kernel[grid2](
        conv_out, pool_out,
        BATCH, OUT_C,
        OUT_H, OUT_W,
        POOL_OUT_H, POOL_OUT_W,
        POOL_H, POOL_W,
        N_TOTAL,
        BLOCK_SIZE,
    )
    
    return pool_out


def get_inputs():
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 256, 256
    kernel_size = 3
    
    p_conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda')
    p_conv_bias = torch.randn(out_channels, device='cuda')
    p_bias = torch.randn(out_channels, 1, 1, device='cuda')
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    
    return [p_conv_weight, p_conv_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    p_conv_weight, p_conv_bias, p_bias, x = inputs
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn.functional as F
        # Reference computation
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        tanh_out = torch.tanh(conv_out)
        scaled = tanh_out * 2.0
        added = scaled + p_bias
        ref = F.max_pool2d(added, kernel_size=4, stride=4)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")