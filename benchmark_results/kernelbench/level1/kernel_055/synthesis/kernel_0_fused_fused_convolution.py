import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Input X is in NHWC layout: (N, H, W, C)
    # Weight W is in (OUT_C, KERNEL_H, KERNEL_W, IN_C) layout
    # Output Y is in NHWC layout: (N, OUT_H, OUT_W, OUT_C)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # M dimension: N * OUT_H * OUT_W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # K dimension: KERNEL_H * KERNEL_W * IN_C
    TOTAL_K = KERNEL_H * KERNEL_W * IN_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        # Valid mask
        valid = (
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )
        
        # Load input tile [BLOCK_M, BLOCK_K] from NHWC layout
        # X stride: (IN_H*IN_W*IN_C, IN_W*IN_C, IN_C, 1)
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_H * IN_W * IN_C) +
            in_c_idx[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Load weight tile [BLOCK_K, BLOCK_N] from (OUT_C, KH, KW, IN_C) layout
        # W stride: (KH*KW*IN_C, KW*IN_C, IN_C, 1)
        w_ptrs = W_ptr + (
            offs_n[None, :] * (KERNEL_H * KERNEL_W * IN_C) +
            kh_idx[:, None] * (KERNEL_W * IN_C) +
            kw_idx[:, None] * IN_C +
            in_c_idx[:, None]
        )
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output in NHWC layout: (N, OUT_H, OUT_W, OUT_C)
    out_mask = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_H * OUT_W * OUT_C) +
        out_h_idx[:, None] * (OUT_W * OUT_C) +
        out_w_idx[:, None] * OUT_C +
        offs_n[None, :]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(weight, x):
    # weight: (OUT_C, IN_C, KH, KW) = (128, 64, 3, 3)
    # x: (N, IN_C, H, W) = (8, 64, 512, 1024)
    
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    
    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_khwc = weight.permute(0, 2, 3, 1).contiguous()
    
    # Output in NHWC layout
    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_khwc, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
    )
    
    # Convert back to NCHW
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 64, 512, 1024, device='cuda', dtype=torch.float32)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    if not args.profile_only:
        # Verify correctness
        weight, x = inputs
        
        # Reference using PyTorch conv2d
        import torch.nn.functional as F
        ref_out = F.conv2d(x, weight, bias=None, stride=1, padding=0)
        
        our_out = kernel_function(weight, x)
        
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff:.6e}")
        if max_diff < 1e-3:
            print("PASSED!")
        else:
            print("FAILED!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")