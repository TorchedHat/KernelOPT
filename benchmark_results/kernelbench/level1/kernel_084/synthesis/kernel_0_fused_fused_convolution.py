import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    GROUPS: tl.constexpr,
    # strides for X (NCHW)
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    # strides for W (groups, 1, KH, KW) -> stored as (OUT_C, 1, KH, KW)
    stride_wc: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    # strides for Out (NCHW)
    stride_on: tl.constexpr, stride_oc: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W (spatial positions)
    # N = OUT_C (channels, but depthwise so each group has 1 input channel)
    # K = KERNEL_H * KERNEL_W (filter positions, since GROUP_IN_C = 1)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output spatial positions for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Masks
    mask_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
    mask_n = offs_n < OUT_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # For depthwise conv: each output channel c uses input channel c
    # GROUP_IN_C = 1, so K = KERNEL_H * KERNEL_W
    TOTAL_K: tl.constexpr = KERNEL_H * KERNEL_W
    
    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            # Input coordinates
            ih = out_h_idx * STRIDE_H + kh - PAD_H  # shape [BLOCK_M]
            iw = out_w_idx * STRIDE_W + kw - PAD_W  # shape [BLOCK_M]
            
            # For each output channel c in offs_n, input channel is also c (depthwise)
            # Load input: X[batch, c, ih, iw] for each (m, n) pair
            # X shape: [BLOCK_M, BLOCK_N] where each column uses a different channel
            
            valid_spatial = mask_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            
            # X_ptr[batch_idx, offs_n, ih, iw]
            x_ptrs = (X_ptr 
            
            x_mask = valid_spatial[:, None] & mask_n[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
            
            # Load weight: W[c, 0, kh, kw] for each c in offs_n
            # W shape: [OUT_C, 1, KH, KW], stride_wc = KH*KW*1
            w_ptrs = W_ptr + offs_n * stride_wc + kh * stride_wkh + kw * stride_wkw
            w_tile = tl.load(w_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]
            
            # acc[m, n] += x_tile[m, n] * w_tile[n]
            acc += x_tile * w_tile[None, :]
    
    # Store output
    out_ptrs = (Out_ptr 
    
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(weight, x):
    # weight: [128, 1, 3, 3], x: [64, 128, 256, 512]
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    GROUPS = IN_C  # depthwise: groups = in_channels
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1
    
    output = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
    
    depthwise_conv2d_kernel[grid](
        x, weight, output,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        GROUPS,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def get_inputs():
    weight = torch.randn(128, 1, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(64, 128, 256, 512, device='cuda', dtype=torch.float32)
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
        
        # Reference using PyTorch
        import torch.nn.functional as F
        ref_out = F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=128)
        
        our_out = kernel_function(weight, x)
        
        max_diff = (ref_out - our_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Correctness check passed!")
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        print("Kernel executed successfully.")