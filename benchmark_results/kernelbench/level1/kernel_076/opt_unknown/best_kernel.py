import torch
import triton
import triton.language as tl

# Conv1d parameters (from the spec)
# Input: (N=64, Ci=64, L=524280)
# Weight: (Co=128, Ci=64, K=3)
# stride=3, dilation=4, padding=0
# Output length: (524280 - 4*(3-1) - 1) // 3 + 1 = (524280 - 9) // 3 + 1 = 174757 + 1 = 174758
# Output: (64, 128, 174758)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, Ci, L_in, Co, L_out,
    stride, dilation, padding,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M, K,
):
    """
    M = N * L_out  (batch * output length)
    N_dim = Co     (output channels)
    K = Ci * KERNEL_SIZE  (reduction dimension)
    
      Y[batch, co, out_pos] = sum over ci, k: X[batch, ci, out_pos*stride + k*dilation] * W[co, ci, k]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # HOISTED outside K-loop: Decode M index: batch and output position
    batch = offs_m // L_out
    out_pos = offs_m % L_out
    # Precompute base pointer for X (includes batch offset and out_pos*stride)
    x_base = X_ptr + batch * (Ci * L_in) + out_pos * stride
    # Precompute valid_m mask (replaces valid_m & valid_batch inside loop)
    valid_m = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K = Ci * KERNEL_SIZE in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K index: channel and kernel position
        ci = offs_k // KERNEL_SIZE
        kpos = offs_k % KERNEL_SIZE
        
        # Input position (with padding subtracted for boundary check)
        in_pos = out_pos[:, None] * stride + kpos[None, :] * dilation - padding
        
        # Validity mask (valid_batch is redundant since M = N * L_out bounds batch)
        valid_k = offs_k[None, :] < K
        valid_in = (in_pos >= 0) & (in_pos < L_in)
        mask_x = valid_m[:, None] & valid_k & valid_in
        
        # Load X: shape [BLOCK_M, BLOCK_K]
        # X is (N, Ci, L_in), stride: Ci*L_in, L_in, 1
        # x_base already includes batch * (Ci * L_in) + out_pos * stride
        x_ptrs = x_base[:, None] + ci[None, :] * L_in + kpos[None, :] * dilation
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load W: shape [BLOCK_K, BLOCK_N]
        # W is (Co, Ci, KERNEL_SIZE), stride: Ci*KERNEL_SIZE, KERNEL_SIZE, 1
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < Co)
        w_ptrs = W_ptr + offs_n[None, :] * (Ci * KERNEL_SIZE) + ci[:, None] * KERNEL_SIZE + kpos[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Store output: Y is (N, Co, L_out)
    # Y[batch, co, out_pos] -> linear index: batch * Co * L_out + co * L_out + out_pos
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < Co)
    y_ptrs = Y_ptr + batch[:, None] * (Co * L_out) + offs_n[None, :] * L_out + out_pos[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)

def kernel_function(weight, x):
    """
    weight: (Co, Ci, K) = (128, 64, 3)
    x: (N, Ci, L_in) = (64, 64, 524280)
    """
    N, Ci, L_in = x.shape
    Co, Ci_w, KERNEL_SIZE = weight.shape
    
    stride = 3
    dilation = 4
    padding = 0
    
    L_out = (L_in + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1
    
    # Allocate output
    y = torch.empty((N, Co, L_out), device=x.device, dtype=torch.float32)
    
    M = N * L_out  # total output positions
    K = Ci * KERNEL_SIZE  # reduction dimension
    
    # Make contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Co, meta['BLOCK_N']),
    )
    
    conv1d_kernel[grid](
        x, weight, y,
        N, Ci, L_in, Co, L_out,
        stride, dilation, padding,
        KERNEL_SIZE,
        M=M, K=K,
    )
    
    return y


def get_inputs():
    batch_size = 64
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    length = 524280
    
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float32, device='cuda')
    x = torch.rand(batch_size, in_channels, length, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    weight, x = inputs
    
    # Verify correctness against PyTorch
    if not args.profile_only:
        import torch.nn.functional as F
        stride = 3
        dilation = 4
        padding = 0
        
        ref = F.conv1d(x, weight, bias=None, stride=stride, dilation=dilation, padding=padding)
        out = kernel_function(weight, x)
        
        torch.cuda.synchronize()
        
        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("PASSED: Results match PyTorch reference.")
        else:
            print("FAILED: Results do not match.")
    else:
        out = kernel_function(weight, x)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")