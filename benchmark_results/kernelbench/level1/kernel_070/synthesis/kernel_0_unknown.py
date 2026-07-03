import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    # X shape: (N, Ci, Di, Hi, Wi)
    N, Ci, Di, Hi, Wi,
    # W shape: (Ci, Co, Kd, Kh, Kw)
    Co, Kd, Kh, Kw,
    # Y shape: (N, Co, Do, Ho, Wo)
    Do, Ho, Wo,
    # Strides for X
    x_sN, x_sCi, x_sDi, x_sHi, x_sWi,
    # Strides for W
    w_sCi, w_sCo, w_sKd, w_sKh, w_sKw,
    # Strides for Y
    y_sN, y_sCo, y_sDo, y_sHo, y_sWo,
    # Stride and padding params
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # Problem dimensions for GEMM
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = batch * Do * Ho * Wo (output spatial positions)
    # N_out = Co (output channels)
    # K = Ci * Kd * Kh * Kw (reduction: input channels * kernel volume)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index -> (batch, od, oh, ow)
    ow_idx = offs_m % Wo
    oh_idx = (offs_m // Wo) % Ho
    od_idx = (offs_m // (Wo * Ho)) % Do
    batch_idx = offs_m // (Wo * Ho * Do)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K index -> (ci, kd, kh, kw)
        # K = Ci * Kd * Kh * Kw
        # ordering: ci varies slowest, kd, kh, kw fastest
        kw_k = offs_k % Kw
        kh_k = (offs_k // Kw) % Kh
        kd_k = (offs_k // (Kw * Kh)) % Kd
        ci_k = offs_k // (Kw * Kh * Kd)
        
        # For ConvTranspose3d:
        # output[n, co, od, oh, ow] += input[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]
        # where: od = id * stride_d - pad_d + kd
        #        oh = ih * stride_h - pad_h + kh
        #        ow = iw * stride_w - pad_w + kw
        # So: id = (od + pad_d - kd) / stride_d  (must be integer and in range)
        #     ih = (oh + pad_h - kh) / stride_h
        #     iw = (ow + pad_w - kw) / stride_w
        
        # Compute input indices
        # offs_m: [BLOCK_M], offs_k: [BLOCK_K]
        # od_idx: [BLOCK_M], kd_k: [BLOCK_K]
        
        od_m = od_idx[:, None]  # [BLOCK_M, 1]
        oh_m = oh_idx[:, None]
        ow_m = ow_idx[:, None]
        batch_m = batch_idx[:, None]
        
        kd_k2 = kd_k[None, :]  # [1, BLOCK_K]
        kh_k2 = kh_k[None, :]
        kw_k2 = kw_k[None, :]
        ci_k2 = ci_k[None, :]
        
        # Numerators for input index computation
        num_d = od_m + pad_d - kd_k2
        num_h = oh_m + pad_h - kh_k2
        num_w = ow_m + pad_w - kw_k2
        
        # Check divisibility by stride
        div_ok_d = (num_d % stride_d) == 0
        div_ok_h = (num_h % stride_h) == 0
        div_ok_w = (num_w % stride_w) == 0
        
        id_idx = num_d // stride_d
        ih_idx = num_h // stride_h
        iw_idx = num_w // stride_w
        
        # Boundary checks
        valid_d = div_ok_d & (id_idx >= 0) & (id_idx < Di)
        valid_h = div_ok_h & (ih_idx >= 0) & (ih_idx < Hi)
        valid_w = div_ok_w & (iw_idx >= 0) & (iw_idx < Wi)
        valid_ci = ci_k2 < Ci
        valid_k = offs_k[None, :] < K
        valid_m = offs_m[:, None] < M
        
        valid_mask = valid_d & valid_h & valid_w & valid_ci & valid_k & valid_m
        
        # Load X: shape [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr 
        x_tile = tl.load(x_ptrs, mask=valid_mask, other=0.0)
        
        # Load W: shape [BLOCK_K, BLOCK_N]
        # W[ci, co, kd, kh, kw]
        valid_n = offs_n[None, :] < N_out
        w_ptrs = (W_ptr 
        w_mask = (offs_k[:, None] < K) & valid_n
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Store output Y[batch, co, od, oh, ow]
    valid_m_out = offs_m[:, None] < M
    valid_n_out = offs_n[None, :] < N_out
    
    y_ptrs = (Y_ptr 
    
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_m_out & valid_n_out)


def kernel_function(weight, x):
    # weight: (Ci, Co, Kd, Kh, Kw) = (48, 24, 3, 3, 3)
    # x: (N, Ci, Di, Hi, Wi) = (8, 48, 96, 96, 96)
    # output: (N, Co, Do, Ho, Wo)
    
    N, Ci, Di, Hi, Wi = x.shape
    Ci_w, Co, Kd, Kh, Kw = weight.shape
    
    stride_d, stride_h, stride_w = 1, 1, 1
    pad_d, pad_h, pad_w = 0, 0, 0
    
    Do = (Di - 1) * stride_d - 2 * pad_d + Kd
    Ho = (Hi - 1) * stride_h - 2 * pad_h + Kh
    Wo = (Wi - 1) * stride_w - 2 * pad_w + Kw
    
    y = torch.zeros(N, Co, Do, Ho, Wo, device=x.device, dtype=x.dtype)
    
    M = N * Do * Ho * Wo
    N_out = Co
    K = Ci * Kd * Kh * Kw
    
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N_out, meta['BLOCK_N']),
        )
    
    conv_transpose3d_kernel[grid](
        x, weight, y,
        N, Ci, Di, Hi, Wi,
        Co, Kd, Kh, Kw,
        Do, Ho, Wo,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        M, N_out, K,
    )
    
    return y


def get_inputs():
    weight = torch.randn(48, 24, 3, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 48, 96, 96, 96, device='cuda', dtype=torch.float32)
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
        ref_out = F.conv_transpose3d(x, weight, bias=None, stride=1, padding=0)
        
        our_out = kernel_function(weight, x)
        
        print(f"Output shape: {our_out.shape}")
        print(f"Reference shape: {ref_out.shape}")
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        print(f"Match: {torch.allclose(our_out, ref_out, atol=1e-3)}")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")