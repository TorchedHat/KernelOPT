import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_epilogue_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X: (N, IN_C, IN_H, IN_W)
    sx_n, sx_c, sx_h, sx_w,
    # strides for W: (IN_C, OUT_C, KH, KW)
    sw_ic, sw_oc, sw_kh, sw_kw,
    # epilogue params
    add_val, mul_val,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    M = N * OUT_H * OUT_W  (output spatial positions)
    N_dim = OUT_C
    K = IN_C * KH * KW  (but only valid (kh,kw) where (oh-kh) % stride == 0)
    
    y[batch, oc, oh, ow] = sum over (ic, kh, kw) where (oh-kh)%stride==0 and (ow-kw)%stride==0:
        x[batch, ic, (oh-kh)//stride, (ow-kw)//stride] * w[ic, oc, kh, kw]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * KH * KW
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (ic, kh, kw)
        # K layout: ic * KH * KW + kh * KW + kw
        kw_k = offs_k % KW
        kh_k = (offs_k // KW) % KH
        ic_k = offs_k // (KW * KH)
        
        # For transposed conv: input position
        # oh = ih * STRIDE_H + kh => ih = (oh - kh) / STRIDE_H (only if divisible)
        # ow = iw * STRIDE_W + kw => iw = (ow - kw) / STRIDE_W (only if divisible)
        
        # oh shape: (BLOCK_M,), kh_k shape: (BLOCK_K,)
        oh_minus_kh = oh[:, None] - kh_k[None, :]  # (BLOCK_M, BLOCK_K)
        ow_minus_kw = ow[:, None] - kw_k[None, :]  # (BLOCK_M, BLOCK_K)
        
        ih = oh_minus_kh // STRIDE_H
        iw = ow_minus_kw // STRIDE_W
        
        valid_h = (oh_minus_kh >= 0) & (oh_minus_kh % STRIDE_H == 0) & (ih < IN_H)
        valid_w = (ow_minus_kw >= 0) & (ow_minus_kw % STRIDE_W == 0) & (iw < IN_W)
        valid_k = offs_k < K
        valid_m = offs_m < M
        valid_oc = offs_n < OUT_C
        
        valid_x = valid_h & valid_w & valid_k[None, :] & valid_m[:, None] & (batch[:, None] < N)
        
        # Load X: x[batch, ic, ih, iw]
        x_ptrs = (X_ptr 
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # Load W: w[ic, oc, kh, kw] -> shape (BLOCK_K, BLOCK_N)
        w_ptrs = (W_ptr 
        valid_w_tile = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w_tile, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    # Epilogue: add_val, min(0), gelu, mul_val
    acc = acc + add_val
    acc = tl.minimum(acc, 0.0)
    
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the standard approximation
    x_gelu = acc
    x3 = x_gelu * x_gelu * x_gelu
    inner = 0.7978845608028654 * (x_gelu + 0.044715 * x3)
    # tanh
    tanh_val = (tl.exp(2.0 * inner) - 1.0) / (tl.exp(2.0 * inner) + 1.0)
    gelu_out = 0.5 * x_gelu * (1.0 + tanh_val)
    
    gelu_out = gelu_out * mul_val
    
    # Store output: Y[batch, oc, oh, ow]
    # Y layout: (N, OUT_C, OUT_H, OUT_W) -> stride (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    sy_n = OUT_C * OUT_H * OUT_W
    sy_c = OUT_H * OUT_W
    sy_h = OUT_W
    sy_w = 1
    
    y_ptrs = (Y_ptr 
    
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    tl.store(y_ptrs, gelu_out.to(tl.float32), mask=valid_out)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, c_lifted_tensor_0, x):
    # x: (N, IN_C, IN_H, IN_W)
    # weight: (IN_C, OUT_C, KH, KW)  -- ConvTranspose2d weight shape
    N, IN_C, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KH, KW = p_conv_transpose_weight.shape
    
    STRIDE_H = 2
    STRIDE_W = 2
    PAD_H = 0
    PAD_W = 0
    
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    add_val = float(c_lifted_tensor_0.item())
    mul_val = 2.0
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv_transpose2d_epilogue_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        add_val, mul_val,
        M, K,
    )
    
    return y


def get_inputs():
    torch.manual_seed(0)
    p_conv_transpose_weight = torch.randn(64, 128, 4, 4, dtype=torch.float32, device='cuda')
    p_conv_transpose_bias = torch.randn(128, dtype=torch.float32, device='cuda')
    c_lifted_tensor_0 = torch.tensor(0.5, dtype=torch.float32, device='cuda')
    x = torch.rand(128, 64, 64, 64, dtype=torch.float32, device='cuda')
    return [p_conv_transpose_weight, p_conv_transpose_bias, c_lifted_tensor_0, x]


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
        
        p_conv_transpose_weight, p_conv_transpose_bias, c_lifted_tensor_0, x = inputs
        
        # Reference computation
        ref = F.conv_transpose2d(x, p_conv_transpose_weight, p_conv_transpose_bias, stride=2)
        add_val = c_lifted_tensor_0.item()
        ref = ref + add_val
        ref = torch.min(ref, torch.tensor(0.0, device=ref.device))
        ref = F.gelu(ref)
        ref = ref * 2.0
        
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")