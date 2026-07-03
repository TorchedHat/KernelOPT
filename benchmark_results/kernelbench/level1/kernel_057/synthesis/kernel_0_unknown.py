import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, out_ptr,
    B, IC, OC, IH, IW, OH, OW,
    KH, KW,
    stride_xb, stride_xic, stride_xih, stride_xiw,
    stride_wic, stride_woc, stride_wkh, stride_wkw,
    stride_ob, stride_ooc, stride_ooh, stride_oow,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes a BLOCK_M x BLOCK_N tile of the output
    # M = B * OH * OW (output spatial positions)
    # N = OC (output channels)
    # K = IC * KH * KW (reduction dimension)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Decode m_offs -> (b, oh, ow)
    # m = b * OH * OW + oh * OW + ow
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # k = ic * KH * KW + kh * KW + kw
        k_mask = k_offs < K
        
        ic_k = k_offs // (KH * KW)
        rem_k = k_offs % (KH * KW)
        kh_k = rem_k // KW
        kw_k = rem_k % KW
        
        # For each (m, k): m -> (b, oh, ow), k -> (ic, kh, kw)
        # input index: ih = oh - kh, iw = ow - kw
        # valid if 0 <= ih < IH and 0 <= iw < IW
        
        m_mask = m_offs < M
        
        b_m = m_offs // (OH * OW)
        rem_m = m_offs % (OH * OW)
        oh_m = rem_m // OW
        ow_m = rem_m % OW
        
        # Compute input indices: ih = oh - kh, iw = ow - kw
        # Shape: [BLOCK_M, BLOCK_K]
        ih_mk = oh_m[:, None] - kh_k[None, :]
        iw_mk = ow_m[:, None] - kw_k[None, :]
        
        valid_mk = (
            m_mask[:, None] & k_mask[None, :] &
            (ih_mk >= 0) & (ih_mk < IH) &
            (iw_mk >= 0) & (iw_mk < IW)
        )
        
        x_idx = (
            b_m[:, None] * stride_xb +
            ic_k[None, :] * stride_xic +
        )
        
        x_tile = tl.load(x_ptr + x_idx, mask=valid_mk, other=0.0)
        
        # Weight: w[ic, oc, kh, kw] -> shape [BLOCK_K, BLOCK_N]
        n_mask = n_offs < N
        
        w_idx = (
            ic_k[:, None] * stride_wic +
            n_offs[None, :] * stride_woc +
            kh_k[:, None] * stride_wkh +
            kw_k[:, None] * stride_wkw
        )
        
        w_tile = tl.load(w_ptr + w_idx, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    m_mask = m_offs < M
    n_mask = n_offs < N
    
    b_m = m_offs // (OH * OW)
    rem_m = m_offs % (OH * OW)
    oh_m = rem_m // OW
    ow_m = rem_m % OW
    
    out_idx = (
        b_m[:, None] * stride_ob +
        n_offs[None, :] * stride_ooc +
        oh_m[:, None] * stride_ooh +
        ow_m[:, None] * stride_oow
    )
    
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(weight, x):
    # weight shape: [IC, OC, KH, KW] (ConvTranspose2d weight layout)
    # x shape: [B, IC, IH, IW]
    # output shape: [B, OC, OH, OW]
    # OH = IH + KH - 1 (no padding, stride=1)
    # OW = IW + KW - 1
    
    B, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    
    assert IC == IC_w, f"IC mismatch: {IC} vs {IC_w}"
    
    # stride=1, padding=0
    OH = IH + KH - 1
    OW = IW + KW - 1
    
    out = torch.zeros(B, OC, OH, OW, device=x.device, dtype=torch.float32)
    
    M = B * OH * OW
    N = OC
    K = IC * KH * KW
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv_transpose2d_kernel[grid](
        x, weight, out,
        B, IC, OC, IH, IW, OH, OW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M, N, K,
    )
    
    return out


def get_inputs():
    weight = torch.randn(64, 64, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 64, 1024, 1024, device='cuda', dtype=torch.float32)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    weight, x = inputs
    
    output = kernel_function(weight, x)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Verify against PyTorch reference
        import torch.nn.functional as F
        ref = F.conv_transpose2d(x, weight, bias=None, stride=1, padding=0)
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance.")
        else:
            print("FAILED: Results do not match!")
        print("Kernel executed successfully.")