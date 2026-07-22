import torch
import triton
import triton.language as tl
import math

# Fused kernel: scale + causal mask + relu applied to attention scores
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def scale_mask_relu_kernel(
    scale: tl.constexpr,
    T: tl.constexpr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load attention score
    att = tl.load(att_ptr + offsets, mask=mask, other=0.0)
    
    # Scale
    att = att * scale
    
    # Compute position in [B, nh, T, T]
    # bias is [1, 1, T, T], so index into bias is (offset % (T*T))
    bias_idx = offsets % (T * T)
    bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    
    # Apply mask: where bias == 0, set to -inf
    neg_inf = float('-inf')
    att = tl.where(bias_val == 0.0, neg_inf, att)
    
    # ReLU
    att = tl.maximum(att, 0.0)
    
    tl.store(out_ptr + offsets, att, mask=mask)


# Kernel to prepare Q (add bias, reshape, transpose) -> contiguous [B*nh, T, hs]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def prepare_qkv_kernel(
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # offsets index into [B, nh, T, hs] with shape (B, nh, T, hs)
    # stride: (nh*T*hs, T*hs, hs, 1)
    T_hs = T * hs
    nh_T_hs = nh * T_hs
    
    i_hs = offsets % hs
    i_T = (offsets // hs) % T
    i_nh = (offsets // T_hs) % nh
    i_B = offsets // nh_T_hs
    
    # Source index in mm output [B*T, 3*C]
    # mm[b*T + t, c] where c = head*hs + hs_idx for Q (offset 0)
    # For Q: col = i_nh * hs + i_hs
    # For K: col = C + i_nh * hs + i_hs
    # For V: col = 2*C + i_nh * hs + i_hs
    row = i_B * T + i_T
    col_q = i_nh * hs + i_hs
    col_k = C + i_nh * hs + i_hs
    col_v = 2 * C + i_nh * hs + i_hs
    
    C3 = 3 * C
    mm_q = tl.load(mm_ptr + row * C3 + col_q, mask=mask, other=0.0)
    mm_k = tl.load(mm_ptr + row * C3 + col_k, mask=mask, other=0.0)
    mm_v = tl.load(mm_ptr + row * C3 + col_v, mask=mask, other=0.0)
    
    bias_q = tl.load(bias_ptr + col_q, mask=mask, other=0.0)
    bias_k = tl.load(bias_ptr + col_k, mask=mask, other=0.0)
    bias_v = tl.load(bias_ptr + col_v, mask=mask, other=0.0)
    
    q_val = mm_q + bias_q
    k_val = mm_k + bias_k
    v_val = mm_v + bias_v
    
    # Q output: [B, nh, T, hs] -> flat index = i_B * nh*T*hs + i_nh * T*hs + i_T * hs + i_hs
    q_idx = offsets  # already in [B, nh, T, hs] order
    tl.store(q_out_ptr + q_idx, q_val, mask=mask)
    
    # V output: [B, nh, T, hs] -> same layout
    tl.store(v_out_ptr + q_idx, v_val, mask=mask)
    
    # K output: [B, nh, hs, T] -> index = i_B * nh*hs*T + i_nh * hs*T + i_hs * T + i_T
    k_idx = i_B * nh_T_hs + i_nh * T_hs + i_hs * T + i_T
    tl.store(k_out_ptr + k_idx, k_val, mask=mask)


# Kernel to reshape output: [B, nh, T, hs] -> [B, T, C]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def reshape_output_kernel(
    B,
    T,
    C,
    nh,
    hs,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Output is [B, T, C] = [B, T, nh*hs]
    # offsets index into [B, T, C]
    i_c = offsets % C
    i_T = (offsets // C) % T
    i_B = offsets // (T * C)
    
    # Map C -> (nh, hs)
    i_nh = i_c // hs
    i_hs = i_c % hs
    
    # Input [B, nh, T, hs]: index = i_B * nh*T*hs + i_nh * T*hs + i_T * hs + i_hs
    inp_idx = i_B * (nh * T * hs) + i_nh * (T * hs) + i_T * hs + i_hs
    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, b_bias, x):
    """
    
    
    """
    B, T, C = x.shape
    nh = 12
    hs = C // nh  # 64
    scale = 1.0 / math.sqrt(hs)
    
    # Step 1: Linear projection using cuBLAS
    # x: [B, T, C] -> [B*T, C]
    x_2d = x.reshape(B * T, C)
    # mm: [B*T, C] @ [C, 3C] = [B*T, 3C]
    mm_out = torch.mm(x_2d, p_c_attn_weight.t())  # [B*T, 3C]
    
    # Step 2: Prepare Q, K^T, V using fused Triton kernel
    numel = B * nh * T * hs
    q = torch.empty(B * nh, T, hs, device=x.device, dtype=x.dtype)
    k_t = torch.empty(B * nh, hs, T, device=x.device, dtype=x.dtype)
    v = torch.empty(B * nh, T, hs, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    prepare_qkv_kernel[grid](
        mm_out, p_c_attn_bias,
        q, k_t, v,
        B, T, C, nh, hs,
        numel,
    )
    
    # Step 3: Compute attention scores Q @ K^T using cuBLAS bmm
    # q: [B*nh, T, hs], k_t: [B*nh, hs, T]
    att = torch.bmm(q, k_t)  # [B*nh, T, T]
    
    # Step 4: Scale + mask + relu using Triton
    att_numel = B * nh * T * T
    att_out = torch.empty_like(att)
    
    # b_bias is [1, 1, T, T], we need [T*T] for indexing
    bias_2d = b_bias.reshape(T * T)
    
    grid2 = lambda meta: (triton.cdiv(att_numel, meta['BLOCK_SIZE']),)
    scale_mask_relu_kernel[grid2](
        att, bias_2d, att_out,
        scale, T,
        att_numel,
    )
    
    # Step 5: Compute output att @ V using cuBLAS bmm
    # att_out: [B*nh, T, T], v: [B*nh, T, hs]
    y_bnh = torch.bmm(att_out, v)  # [B*nh, T, hs]
    
    # Step 6: Reshape [B*nh, T, hs] -> [B, nh, T, hs] -> [B, T, C]
    # First reshape to [B, nh, T, hs]
    y_4d = y_bnh.reshape(B, nh, T, hs)
    
    # Then reshape to [B, T, C] using Triton kernel
    out_numel = B * T * C
    y_out = torch.empty(B, T, C, device=x.device, dtype=x.dtype)
    
    grid3 = lambda meta: (triton.cdiv(out_numel, meta['BLOCK_SIZE']),)
    reshape_output_kernel[grid3](
        y_4d, y_out,
        B, T, C, nh, hs,
        out_numel,
    )
    
    return y_out


def get_inputs():
    batch_size = 16
    max_seqlen = 1024
    n_embd = 768
    n_head = 12
    
    p_c_attn_weight = torch.randn(3 * n_embd, n_embd, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(3 * n_embd, device='cuda', dtype=torch.float32)
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device='cuda', dtype=torch.float32)).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(batch_size, max_seqlen, n_embd, device='cuda', dtype=torch.float32)
    
    return [p_c_attn_weight, p_c_attn_bias, b_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        p_c_attn_weight, p_c_attn_bias, b_bias, x = inputs
        B, T, C = x.shape
        n_head = 12
        
        # Reference implementation
        qkv = torch.nn.functional.linear(x, p_c_attn_weight, p_c_attn_bias)
        q_ref, k_ref, v_ref = qkv.split(C, dim=2)
        
        k_ref = k_ref.view(B, T, n_head, C // n_head).transpose(1, 2)
        q_ref = q_ref.view(B, T, n_head, C // n_head).transpose(1, 2)
        v_ref = v_ref.view(B, T, n_head, C // n_head).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(k_ref.size(-1))
        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * scale
        att_ref = att_ref.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = torch.relu(att_ref)
        
        y_ref = att_ref @ v_ref
        y_ref = y_ref.transpose(1, 2).contiguous().view(B, T, C)
        
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - y_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")