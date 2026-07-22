import torch
import triton
import triton.language as tl
import math

# We'll implement the key fused kernels in Triton and use torch for matmuls

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def segsum_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # offsets index into [B, H, C, L, L]
    # stride: [H*C*L*L, C*L*L, L*L, L, 1]
    ll = L * L
    cl = C * ll
    hcl = H * cl
    
    idx = offsets
    b_idx = idx // hcl
    rem = idx % hcl
    h_idx = rem // cl
    rem2 = rem % cl
    c_idx = rem2 // ll
    rem3 = rem2 % ll
    i_idx = rem3 // L  # row
    j_idx = rem3 % L   # col
    
    # cumsum is [B, H, C, L] with stride [H*C*L, C*L, L, 1]
    cs_stride_b = H * C * L
    cs_stride_h = C * L
    cs_stride_c = L
    
    # x_segsum[b,h,c,i,j] = cumsum[b,h,c,i] - cumsum[b,h,c,j]
    cs_i = tl.load(x_ptr + b_idx * cs_stride_b + h_idx * cs_stride_h + c_idx * cs_stride_c + i_idx, mask=mask, other=0.0)
    cs_j = tl.load(x_ptr + b_idx * cs_stride_b + h_idx * cs_stride_h + c_idx * cs_stride_c + j_idx, mask=mask, other=0.0)
    
    val = cs_i - cs_j
    
    # Apply causal mask: only keep lower triangular (i >= j)
    is_lower = i_idx >= j_idx
    val = tl.where(is_lower & mask, val, float('-inf'))
    
    # exp
    val = tl.exp(val)
    val = tl.where(~is_lower & mask, 0.0, val)  # exp(-inf) = 0 but let's be explicit
    
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def decay_states_kernel(
    out_ptr,       # [B, H, C, L]  = exp(cumsum[...,-1:] - cumsum)
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # [B, H, C, L] layout
    cl = C * L
    hcl = H * cl
    
    idx = offsets
    b_idx = idx // hcl
    rem = idx % hcl
    h_idx = rem // cl
    rem2 = rem % cl
    c_idx = rem2 // L
    l_idx = rem2 % L
    
    base = b_idx * hcl + h_idx * cl + c_idx * L
    last_val = tl.load(cumsum_ptr + base + (L - 1), mask=mask, other=0.0)
    cur_val = tl.load(cumsum_ptr + offsets, mask=mask, other=0.0)
    
    result = tl.exp(last_val - cur_val)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def segsum_inter_kernel(
    x_ptr,      # [B, H, C_pad] cumsum input (C_pad = C+1 = 3)
    B: tl.constexpr,
    H: tl.constexpr,
    C_pad: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    cc = C_pad * C_pad
    hcc = H * cc
    
    idx = offsets
    b_idx = idx // hcc
    rem = idx % hcc
    h_idx = rem // cc
    rem2 = rem % cc
    i_idx = rem2 // C_pad
    j_idx = rem2 % C_pad
    
    cs_stride_b = H * C_pad
    cs_stride_h = C_pad
    
    cs_i = tl.load(x_ptr + b_idx * cs_stride_b + h_idx * cs_stride_h + i_idx, mask=mask, other=0.0)
    cs_j = tl.load(x_ptr + b_idx * cs_stride_b + h_idx * cs_stride_h + j_idx, mask=mask, other=0.0)
    
    val = cs_i - cs_j
    is_lower = i_idx >= j_idx
    val = tl.where(is_lower & mask, val, float('-inf'))
    val = tl.exp(val)
    val = tl.where(~is_lower & mask, 0.0, val)
    
    tl.store(out_ptr + offsets, val, mask=mask)


def kernel_function(p_a, p_b, p_c, x):
    """
    
    
    """
    batch = 2048
    seq_len = 128
    n_heads = 8
    d_head = 64
    d_state = 16
    block_len = 64
    n_chunks = seq_len // block_len  # = 2
    
    device = x.device
    dtype = x.dtype
    
    # Reshape into blocks
    # X_blocks: [B, C, L, H, d_head]
    X_blocks = x.reshape(batch, n_chunks, block_len, n_heads, d_head)
    # A_blocks: [B, C, L, H] -> permute to [B, H, C, L]
    A_blocks = p_a.reshape(batch, n_chunks, block_len, n_heads).permute(0, 3, 1, 2).contiguous()
    # B_blocks: [B, C, L, H, d_state]
    B_blocks = p_b.reshape(batch, n_chunks, block_len, n_heads, d_state)
    # C_blocks: [B, C, L, H, d_state]
    C_blocks = p_c.reshape(batch, n_chunks, block_len, n_heads, d_state)
    
    # A_cumsum: [B, H, C, L]
    A_cumsum = torch.cumsum(A_blocks, dim=-1)
    
    # ===== 1. Compute diagonal block outputs =====
    # L_matrix = exp(segsum(A_blocks)) shape [B, H, C, L, L]
    # segsum: x_segsum[b,h,c,i,j] = cumsum[b,h,c,i] - cumsum[b,h,c,j], masked lower triangular
    
    N_seg = batch * n_heads * n_chunks * block_len * block_len
    L_matrix = torch.empty(batch, n_heads, n_chunks, block_len, block_len, device=device, dtype=dtype)
    
    grid_seg = lambda meta: (triton.cdiv(N_seg, meta['BLOCK_SIZE']),)
    segsum_kernel[grid_seg](
        A_cumsum, L_matrix,
        batch, n_heads, n_chunks, block_len,
        N_seg,
    )
    
    # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_matrix, X_blocks)
    # Let's break this down:
    # C_blocks: [B, C, L, H, N] (N=d_state)
    # B_blocks: [B, C, S, H, N] (S=L=block_len, same dim)
    # L_matrix: [B, H, C, L, S]
    # X_blocks: [B, C, S, H, P] (P=d_head)
    # Output: [B, C, L, H, P]
    
    # Step 1: contract over s and n: 
    # First compute: tmp1[b,c,l,h,s] = sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
    # = einsum("bclhn,bcshn->bclhs", C_blocks, B_blocks)
    # C_blocks: [B,C,L,H,N], B_blocks: [B,C,S,H,N]
    # Reshape for bmm: [B*C*H, L, N] x [B*C*H, N, S] -> [B*C*H, L, S]
    
    BCH = batch * n_chunks * n_heads
    C_reshaped = C_blocks.permute(0,1,3,2,4).reshape(BCH, block_len, d_state)  # [BCH, L, N]
    B_reshaped = B_blocks.permute(0,1,3,4,2).reshape(BCH, d_state, block_len)  # [BCH, N, S]
    CB = torch.bmm(C_reshaped, B_reshaped)  # [BCH, L, S]
    CB = CB.reshape(batch, n_chunks, n_heads, block_len, block_len)  # [B,C,H,L,S]
    
    # Now multiply by L_matrix[B,H,C,L,S] and sum over s with X_blocks[B,C,S,H,P]
    # tmp2[b,c,l,h,s] = CB[b,c,h,l,s] * L[b,h,c,l,s]
    # L_matrix: [B,H,C,L,S] -> permute to [B,C,H,L,S]
    L_perm = L_matrix.permute(0,2,1,3,4)  # [B,C,H,L,S]
    CBL = CB * L_perm  # [B,C,H,L,S]
    
    # Y_diag[b,c,l,h,p] = sum_s CBL[b,c,h,l,s] * X[b,c,s,h,p]
    # Reshape: CBL [B,C,H,L,S] -> [BCH, L, S]
    # X_blocks [B,C,S,H,P] -> [BCH, S, P]
    CBL_reshaped = CBL.reshape(BCH, block_len, block_len)
    X_reshaped = X_blocks.permute(0,1,3,2,4).reshape(BCH, block_len, d_head)  # [BCH, S, P]
    Y_diag = torch.bmm(CBL_reshaped, X_reshaped)  # [BCH, L, P]
    Y_diag = Y_diag.reshape(batch, n_chunks, n_heads, block_len, d_head)
    Y_diag = Y_diag.permute(0,1,3,2,4)  # [B,C,L,H,P]
    
    # ===== 2. Compute intra-chunk states =====
    # decay_states = exp(A_cumsum[...,-1:] - A_cumsum)  [B,H,C,L]
    N_decay = batch * n_heads * n_chunks * block_len
    decay_states = torch.empty_like(A_cumsum)
    grid_decay = lambda meta: (triton.cdiv(N_decay, meta['BLOCK_SIZE']),)
    decay_states_kernel[grid_decay](
        A_cumsum, decay_states,
        batch, n_heads, n_chunks, block_len,
        N_decay,
    )
    
    # states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
    # B_blocks: [B,C,L,H,N]
    # decay_states: [B,H,C,L]
    # X_blocks: [B,C,L,H,P]
    # Output: [B,C,H,P,N]
    
    # Step: multiply B and decay: B_decay[b,c,l,h,n] = B[b,c,l,h,n] * decay[b,h,c,l]
    # decay: [B,H,C,L] -> [B,C,L,H,1]
    decay_perm = decay_states.permute(0,2,3,1).unsqueeze(-1)  # [B,C,L,H,1]
    B_decay = B_blocks * decay_perm  # [B,C,L,H,N]
    
    # states[b,c,h,p,n] = sum_l B_decay[b,c,l,h,n] * X[b,c,l,h,p]
    # Reshape: B_decay [B,C,L,H,N] -> [BCH, L, N]
    # X_blocks [B,C,L,H,P] -> [BCH, L, P]
    # states = X^T @ B_decay -> [BCH, P, N]
    B_decay_r = B_decay.permute(0,1,3,2,4).reshape(BCH, block_len, d_state)  # [BCH, L, N]
    X_r2 = X_blocks.permute(0,1,3,2,4).reshape(BCH, block_len, d_head)  # [BCH, L, P]
    states = torch.bmm(X_r2.transpose(1,2), B_decay_r)  # [BCH, P, N]
    states = states.reshape(batch, n_chunks, n_heads, d_head, d_state)  # [B,C,H,P,N]
    
    # ===== 3. Compute inter-chunk recurrence =====
    # initial_states = zeros_like(states[:, :1])  [B,1,H,P,N]
    initial_states = torch.zeros(batch, 1, n_heads, d_head, d_state, device=device, dtype=dtype)
    states_cat = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N] = [B,3,H,P,N]
    
    # select last element of A_cumsum along L dim: [B,H,C]
    A_last = A_cumsum[:, :, :, -1]  # [B,H,C]
    
    # pad: [B,H,C+1] with 0 prepended
    A_last_padded = torch.nn.functional.pad(A_last, (1, 0), value=0.0)  # [B,H,3]
    
    # cumsum of padded: [B,H,3]
    A_inter_cumsum = torch.cumsum(A_last_padded, dim=-1)  # [B,H,3]
    
    # segsum for inter-chunk: [B,H,3,3]
    N_inter = batch * n_heads * 3 * 3
    decay_inter = torch.empty(batch, n_heads, 3, 3, device=device, dtype=dtype)
    grid_inter = lambda meta: (triton.cdiv(N_inter, meta['BLOCK_SIZE']),)
    segsum_inter_kernel[grid_inter](
        A_inter_cumsum, decay_inter,
        batch, n_heads, 3,
        N_inter,
    )
    
    # new_states = einsum("bhij,bjhpn->bihpn", decay_inter, states_cat)
    # decay_inter: [B,H,3,3]
    # states_cat: [B,3,H,P,N]
    # Output: [B,3,H,P,N]
    
    # Reshape for bmm:
    # decay_inter: [B,H,3,3] -> [B*H, 3, 3]
    # states_cat: [B,3,H,P,N] -> [B,H,3,P*N] -> [B*H, 3, P*N]
    BH = batch * n_heads
    decay_inter_r = decay_inter.reshape(BH, 3, 3)
    states_cat_r = states_cat.permute(0,2,1,3,4).reshape(BH, 3, d_head * d_state)
    new_states = torch.bmm(decay_inter_r, states_cat_r)  # [BH, 3, P*N]
    new_states = new_states.reshape(batch, n_heads, 3, d_head, d_state)
    new_states = new_states.permute(0,2,1,3,4)  # [B,3,H,P,N]
    
    # select index 2 along dim 1: [B,H,P,N]
    result = new_states[:, 2, :, :, :]  # [B,H,P,N] = [B,H,d_head,d_state]
    # Reshape to [B,H,d_head,d_state] which matches [2048, 8, 64, 16]
    
    return result


def get_inputs():
    batch = 2048
    seq_len = 128
    n_heads = 8
    d_head = 64
    d_state = 16
    
    p_a = torch.randn(batch, seq_len, n_heads, device='cuda', dtype=torch.float32)
    p_b = torch.randn(batch, seq_len, n_heads, d_state, device='cuda', dtype=torch.float32)
    p_c = torch.randn(batch, seq_len, n_heads, d_state, device='cuda', dtype=torch.float32)
    x = torch.randn(batch, seq_len, n_heads, d_head, device='cuda', dtype=torch.float32)
    
    return [p_a, p_b, p_c, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Reference implementation
        p_a, p_b, p_c, x = inputs
        batch = 2048
        seq_len = 128
        n_heads = 8
        d_head = 64
        d_state = 16
        block_len = 64
        n_chunks = seq_len // block_len
        
        X_blocks = x.reshape(batch, n_chunks, block_len, n_heads, d_head)
        A_blocks = p_a.reshape(batch, n_chunks, block_len, n_heads).permute(0, 3, 1, 2).contiguous()
        B_blocks = p_b.reshape(batch, n_chunks, block_len, n_heads, d_state)
        C_blocks = p_c.reshape(batch, n_chunks, block_len, n_heads, d_state)
        
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        def segsum_ref(x):
            T = x.size(-1)
            x_cumsum = torch.cumsum(x, dim=-1)
            x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool))
            x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
            return x_segsum
        
        L_ref = torch.exp(segsum_ref(A_blocks))
        Y_diag_ref = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_ref, X_blocks)
        
        decay_states_ref = torch.exp(A_cumsum[:,:,:,-1:] - A_cumsum)
        states_ref = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states_ref, X_blocks)
        
        initial_states_ref = torch.zeros_like(states_ref[:, :1])
        states_cat_ref = torch.cat([initial_states_ref, states_ref], dim=1)
        
        A_last_ref = A_cumsum[:,:,:,-1]
        A_last_padded_ref = torch.nn.functional.pad(A_last_ref, (1, 0), value=0.0)
        A_inter_cumsum_ref = torch.cumsum(A_last_padded_ref, dim=-1)
        
        decay_inter_ref = torch.exp(segsum_ref(A_inter_cumsum_ref))
        new_states_ref = torch.einsum("bhij,bjhpn->bihpn", decay_inter_ref, states_cat_ref)
        result_ref = new_states_ref[:, 2, :, :, :]
        
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {result_ref.shape}")
        
        max_diff = (output - result_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        
        assert torch.allclose(output, result_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")