import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Kernel 1: Fused cumsum with permute (A_blocks -> A_cumsum)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 2}, num_warps=4),
        triton.Config({'XBLOCK': 4}, num_warps=8),
        triton.Config({'XBLOCK': 8}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def cumsum_permute_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    xnumel: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length L
    # xnumel = B * H * C
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    for xi in range(XBLOCK):
        row = xoffset + xi
        if row < xnumel:
            b = row // (H * C)
            hc = row % (H * C)
            h = hc // C
            c = hc % C
            
            acc = tl.zeros([1], dtype=tl.float32)[0]
            for l in range(L):
                # in_ptr layout: [B, C, L, H] -> index = b*C*L*H + c*L*H + l*H + h
                val = tl.load(in_ptr + b * C * L * H + c * L * H + l * H + h)
                acc = acc + val
                # out_ptr layout: [B, H, C, L] -> index = b*H*C*L + h*C*L + c*L + l
                tl.store(out_ptr + b * H * C * L + h * C * L + c * L + l, acc)


# ============================================================
# Kernel 2: Segsum + masked_fill + exp (for L=64 case)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 16}, num_warps=2),
        triton.Config({'BLOCK_T': 8}, num_warps=2),
    ],
    key=['T'],
)
@triton.jit
def segsum_exp_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one (b, h, c) -> T x T matrix
    pid = tl.program_id(0)
    total = B * H * C
    if pid >= total:
        return
    
    b = pid // (H * C)
    hc = pid % (H * C)
    h = hc // C
    c = hc % C
    
    base = b * H * C * T + h * C * T + c * T
    
    # Load the T cumsum values
    offs = tl.arange(0, BLOCK_T)
    mask = offs < T
    vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    
    # Compute T x T segsum matrix and write
    # segsum[i, j] = cumsum[i] - cumsum[j]
    # mask: tril (i >= j), else -inf
    out_base = pid * T * T
    
    row_offs = tl.arange(0, BLOCK_T)
    col_offs = tl.arange(0, BLOCK_T)
    
    row_mask = row_offs < T
    col_mask = col_offs < T
    
    row_vals = tl.load(x_ptr + base + row_offs, mask=row_mask, other=0.0)
    col_vals = tl.load(x_ptr + base + col_offs, mask=col_mask, other=0.0)
    
    # row_vals[:, None] - col_vals[None, :]
    diff = row_vals[:, None] - col_vals[None, :]
    
    # causal mask: i >= j (tril)
    causal = row_offs[:, None] >= col_offs[None, :]
    
    diff = tl.where(causal, diff, float('-inf'))
    result = libdevice.exp(diff)
    
    rc_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + out_base + row_offs[:, None] * T + col_offs[None, :], result, mask=rc_mask)


# ============================================================
# Kernel 3: Segsum + masked_fill + exp (for T=3 case)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 4}, num_warps=2),
        triton.Config({'BLOCK_T': 4}, num_warps=4),
        triton.Config({'BLOCK_T': 4}, num_warps=1),
        triton.Config({'BLOCK_T': 4}, num_warps=8),
    ],
    key=['T'],
)
@triton.jit
def segsum_exp_small_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return
    
    b = pid // H
    h = pid % H
    
    base = b * H * T + h * T
    out_base = pid * T * T
    
    row_offs = tl.arange(0, BLOCK_T)
    col_offs = tl.arange(0, BLOCK_T)
    
    row_mask = row_offs < T
    col_mask = col_offs < T
    
    row_vals = tl.load(x_ptr + base + row_offs, mask=row_mask, other=0.0)
    col_vals = tl.load(x_ptr + base + col_offs, mask=col_mask, other=0.0)
    
    diff = row_vals[:, None] - col_vals[None, :]
    causal = row_offs[:, None] >= col_offs[None, :]
    diff = tl.where(causal, diff, float('-inf'))
    result = libdevice.exp(diff)
    
    rc_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + out_base + row_offs[:, None] * T + col_offs[None, :], result, mask=rc_mask)


# ============================================================
# Kernel 4: decay_states computation
# decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def decay_states_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    
    # offs indexes into [B, H, C, L] flat
    l_idx = offs % L
    c_idx = (offs // L) % C
    h_idx = (offs // (L * C)) % H
    b_idx = offs // (L * C * H)
    
    # last element index in L dim
    last_idx = b_idx * H * C * L + h_idx * C * L + c_idx * L + (L - 1)
    cur_idx = b_idx * H * C * L + h_idx * C * L + c_idx * L + l_idx
    
    last_val = tl.load(cumsum_ptr + last_idx, mask=mask, other=0.0)
    cur_val = tl.load(cumsum_ptr + cur_idx, mask=mask, other=0.0)
    
    result = libdevice.exp(last_val - cur_val)
    tl.store(out_ptr + offs, result, mask=mask)


# ============================================================
# Main kernel_function
# ============================================================
def kernel_function(p_a, p_b, p_c, x):
    B = p_a.shape[0]   # 2048
    seq_len = p_a.shape[1]  # 128
    H = p_a.shape[2]   # 8
    d_state = p_b.shape[3]  # 16
    d_head = x.shape[3]  # 64
    block_len = 64
    C = seq_len // block_len  # 2
    L = block_len  # 64
    
    device = p_a.device
    dtype = p_a.dtype
    
    # Step 1: Reshape inputs into blocks
    # X_blocks: [B, C, L, H, d_head]
    X_blocks = x.reshape(B, C, L, H, d_head)
    # A_blocks: [B, C, L, H]
    A_blocks = p_a.reshape(B, C, L, H)
    # B_blocks: [B, C, L, H, d_state]
    B_blocks = p_b.reshape(B, C, L, H, d_state)
    # C_blocks: [B, C, L, H, d_state]
    C_blocks = p_c.reshape(B, C, L, H, d_state)
    
    # Step 2: A_blocks permute [B, C, L, H] -> [B, H, C, L]
    A_perm = A_blocks.permute(0, 3, 1, 2).contiguous()
    
    # Step 3: A_cumsum [B, H, C, L]
    A_cumsum = torch.cumsum(A_perm, dim=-1)
    
    # Step 4: Compute L (segsum + exp) for diagonal blocks
    # segsum of A_blocks (shape [B, H, C, L]) -> [B, H, C, L, L]
    # segsum[i,j] = cumsum[i] - cumsum[j], masked to tril
    L_mat = torch.empty(B, H, C, L, L, device=device, dtype=dtype)
    
    total_bhc = B * H * C
    grid_segsum = (total_bhc,)
    segsum_exp_kernel[grid_segsum](
        A_cumsum, L_mat,
        B, H, C, L,
    )
    
    # Step 5: Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_mat, X_blocks)
    # This is a complex einsum. Let's compute it step by step.
    # L_mat: [B, H, C, L, L] (L[b,h,c,l,s] = exp(segsum))
    # Y_diag[b,c,l,h,p] = sum_s sum_n C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    
    # Rearrange for batched matmul:
    # For each (b, h, c):
    #   weighted_X[l, p] = sum_s L[l,s] * X[s, p]  -> [L, L] x [L, d_head] = [L, d_head]
    #   then Y_diag[l, p] = sum_n C[l, n] * (sum_s B[s, n] * weighted_X_s[l, p])
    
    # Actually let's do it as:
    # Step 5a: BX[b,c,h,s,p] = sum_s (already indexed) -> B_blocks * X_blocks summed over n? No.
    # Let me think more carefully.
    # Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    # = sum_s L[b,h,c,l,s] * X[b,c,s,h,p] * (sum_n C[b,c,l,h,n] * B[b,c,s,h,n])
    
    # CB[b,c,l,s,h] = sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
    # C_blocks: [B, C, L, H, N], B_blocks: [B, C, L, H, N]
    # CB = einsum("bclhn,bcshn->bclsh", C_blocks, B_blocks)
    
    # Reshape for batched matmul:
    # C_blocks: [B*C*H, L, N]
    # B_blocks: [B*C*H, L, N] -> [B*C*H, N, L]
    # CB = bmm -> [B*C*H, L, L] -> reshape [B, C, H, L, L]
    
    C_flat = C_blocks.permute(0, 1, 3, 2, 4).reshape(B * C * H, L, d_state)  # [B*C*H, L, N]
    B_flat = B_blocks.permute(0, 1, 3, 2, 4).reshape(B * C * H, L, d_state)  # [B*C*H, L, N]
    
    # CB[b*c*h, l, s] = sum_n C[l,n] * B[s,n]
    CB = torch.bmm(C_flat, B_flat.transpose(1, 2))  # [B*C*H, L, L]
    CB = CB.reshape(B, C, H, L, L)  # [B, C, H, L, L]
    
    # Now L_mat: [B, H, C, L, L] -> permute to [B, C, H, L, L]
    L_mat_perm = L_mat.permute(0, 2, 1, 3, 4)  # [B, C, H, L, L]
    
    # Element-wise multiply CB * L_mat_perm: [B, C, H, L, L]
    CBL = CB * L_mat_perm  # [B, C, H, L, L]
    
    # Y_diag[b,c,l,h,p] = sum_s CBL[b,c,h,l,s] * X[b,c,s,h,p]
    # Reshape: CBL -> [B*C*H, L, L], X_blocks -> [B*C*H, L, d_head]
    CBL_flat = CBL.reshape(B * C * H, L, L)
    X_flat = X_blocks.permute(0, 1, 3, 2, 4).reshape(B * C * H, L, d_head)  # [B*C*H, L, d_head]
    
    Y_diag_flat = torch.bmm(CBL_flat, X_flat)  # [B*C*H, L, d_head]
    Y_diag = Y_diag_flat.reshape(B, C, H, L, d_head).permute(0, 1, 3, 2, 4)  # [B, C, L, H, d_head]
    
    # Step 6: Compute intra-chunk states
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)  [B, H, C, L]
    decay_states = torch.empty_like(A_cumsum)
    numel_decay = B * H * C * L
    grid_decay = (triton.cdiv(numel_decay, 1024),)
    decay_states_kernel[grid_decay](
        A_cumsum, decay_states,
        B, H, C, L,
        numel_decay,
    )
    
    # states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
    # = sum_{l} B[b,c,l,h,n] * decay[b,h,c,l] * X[b,c,l,h,p]
    # For each (b, c, h): states[p, n] = sum_l (decay[l] * X[l, p]) * B[l, n]
    # = (decay * X)^T @ B  where decay*X: [L, d_head], B: [L, d_state]
    
    # decay_states: [B, H, C, L] -> [B, C, H, L]
    decay_perm = decay_states.permute(0, 2, 1, 3)  # [B, C, H, L]
    
    # X_blocks: [B, C, L, H, d_head] -> [B*C*H, L, d_head]
    # decay_perm: [B, C, H, L] -> [B*C*H, L, 1]
    decay_flat = decay_perm.reshape(B * C * H, L, 1)
    
    # weighted_X: [B*C*H, L, d_head]
    weighted_X = X_flat * decay_flat  # [B*C*H, L, d_head]
    
    # B_flat: [B*C*H, L, d_state]
    # states[b*c*h, p, n] = sum_l weighted_X[l, p] * B[l, n]
    # = weighted_X^T @ B_flat -> [B*C*H, d_head, d_state]
    states_flat = torch.bmm(weighted_X.transpose(1, 2), B_flat)  # [B*C*H, d_head, d_state]
    states = states_flat.reshape(B, C, H, d_head, d_state).permute(0, 1, 2, 3, 4)
    # states: [B, C, H, d_head, d_state]
    # Need shape [B, C, H, d_head, d_state] -> reference uses [B, C, H, d_head, d_state]
    # Actually reference einsum: "bclhn,bhcl,bclhp->bchpn" -> [B, C, H, d_head, d_state]
    # Our states_flat: [B*C*H, d_head, d_state] -> reshape [B, C, H, d_head, d_state]
    
    # Step 7: Inter-chunk recurrence
    # initial_states = zeros_like(states[:, :1])  [B, 1, H, d_head, d_state]
    initial_states = torch.zeros(B, 1, H, d_head, d_state, device=device, dtype=dtype)
    
    # states = cat([initial_states, states], dim=1)  [B, C+1, H, d_head, d_state]
    states_cat = torch.cat([initial_states, states], dim=1)  # [B, 3, H, d_head, d_state]
    
    # A_tail = A_cumsum[:, :, :, -1]  [B, H, C]
    # select(cumsum, dim=3, index=-1) -> [B, H, C]
    A_tail = A_cumsum[:, :, :, -1]  # [B, H, C]
    
    # pad A_tail: [B, H, C] -> [B, H, C+1] with 0 prepended
    A_tail_padded = torch.nn.functional.pad(A_tail, (1, 0), value=0.0)  # [B, H, 3]
    
    # cumsum of padded A_tail
    A_tail_cumsum = torch.cumsum(A_tail_padded, dim=-1)  # [B, H, 3]
    
    # segsum of A_tail_cumsum -> [B, H, 3, 3]
    T_small = C + 1  # 3
    L_inter = torch.empty(B, H, T_small, T_small, device=device, dtype=dtype)
    
    grid_small = (B * H,)
    segsum_exp_small_kernel[grid_small](
        A_tail_cumsum, L_inter,
        B, H, T_small,
    )
    
    # new_states = einsum("bhij,bjhpn->bihpn", L_inter, states_cat)
    # For each (b, h): new_states[i, p, n] = sum_j L_inter[i, j] * states_cat[j, p, n]
    # states_cat: [B, C+1, H, d_head, d_state] -> [B, H, C+1, d_head*d_state]
    # L_inter: [B, H, T, T]
    
    states_cat_perm = states_cat.permute(0, 2, 1, 3, 4)  # [B, H, C+1, d_head, d_state]
    states_cat_flat = states_cat_perm.reshape(B * H, T_small, d_head * d_state)
    L_inter_flat = L_inter.reshape(B * H, T_small, T_small)
    
    new_states_flat = torch.bmm(L_inter_flat, states_cat_flat)  # [B*H, T_small, d_head*d_state]
    new_states = new_states_flat.reshape(B, H, T_small, d_head, d_state)
    new_states = new_states.permute(0, 2, 1, 3, 4)  # [B, T_small, H, d_head, d_state]
    
    # select last chunk: new_states[:, -1] -> [B, H, d_head, d_state]
    # Actually the reference returns select(new_states, dim=1, index=-1)
    # which gives [B, H, d_head, d_state]
    final_state = new_states[:, -1]  # [B, H, d_head, d_state]
    
    return final_state


def get_inputs():
    B = 2048
    seq_len = 128
    H = 8
    d_head = 64
    d_state = 16
    
    p_a = torch.randn(B, seq_len, H, device='cuda', dtype=torch.float32)
    p_b = torch.randn(B, seq_len, H, d_state, device='cuda', dtype=torch.float32)
    p_c = torch.randn(B, seq_len, H, d_state, device='cuda', dtype=torch.float32)
    x = torch.randn(B, seq_len, H, d_head, device='cuda', dtype=torch.float32)
    
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
        p_a, p_b, p_c, x = inputs
        B = p_a.shape[0]
        seq_len = p_a.shape[1]
        H = p_a.shape[2]
        d_state = p_b.shape[3]
        d_head = x.shape[3]
        block_len = 64
        C = seq_len // block_len
        L = block_len
        
        # Reference implementation
        def segsum_ref(x):
            T = x.size(-1)
            x_cumsum = torch.cumsum(x, dim=-1)
            x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
            return x_segsum
        
        X_blocks = x.reshape(B, C, L, H, d_head)
        A_blocks = p_a.reshape(B, C, L, H)
        B_blocks = p_b.reshape(B, C, L, H, d_state)
        C_blocks = p_c.reshape(B, C, L, H, d_state)
        
        A_blocks_perm = A_blocks.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A_blocks_perm, dim=-1)
        
        L_ref = torch.exp(segsum_ref(A_blocks_perm))
        Y_diag_ref = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
        
        decay_states_ref = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states_ref = torch.einsum("bclhn,bhcl,bclhp->bchpn",
        
        initial_states_ref = torch.zeros_like(states_ref[:, :1])
        states_cat_ref = torch.cat([initial_states_ref, states_ref], dim=1)
        
        A_tail_ref = A_cumsum[:, :, :, -1]
        A_tail_padded_ref = torch.nn.functional.pad(A_tail_ref, (1, 0), value=0.0)
        A_tail_cumsum_ref = torch.cumsum(A_tail_padded_ref, dim=-1)
        
        L_inter_ref = torch.exp(segsum_ref(A_tail_cumsum_ref))
        new_states_ref = torch.einsum("bhij,bjhpn->bihpn", L_inter_ref, states_cat_ref)
        final_state_ref = new_states_ref[:, -1]
        
        assert torch.allclose(output, final_state_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - final_state_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")