import torch
import triton
import triton.language as tl

# Kernel 1: Fused permute + cumsum for A_blocks
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
def fused_permute_cumsum_kernel(
    in_ptr,   # [B, 128, 8] = [B, C*L, H]
    B: tl.constexpr,
    xnumel: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length L
    # Total rows = B * H * C
    pid = tl.program_id(0)
    total_rows = B * H * C
    
    row_idx = pid * XBLOCK + tl.arange(0, XBLOCK)
    row_mask = row_idx < total_rows
    
    # Decode row_idx -> (b, h, c)
    b_idx = row_idx // (H * C)
    hc_idx = row_idx % (H * C)
    h_idx = hc_idx // C
    c_idx = hc_idx % C
    
    # Load L elements and compute cumsum
    # in_ptr layout: [B, C*L, H] with strides [C*L*H, H, 1]
    # Element at [b, c*L + l, h] = in_ptr[b * C*L*H + (c*L+l)*H + h]
    
    # We need to do cumsum over L dimension
    # For each row, load all L elements
    l_offsets = tl.arange(0, L)
    
    # Compute input indices: in_ptr[b, c*L + l, h]
    in_idx = (b_idx[:, None] * (C * L * H) + 
              (c_idx[:, None] * L + l_offsets[None, :]) * H + 
              h_idx[:, None])
    
    vals = tl.load(in_ptr + in_idx, mask=row_mask[:, None], other=0.0)
    
    # Compute cumsum along L dimension (axis=1 here)
    # Manual prefix sum
    acc = tl.cumsum(vals, axis=1)
    
    # out_ptr layout: [B, H, C, L] with strides [H*C*L, C*L, L, 1]
    out_idx = (b_idx[:, None] * (H * C * L) + 
               h_idx[:, None] * (C * L) + 
               c_idx[:, None] * L + 
               l_offsets[None, :])
    
    tl.store(out_ptr + out_idx, acc, mask=row_mask[:, None])


# Kernel 2: Compute segsum mask and exp
# segsum(x)[..., i, j] = cumsum[..., i] - cumsum[..., j], masked to lower triangular
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=4),
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=8),
        triton.Config({'BLOCK': 512}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def segsum_exp_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    # numel = B * H * C * L * L
    # Decode: offsets -> (b, h, c, i, j)
    tmp = offsets
    j = tmp % L
    tmp = tmp // L
    i = tmp % L
    tmp = tmp // L
    c = tmp % C
    tmp = tmp // C
    h = tmp % H
    b = tmp // H
    
    # Load cumsum[b, h, c, i] and cumsum[b, h, c, j]
    cs_i = tl.load(cumsum_ptr + b * (H * C * L) + h * (C * L) + c * L + i, mask=mask, other=0.0)
    cs_j = tl.load(cumsum_ptr + b * (H * C * L) + h * (C * L) + c * L + j, mask=mask, other=0.0)
    
    diff = cs_i - cs_j
    
    # Apply lower triangular mask: valid if i >= j (j <= i)
    is_lower = i >= j
    result = tl.where(is_lower, diff, float('-inf'))
    result = tl.exp(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)


# Kernel 3: Compute decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=2),
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
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
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    # Decode: (b, h, c, l)
    tmp = offsets
    l = tmp % L
    tmp = tmp // L
    c = tmp % C
    tmp = tmp // C
    h = tmp % H
    b = tmp // H
    
    base = b * (H * C * L) + h * (C * L) + c * L
    
    # Load cumsum[b, h, c, L-1] (last element)
    cs_last = tl.load(cumsum_ptr + base + (L - 1), mask=mask, other=0.0)
    # Load cumsum[b, h, c, l]
    cs_l = tl.load(cumsum_ptr + base + l, mask=mask, other=0.0)
    
    result = tl.exp(cs_last - cs_l)
    tl.store(out_ptr + offsets, result, mask=mask)


# Kernel 4: Compute inter-chunk decay matrix and apply
# decay_chunk[b, h, i, j] = exp(segsum(A_cumsum_last)[b, h, i, j])
# where A_cumsum_last = cumsum of last elements of each chunk
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=2),
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def inter_chunk_decay_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # C_pad = C + 1 (padded)
    C_pad = C + 1
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    # numel = B * H * C_pad * C_pad
    tmp = offsets
    j = tmp % C_pad
    tmp = tmp // C_pad
    i = tmp % C_pad
    tmp = tmp // C_pad
    h = tmp % H
    b = tmp // H
    
    # Load cumsum_last[b, h, i] and cumsum_last[b, h, j]
    # cumsum_last is [B, H, C_pad] with 0 prepended
    cs_i = tl.load(cumsum_last_ptr + b * (H * C_pad) + h * C_pad + i, mask=mask, other=0.0)
    cs_j = tl.load(cumsum_last_ptr + b * (H * C_pad) + h * C_pad + j, mask=mask, other=0.0)
    
    diff = cs_i - cs_j
    is_lower = i >= j
    result = tl.where(is_lower, diff, float('-inf'))
    result = tl.exp(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)


# Kernel 5: Final add and reshape (pointwise)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def kernel_function(p_a, p_b, p_c, x):
    """
    
    """
    B = p_a.shape[0]
    seq_len = p_a.shape[1]
    H = p_a.shape[2]
    d_state = p_b.shape[3]
    d_head = x.shape[3]
    block_len = 64
    C = seq_len // block_len  # number of chunks = 2
    L = block_len  # 64
    
    device = p_a.device
    dtype = p_a.dtype
    
    # Step 1: Reshape inputs into blocks
    # X_blocks: [B, C, L, H, d_head]
    X_blocks = x.reshape(B, C, L, H, d_head)
    # A_blocks: [B, C, L, H] -> permute to [B, H, C, L]
    A_blocks_raw = p_a.reshape(B, C, L, H)
    # B_blocks: [B, C, L, H, d_state]
    B_blocks = p_b.reshape(B, C, L, H, d_state)
    # C_blocks: [B, C, L, H, d_state]
    C_blocks = p_c.reshape(B, C, L, H, d_state)
    
    # Step 2: Permute A and compute cumsum
    # A_blocks_perm: [B, H, C, L]
    A_blocks_perm = A_blocks_raw.permute(0, 3, 1, 2).contiguous()
    
    # Compute A_cumsum using Triton kernel
    A_cumsum = torch.empty(B, H, C, L, device=device, dtype=dtype)
    total_rows = B * H * C
    
    # Use the fused permute+cumsum kernel
    # Actually A_blocks_perm is already permuted, just need cumsum
    # Let's use torch.cumsum for correctness and focus Triton on the complex parts
    A_cumsum = torch.cumsum(A_blocks_perm, dim=-1)
    
    # Step 3: Compute L matrix (diagonal block decay) using segsum
    # segsum(A_blocks_perm)[b, h, c, i, j] = A_cumsum[b,h,c,i] - A_cumsum[b,h,c,j], lower tri
    # L_mat: [B, H, C, L, L]
    L_mat = torch.empty(B, H, C, L, L, device=device, dtype=dtype)
    numel_L = B * H * C * L * L
    grid_L = lambda meta: (triton.cdiv(numel_L, meta['BLOCK']),)
    segsum_exp_kernel[grid_L](
        A_cumsum, L_mat,
        B, H, C, L, numel_L,
    )
    
    # Step 4: Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_mat, X_blocks)
    # Rearrange for einsum:
    # C_blocks: [B, C, L, H, N]  (N=d_state)
    # B_blocks: [B, C, S, H, N]  (S=L, same dim)
    # L_mat: [B, H, C, L, S]
    # X_blocks: [B, C, S, H, P]  (P=d_head)
    # Output: [B, C, L, H, P]
    
    # Reorder L_mat to [B, C, H, L, S] for easier einsum
    # einsum: bclhn, bcshn, bhcls, bcshp -> bclhp
    # = sum over s,n: C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    
    # Compute step by step:
    # First: CB[b,c,l,s,h] = sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
    # Then: CBL[b,c,l,s,h] = CB[b,c,l,s,h] * L[b,h,c,l,s]
    # Then: Y_diag[b,c,l,h,p] = sum_s CBL[b,c,l,s,h] * X[b,c,s,h,p]
    
    # Use einsum directly (these are matmul-like operations)
    # L_mat needs to be [B, C, H, L, S] -> permute from [B, H, C, L, S]
    L_mat_bcls = L_mat.permute(0, 2, 1, 3, 4)  # [B, C, H, L, S] -- wait
    # L_mat is [B, H, C, L, S], need [B, C, L, S, H] for broadcasting
    # Let's just use torch.einsum
    
    # Y_diag: einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_mat, X_blocks)
    # This is complex - break it down
    # Step A: inner product over n: C_blocks * B_blocks -> [B, C, L, S, H]
    # C_blocks: [B, C, L, H, N], B_blocks: [B, C, S, H, N]
    # CB[b,c,l,s,h] = sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
    CB = torch.einsum('bclhn,bcshn->bclsh', C_blocks, B_blocks)  # [B, C, L, S, H]
    
    # L_mat: [B, H, C, L, S] -> need [B, C, L, S, H]
    L_mat_bclsh = L_mat.permute(0, 2, 3, 4, 1)  # [B, C, L, S, H]
    
    # CBL[b,c,l,s,h] = CB[b,c,l,s,h] * L_mat[b,c,l,s,h]
    CBL = CB * L_mat_bclsh  # [B, C, L, S, H]
    
    # Y_diag[b,c,l,h,p] = sum_s CBL[b,c,l,s,h] * X[b,c,s,h,p]
    # CBL: [B, C, L, S, H], X_blocks: [B, C, S, H, P]
    # Need to sum over s
    Y_diag = torch.einsum('bclsh,bcshp->bclhp', CBL, X_blocks)  # [B, C, L, H, P]
    
    # Step 5: Compute intra-chunk states
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    # A_cumsum: [B, H, C, L]
    A_cumsum_last = A_cumsum[:, :, :, -1:]  # [B, H, C, 1]
    decay_states_vals = torch.empty_like(A_cumsum)
    numel_ds = B * H * C * L
    grid_ds = lambda meta: (triton.cdiv(numel_ds, meta['BLOCK']),)
    decay_states_kernel[grid_ds](
        A_cumsum, decay_states_vals,
        B, H, C, L, numel_ds,
    )
    # decay_states: [B, H, C, L]
    
    # states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
    # B_blocks: [B, C, L, H, N]
    # decay_states: [B, H, C, L]
    # X_blocks: [B, C, L, H, P]
    # Output: [B, C, H, P, N]
    
    # decay_states needs to be [B, C, L, H] for broadcasting
    decay_states_bcl = decay_states_vals.permute(0, 2, 3, 1)  # [B, C, L, H]
    
    # BX[b,c,l,h,p,n] = B[b,c,l,h,n] * X[b,c,l,h,p] * decay[b,c,l,h]
    # states[b,c,h,p,n] = sum_l BX[b,c,l,h,p,n]
    states = torch.einsum('bclhn,bclh,bclhp->bchpn', B_blocks, decay_states_bcl, X_blocks)
    # states: [B, C, H, P, N]
    
    # Step 6: Inter-chunk recurrence
    # initial_states = zeros_like(states[:, :1])
    initial_states = torch.zeros_like(states[:, :1])  # [B, 1, H, P, N]
    states_cat = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N]
    
    # Compute decay_chunk using segsum on A_cumsum last values
    # A_cumsum_last_vals: [B, H, C] = A_cumsum[:, :, :, -1]
    A_cumsum_last_vals = A_cumsum[:, :, :, -1]  # [B, H, C]
    
    # select last element of each chunk, pad with 0 at start
    # pad: [B, H, C+1] with 0 prepended
    A_last_padded = torch.cat([
        torch.zeros(B, H, 1, device=device, dtype=dtype),
    ], dim=-1)  # [B, H, C+1]
    
    # cumsum of padded
    A_last_cumsum = torch.cumsum(A_last_padded, dim=-1)  # [B, H, C+1]
    
    # decay_chunk[b, h, i, j] = exp(segsum(A_last_cumsum)[b, h, i, j])
    # segsum: [B, H, C+1, C+1]
    C_pad = C + 1
    decay_chunk = torch.empty(B, H, C_pad, C_pad, device=device, dtype=dtype)
    numel_dc = B * H * C_pad * C_pad
    grid_dc = lambda meta: (triton.cdiv(numel_dc, meta['BLOCK']),)
    inter_chunk_decay_kernel[grid_dc](
        A_last_cumsum, decay_chunk,
        B, H, C, numel_dc,
    )
    
    # Y_off = einsum("bhij,bjhpn->bihpn", decay_chunk[:, :, 1:, :], states_cat)
    # decay_chunk: [B, H, C+1, C+1], slice [1:, :] -> [B, H, C, C+1]
    # states_cat: [B, C+1, H, P, N]
    # Output: [B, C, H, P, N]
    
    decay_chunk_slice = decay_chunk[:, :, 1:, :]  # [B, H, C, C+1]
    
    # Y_off[b,i,h,p,n] = sum_j decay_chunk[b,h,i,j] * states_cat[b,j,h,p,n]
    Y_off_states = torch.einsum('bhij,bjhpn->bihpn', decay_chunk_slice, states_cat)
    # Y_off_states: [B, C, H, P, N]
    
    # Now compute Y_off contribution to output
    # Y_off[b,c,l,h,p] = sum_n C_blocks[b,c,l,h,n] * Y_off_states[b,c,h,p,n]
    # C_blocks: [B, C, L, H, N], Y_off_states: [B, C, H, P, N]
    
    # Also need decay for within-chunk: exp(A_cumsum[b,h,c,l])
    # Actually the formula is:
    # Y_off[b,c,l,h,p] = exp(A_cumsum[b,h,c,l]) * sum_n C[b,c,l,h,n] * Y_off_states[b,c,h,p,n]
    
    # exp(A_cumsum): [B, H, C, L] -> [B, C, L, H]
    exp_A_cumsum = torch.exp(A_cumsum).permute(0, 2, 3, 1)  # [B, C, L, H]
    
    # C_blocks * exp_A_cumsum: [B, C, L, H, N] * [B, C, L, H, 1]
    C_scaled = C_blocks * exp_A_cumsum.unsqueeze(-1)  # [B, C, L, H, N]
    
    # Y_off[b,c,l,h,p] = sum_n C_scaled[b,c,l,h,n] * Y_off_states[b,c,h,p,n]
    Y_off = torch.einsum('bclhn,bchpn->bclhp', C_scaled, Y_off_states)  # [B, C, L, H, P]
    
    # Step 7: Add Y_diag and Y_off
    numel_out = B * C * L * H * d_head
    Y = torch.empty(B, C, L, H, d_head, device=device, dtype=dtype)
    grid_add = lambda meta: (triton.cdiv(numel_out, meta['BLOCK']),)
    add_kernel[grid_add](
        Y_diag, Y_off, Y,
        numel_out,
    )
    
    # Reshape to [B, seq_len, H, d_head]
    Y = Y.reshape(B, seq_len, H, d_head)
    
    return Y


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


def reference_implementation(p_a, p_b, p_c, x):
    """Reference PyTorch implementation."""
    B = p_a.shape[0]
    seq_len = p_a.shape[1]
    H = p_a.shape[2]
    d_state = p_b.shape[3]
    d_head = x.shape[3]
    block_len = 64
    C = seq_len // block_len
    L = block_len
    
    def segsum(t):
        T = t.size(-1)
        x_cumsum = torch.cumsum(t, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=t.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    X_blocks = x.reshape(B, C, L, H, d_head)
    A_blocks = p_a.reshape(B, C, L, H)
    B_blocks = p_b.reshape(B, C, L, H, d_state)
    C_blocks = p_c.reshape(B, C, L, H, d_state)
    
    A_blocks_perm = A_blocks.permute(0, 3, 1, 2)  # [B, H, C, L]
    A_cumsum = torch.cumsum(A_blocks_perm, dim=-1)
    
    # Diagonal blocks
    L_mat = torch.exp(segsum(A_blocks_perm))  # [B, H, C, L, L]
    
    # Y_diag
    Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C_blocks, B_blocks, L_mat, X_blocks)
    
    # Intra-chunk states
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, H, C, L]
    decay_states_bcl = decay_states.permute(0, 2, 3, 1)  # [B, C, L, H]
    states = torch.einsum('bclhn,bclh,bclhp->bchpn', B_blocks, decay_states_bcl, X_blocks)
    
    # Inter-chunk
    initial_states = torch.zeros_like(states[:, :1])
    states_cat = torch.cat([initial_states, states], dim=1)
    
    A_last = A_cumsum[:, :, :, -1]  # [B, H, C]
    A_last_padded = torch.cat([torch.zeros(B, H, 1, device=p_a.device, dtype=p_a.dtype), A_last], dim=-1)
    A_last_cumsum = torch.cumsum(A_last_padded, dim=-1)
    
    decay_chunk = torch.exp(segsum(A_last_cumsum))  # [B, H, C+1, C+1]
    decay_chunk_slice = decay_chunk[:, :, 1:, :]  # [B, H, C, C+1]
    
    Y_off_states = torch.einsum('bhij,bjhpn->bihpn', decay_chunk_slice, states_cat)
    
    exp_A_cumsum = torch.exp(A_cumsum).permute(0, 2, 3, 1)  # [B, C, L, H]
    C_scaled = C_blocks * exp_A_cumsum.unsqueeze(-1)
    Y_off = torch.einsum