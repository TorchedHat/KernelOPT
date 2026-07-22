import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Kernel 1: Cumsum with permute (A_blocks -> A_cumsum)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 2}, num_warps=4),
        triton.Config({'XBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def cumsum_permute_kernel(in_ptr, out_ptr, xnumel, L: tl.constexpr, XBLOCK: tl.constexpr):
    # in_ptr: [B, C, L, H] with strides [C*L*H, L*H, H, 1]
    # We want to permute to [B, H, C, L] and cumsum over last dim
    # B=2048, C=2, L=64, H=8
    # xnumel = B*H*C = 2048*8*2 = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # xindex encodes (b, h, c) in [B, H, C] layout
    # b = xindex // (H*C), h = (xindex // C) % H, c = xindex % C
    H = 8
    C = 2
    b = xindex // (H * C)
    h = (xindex // C) % H
    c = xindex % C
    
    # Load L elements and compute cumsum
    # in_ptr layout: [B, C, L, H] -> index = b*C*L*H + c*L*H + l*H + h
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for l in range(L):
        in_idx = b * (C * L * H) + c * (L * H) + l * H + h
        val = tl.load(in_ptr + in_idx, mask=xmask, other=0.0)
        acc = acc + val
        # out_ptr layout: [B, H, C, L] -> index = b*H*C*L + h*C*L + c*L + l
        out_idx = b * (H * C * L) + h * (C * L) + c * L + l
        tl.store(out_ptr + out_idx, acc, mask=xmask)


# ============================================================
# Kernel 2: Segsum computation -> L matrix [B, H, C, L, L]
# segsum(x)[i,j] = cumsum[i] - cumsum[j], masked with tril, then exp
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 32}, num_warps=4),
        triton.Config({'BLOCK_L': 16}, num_warps=2),
        triton.Config({'BLOCK_L': 8}, num_warps=2),
    ],
    key=['L'],
)
@triton.jit
def segsum_exp_kernel(
    B, H, C, L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Each program handles one (b, h, c) triple and one row i
    pid = tl.program_id(0)
    total = B * H * C * L
    if pid >= total:
        return
    
    b = pid // (H * C * L)
    rem = pid % (H * C * L)
    h = rem // (C * L)
    rem2 = rem % (C * L)
    c = rem2 // L
    i = rem2 % L
    
    # cumsum base for this (b, h, c)
    base = b * (H * C * L) + h * (C * L) + c * L
    
    # Load cumsum[i]
    cs_i = tl.load(cumsum_ptr + base + i)
    
    # For each j in [0, L), compute cs_i - cs_j, mask with j <= i, exp
    for jb in range(0, L, BLOCK_L):
        j_offs = jb + tl.arange(0, BLOCK_L)
        j_mask = j_offs < L
        cs_j = tl.load(cumsum_ptr + base + j_offs, mask=j_mask, other=0.0)
        diff = cs_i - cs_j
        # tril mask: j <= i
        tril_mask = j_offs <= i
        val = tl.where(tril_mask & j_mask, libdevice.exp(diff), 0.0)
        # out[b, h, c, i, j]
        out_base = b * (H * C * L * L) + h * (C * L * L) + c * (L * L) + i * L
        tl.store(out_ptr + out_base + j_offs, val, mask=j_mask)


# ============================================================
# Kernel 3: Y_diag einsum
# Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# = sum_s L[b,h,c,l,s] * (sum_n C[b,c,l,h,n]*B[b,c,s,h,n]) * X[b,c,s,h,p]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 32}, num_warps=8),
    ],
    key=['L', 'N', 'P'],
)
@triton.jit
def ydiag_kernel(
    B_dim, C_dim, L: tl.constexpr, H, N: tl.constexpr, P: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # Each program: one (b, c, l, h) -> compute Y[b,c,l,h,:]
    pid = tl.program_id(0)
    total = B_dim * C_dim * L * H
    if pid >= total:
        return
    
    b = pid // (C_dim * L * H)
    rem = pid % (C_dim * L * H)
    c = rem // (L * H)
    rem2 = rem % (L * H)
    l = rem2 // H
    h = rem2 % H
    
    # C[b,c,l,h,:] shape [N]
    C_base = b * (C_dim * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
    
    # Lmat[b,h,c,l,:] shape [L] (the row for position l)
    Lmat_base = b * (H * C_dim * L * L) + h * (C_dim * L * L) + c * (L * L) + l * L
    
    # Y[b,c,l,h,:] shape [P]
    Y_base = b * (C_dim * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
    
    # Load C[b,c,l,h,:] - shape [N]
    n_offs = tl.arange(0, N)
    c_vals = tl.load(C_ptr + C_base + n_offs)  # [N]
    
    # Accumulate over p
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)
    
    for pb in range(0, P, BLOCK_P):
        p_offs = pb + tl.arange(0, BLOCK_P)
        p_mask = p_offs < P
        acc = tl.zeros([BLOCK_P], dtype=tl.float32)
        
        for sb in range(0, L, BLOCK_S):
            s_offs = sb + tl.arange(0, BLOCK_S)
            s_mask = s_offs < L
            
            # Load Lmat[b,h,c,l,s] for s in block
            lmat_vals = tl.load(Lmat_ptr + Lmat_base + s_offs, mask=s_mask, other=0.0)  # [BLOCK_S]
            
            # Load B[b,c,s,h,:] for s in block -> [BLOCK_S, N]
            B_base_s = b * (C_dim * L * H * N) + c * (L * H * N) + h * N
            # B[b,c,s,h,n] = B_ptr[b*C*L*H*N + c*L*H*N + s*H*N + h*N + n]
            s_full = sb + tl.arange(0, BLOCK_S)
            b_ptrs = B_ptr + b * (C_dim * L * H * N) + c * (L * H * N) + s_full[:, None] * (H * N) + h * N + n_offs[None, :]
            b_vals = tl.load(b_ptrs, mask=s_mask[:, None] & (n_offs[None, :] < N), other=0.0)  # [BLOCK_S, N]
            
            # CB[s] = dot(c_vals, b_vals[s,:]) for each s
            cb = tl.sum(c_vals[None, :] * b_vals, axis=1)  # [BLOCK_S]
            
            # weighted = lmat_vals * cb  [BLOCK_S]
            weighted = lmat_vals * cb  # [BLOCK_S]
            
            # Load X[b,c,s,h,p] for s in block, p in block -> [BLOCK_S, BLOCK_P]
            x_ptrs = X_ptr + b * (C_dim * L * H * P) + c * (L * H * P) + s_full[:, None] * (H * P) + h * P + p_offs[None, :]
            x_vals = tl.load(x_ptrs, mask=s_mask[:, None] & p_mask[None, :], other=0.0)  # [BLOCK_S, BLOCK_P]
            
            # acc[p] += sum_s weighted[s] * x_vals[s, p]
            acc += tl.sum(weighted[:, None] * x_vals, axis=0)
        
        tl.store(Y_ptr + Y_base + p_offs, acc, mask=p_mask)


# ============================================================
# Kernel 4: States computation
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay[b,h,c,l] * X[b,c,l,h,p]
# where decay[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_N': 8}, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_N': 16}, num_warps=8),
    ],
    key=['L', 'N', 'P'],
)
@triton.jit
def states_kernel(
    B_dim, C_dim, L: tl.constexpr, H, N: tl.constexpr, P: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program: one (b, c, h, p)
    pid = tl.program_id(0)
    total = B_dim * C_dim * H * P
    if pid >= total:
        return
    
    b = pid // (C_dim * H * P)
    rem = pid % (C_dim * H * P)
    c = rem // (H * P)
    rem2 = rem % (H * P)
    h = rem2 // P
    p = rem2 % P
    
    # Load A_cumsum last value: cumsum[b,h,c,-1]
    cs_last = tl.load(cumsum_ptr + b * (H * C_dim * L) + h * (C_dim * L) + c * L + (L - 1))
    
    # states[b,c,h,p,:] = sum_l decay[l] * X[b,c,l,h,p] * B[b,c,l,h,:]
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    n_offs = tl.arange(0, BLOCK_N)
    
    for nb in range(0, N, BLOCK_N):
        n_offs_cur = nb + tl.arange(0, BLOCK_N)
        n_mask = n_offs_cur < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for lb in range(0, L, BLOCK_L):
            l_offs = lb + tl.arange(0, BLOCK_L)
            l_mask = l_offs < L
            
            # Load cumsum[b,h,c,l]
            cs_vals = tl.load(cumsum_ptr + b * (H * C_dim * L) + h * (C_dim * L) + c * L + l_offs, mask=l_mask, other=0.0)
            decay = libdevice.exp(cs_last - cs_vals)  # [BLOCK_L]
            
            # Load X[b,c,l,h,p]
            x_vals = tl.load(X_ptr + b * (C_dim * L * H * P) + c * (L * H * P) + l_offs * (H * P) + h * P + p, mask=l_mask, other=0.0)  # [BLOCK_L]
            
            # weighted = decay * x_vals [BLOCK_L]
            weighted = decay * x_vals
            
            # Load B[b,c,l,h,n] for l in block, n in block -> [BLOCK_L, BLOCK_N]
            b_ptrs = B_ptr + b * (C_dim * L * H * N) + c * (L * H * N) + l_offs[:, None] * (H * N) + h * N + n_offs_cur[None, :]
            b_vals = tl.load(b_ptrs, mask=l_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_L, BLOCK_N]
            
            # acc[n] += sum_l weighted[l] * b_vals[l, n]
            acc += tl.sum(weighted[:, None] * b_vals, axis=0)
        
        # states[b,c,h,p,n]
        states_base = b * (C_dim * H * P * N) + c * (H * P * N) + h * (P * N) + p * N
        tl.store(states_ptr + states_base + n_offs_cur, acc, mask=n_mask)


# ============================================================
# Kernel 5: Inter-chunk recurrence segsum
# For the inter-chunk part: segsum over C+1 chunks
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 2}, num_warps=2),
        triton.Config({'XBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def inter_chunk_cumsum_kernel(
    xnumel,
    C_plus1: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # in_ptr has shape [B, H, C+1] where first element is 0
    # We compute cumsum along last dim
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    B_H = xnumel  # B*H
    
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for i in range(C_plus1):
        val = tl.load(in_ptr + xindex * C_plus1 + i, mask=xmask, other=0.0)
        acc = acc + val
        tl.store(out_ptr + xindex * C_plus1 + i, acc, mask=xmask)


# ============================================================
# Kernel 6: Inter-chunk segsum exp matrix [B, H, C+1, C+1]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 4}, num_warps=2),
        triton.Config({'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=8),
    ],
    key=['C_plus1'],
)
@triton.jit
def inter_segsum_exp_kernel(
    B, H, C_plus1: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H * C_plus1
    if pid >= total:
        return
    
    b = pid // (H * C_plus1)
    rem = pid % (H * C_plus1)
    h = rem // C_plus1
    i = rem % C_plus1
    
    base = b * (H * C_plus1) + h * C_plus1
    cs_i = tl.load(cumsum_ptr + base + i)
    
    for jb in range(0, C_plus1, BLOCK_C):
        j_offs = jb + tl.arange(0, BLOCK_C)
        j_mask = j_offs < C_plus1
        cs_j = tl.load(cumsum_ptr + base + j_offs, mask=j_mask, other=0.0)
        diff = cs_i - cs_j
        tril_mask = j_offs <= i
        val = tl.where(tril_mask & j_mask, libdevice.exp(diff), 0.0)
        out_base = b * (H * C_plus1 * C_plus1) + h * (C_plus1 * C_plus1) + i * C_plus1
        tl.store(out_ptr + out_base + j_offs, val, mask=j_mask)


# ============================================================
# Kernel 7: New states from inter-chunk recurrence
# new_states[b,c,h,p,n] = sum_{c2} decay_inter[b,h,c,c2] * states[b,c2,h,p,n]
# where c ranges over [1, C+1) and c2 ranges over [0, C+1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_C2': 4}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_C2': 4}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_C2': 2}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C2': 4}, num_warps=8),
    ],
    key=['N', 'P', 'C_plus1'],
)
@triton.jit
def new_states_kernel(
    B, H, C_dim, P: tl.constexpr, N: tl.constexpr, C_plus1: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C2: tl.constexpr,
):
    # Each program: one (b, c, h, p)
    pid = tl.program_id(0)
    total = B * C_dim * H * P
    if pid >= total:
        return
    
    b = pid // (C_dim * H * P)
    rem = pid % (C_dim * H * P)
    c = rem // (H * P)  # 0-indexed chunk [0, C_dim)
    rem2 = rem % (H * P)
    h = rem2 // P
    p = rem2 % P
    
    # In the inter-chunk matrix, row index for chunk c is c+1 (since we prepended initial state)
    row_idx = c + 1
    
    # decay_inter[b, h, row_idx, :] gives weights for each source state
    decay_base = b * (H * C_plus1 * C_plus1) + h * (C_plus1 * C_plus1) + row_idx * C_plus1
    
    for nb in range(0, N, BLOCK_N):
        n_offs = nb + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for c2b in range(0, C_plus1, BLOCK_C2):
            c2_offs = c2b + tl.arange(0, BLOCK_C2)
            c2_mask = c2_offs < C_plus1
            
            # Load decay weights
            decay_vals = tl.load(decay_inter_ptr + decay_base + c2_offs, mask=c2_mask, other=0.0)  # [BLOCK_C2]
            
            # Load states[b, c2, h, p, n] for c2 in block
            # states shape: [B, C+1, H, P, N]
            s_ptrs = states_ptr + b * (C_plus1 * H * P * N) + c2_offs[:, None] * (H * P * N) + h * (P * N) + p * N + n_offs[None, :]
            s_vals = tl.load(s_ptrs, mask=c2_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_C2, BLOCK_N]
            
            acc += tl.sum(decay_vals[:, None] * s_vals, axis=0)
        
        out_base = b * (C_dim * H * P * N) + c * (H * P * N) + h * (P * N) + p * N
        tl.store(new_states_ptr + out_base + n_offs, acc, mask=n_mask)


# ============================================================
# Main kernel function
# ============================================================
def kernel_function(p_a, p_b, p_c, x):
    """
    
    p_a: [B, seq_len, H] = [2048, 128, 8]
    p_b: [B, seq_len, H, N] = [2048, 128, 8, 16]
    p_c: [B, seq_len, H, N] = [2048, 128, 8, 16]
    x:   [B, seq_len, H, P] = [2048, 128, 8, 64]
    
    Returns: final state [B, H, P, N] = [2048, 8, 64, 16]
    """
    B = p_a.shape[0]
    seq_len = p_a.shape[1]
    H = p_a.shape[2]
    N = p_b.shape[3]
    P = x.shape[3]
    block_len = 64
    C = seq_len // block_len  # number of chunks = 2
    
    device = p_a.device
    dtype = p_a.dtype
    
    # Step 1: Compute A_cumsum [B, H, C, L]
    A_cumsum = torch.empty(B, H, C, block_len, device=device, dtype=dtype)
    xnumel = B * H * C
    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)
    cumsum_permute_kernel[grid](p_a, A_cumsum, xnumel, block_len)
    
    # Step 2: Compute segsum exp matrix L [B, H, C, L, L]
    L_mat = torch.empty(B, H, C, block_len, block_len, device=device, dtype=dtype)
    total_rows = B * H * C * block_len
    grid2 = (total_rows,)
    segsum_exp_kernel[grid2](A_cumsum, L_mat, B, H, C, block_len)
    
    # Step 3: Compute Y_diag
    # Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    # Reshape inputs to [B, C, L, H, N/P]
    # p_c: [B, seq_len, H, N] -> [B, C, L, H, N]
    # p_b: [B, seq_len, H, N] -> [B, C, L, H, N]
    # x:   [B, seq_len, H, P] -> [B, C, L, H, P]
    
    Y_diag = torch.empty(B, C, block_len, H, P, device=device, dtype=dtype)
    total_ydiag = B * C * block_len * H
    grid3 = (total_ydiag,)
    ydiag_kernel[grid3](p_c, p_b, L_mat, x, Y_diag, B, C, block_len, H, N, P)
    
    # Step 4: Compute states [B, C, H, P, N]
    states = torch.empty(B, C, H, P, N, device=device, dtype=dtype)
    total_states = B * C * H * P
    grid4 = (total_states,)
    states_kernel[grid4](p_b, A_cumsum, x, states, B, C, block_len, H, N, P)
    
    # Step 5: Inter-chunk recurrence
    # Prepend initial states (zeros)
    initial_states = torch.zeros(B, 1, H, P, N, device=device, dtype=dtype)
    states_with_init = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N]
    
    # Get last cumsum values per chunk: A_cumsum[:, :, :, -1] -> [B, H, C]
    # Pad with 0 at front -> [B, H, C+1]
    last_cumsum = A_cumsum[:, :, :, -1]  # [B, H, C]
    padded_cumsum = torch.cat([torch.zeros(B, H, 1, device=device, dtype=dtype), last_cumsum], dim=2)  # [B, H, C+1]
    
    # Compute cumsum of padded -> [B, H, C+1]
    inter_cumsum = torch.empty(B, H, C + 1, device=device, dtype=dtype)
    xnumel2 = B * H
    grid5 = lambda meta: (triton.cdiv(xnumel2, meta['XBLOCK']),)
    inter_chunk_cumsum_kernel[grid5](padded_cumsum, inter_cumsum, xnumel2, C + 1)
    
    # Compute inter-chunk segsum exp [B, H, C+1, C+1]
    decay_inter = torch