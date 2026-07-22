import torch
import triton
import triton.language as tl

# ─── Kernel 1: permute + cumsum for A_blocks ───────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 32, 'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_X': 16, 'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_X': 8,  'BLOCK_L': 64}, num_warps=8),
        triton.Config({'BLOCK_X': 4,  'BLOCK_L': 64}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def kernel_permute_cumsum(
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_L: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length L
    pid = tl.program_id(0)
    total_rows = B * H * C
    if pid >= total_rows:
        return
    b = pid // (H * C)
    hc = pid % (H * C)
    h = hc // C
    c = hc % C

    offs_l = tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    # Load from [B, C, L, H]: stride = (C*L*H, L*H, H, 1)
    in_base = b * (C * L * H) + c * (L * H) + h
    vals = tl.load(in_ptr + in_base + offs_l * H, mask=mask_l, other=0.0)

    # Cumsum along L
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    # We do a simple prefix sum
    # Use associative scan pattern
    cumsum_vals = tl.cumsum(vals, axis=0)

    # Store to [B, H, C, L]: stride = (H*C*L, C*L, L, 1)
    out_base = b * (H * C * L) + h * (C * L) + c * L
    tl.store(out_ptr + out_base + offs_l, cumsum_vals, mask=mask_l)


# ─── Kernel 2: segsum + exp for diagonal blocks ────────────────────────────
# Computes L[b,h,c,i,j] = exp(cumsum[b,h,c,i] - cumsum[b,h,c,j]) with tril mask
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 64}, num_warps=8),
        triton.Config({'BLOCK_L': 32}, num_warps=4),
        triton.Config({'BLOCK_L': 32}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def kernel_segsum_exp(
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Each program handles one (b, h, c) block of size L x L
    pid = tl.program_id(0)
    total = B * H * C
    if pid >= total:
        return
    b = pid // (H * C)
    hc = pid % (H * C)
    h = hc // C
    c = hc % C

    cs_base = b * (H * C * L) + h * (C * L) + c * L
    out_base = b * (H * C * L * L) + h * (C * L * L) + c * (L * L)

    offs = tl.arange(0, BLOCK_L)
    mask = offs < L

    cs = tl.load(cumsum_ptr + cs_base + offs, mask=mask, other=0.0)

    # For each row i, col j: val = cs[i] - cs[j], masked by i >= j
    for i in range(L):
        cs_i = tl.load(cumsum_ptr + cs_base + i)
        diff = cs_i - cs
        # tril mask: j <= i
        tril_mask = offs <= i
        val = tl.where(tril_mask & mask, diff, float('-inf'))
        result = tl.exp(val)
        tl.store(out_ptr + out_base + i * L + offs, result, mask=mask)


# ─── Kernel 3: Y_diag einsum ───────────────────────────────────────────────
# Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# = sum_s L[b,h,c,l,s] * X[b,c,s,h,p] * sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 32, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 16, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'L', 'P', 'N'],
)
@triton.jit
def kernel_ydiag(
    B_sz: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B*C*H, ceil(L/BLOCK_L), ceil(P/BLOCK_P))
    pid_bch = tl.program_id(0)
    pid_l   = tl.program_id(1)
    pid_p   = tl.program_id(2)

    b = pid_bch // (C * H)
    ch = pid_bch % (C * H)
    c = ch // H
    h = ch % H

    l_start = pid_l * BLOCK_L
    p_start = pid_p * BLOCK_P

    offs_l = l_start + tl.arange(0, BLOCK_L)
    offs_p = p_start + tl.arange(0, BLOCK_P)
    mask_l = offs_l < L
    mask_p = offs_p < P

    # Accumulator: [BLOCK_L, BLOCK_P]
    acc = tl.zeros([BLOCK_L, BLOCK_P], dtype=tl.float32)

    # C[b,c,l,h,n]: stride (C*L*H*N, L*H*N, H*N, N, 1)
    C_base = b*(C*L*H*N) + c*(L*H*N) + h*N
    # B[b,c,s,h,n]: same layout
    B_base = b*(C*L*H*N) + c*(L*H*N) + h*N
    # L[b,h,c,l,s]: stride (H*C*L*L, C*L*L, L*L, L, 1)
    Lm_base = b*(H*C*L*L) + h*(C*L*L) + c*(L*L)
    # X[b,c,s,h,p]: stride (C*L*H*P, L*H*P, H*P, P, 1)
    X_base = b*(C*L*H*P) + c*(L*H*P) + h*P

    for s in range(L):
        # Load B[b,c,s,h,:] [N]
        offs_n = tl.arange(0, BLOCK_N)
        B_s = tl.zeros([BLOCK_N], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n2 = n_start + tl.arange(0, BLOCK_N)
            mask_n2 = offs_n2 < N
            B_s_chunk = tl.load(B_ptr + B_base + s*(H*N) + offs_n2, mask=mask_n2, other=0.0)
            # We'll accumulate CB product inline
            _ = B_s_chunk  # placeholder

        # Load X[b,c,s,h,p] [BLOCK_P]
        X_s = tl.load(X_ptr + X_base + s*(H*P) + offs_p, mask=mask_p, other=0.0)

        # Load L[b,h,c,l,s] for each l in BLOCK_L
        Lm_s = tl.load(Lm_base + Lm_base*0 + b*(H*C*L*L) + h*(C*L*L) + c*(L*L) + offs_l*L + s,
                        mask=mask_l, other=0.0)

        # Compute CB = sum_n C[b,c,l,h,n]*B[b,c,s,h,n] for each l
        CB = tl.zeros([BLOCK_L], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n2 = n_start + tl.arange(0, BLOCK_N)
            mask_n2 = offs_n2 < N
            C_l = tl.load(C_ptr + C_base + offs_l[:, None]*(H*N) + offs_n2[None, :],
                          mask=mask_l[:, None] & mask_n2[None, :], other=0.0)
            B_s_chunk = tl.load(B_ptr + B_base + s*(H*N) + offs_n2,
                                mask=mask_n2, other=0.0)
            CB += tl.sum(C_l * B_s_chunk[None, :], axis=1)

        # acc[l, p] += Lm_s[l] * CB[l] * X_s[p]
        acc += (Lm_s * CB)[:, None] * X_s[None, :]

    Y_base = b*(C*L*H*P) + c*(L*H*P) + h*P
    tl.store(Y_ptr + Y_base + offs_l[:, None]*(H*P) + offs_p[None, :],
             acc, mask=mask_l[:, None] & mask_p[None, :])


# ─── Kernel 4: decay_states + states einsum ────────────────────────────────
# decay_states[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay_states[b,h,c,l] * X[b,c,l,h,p]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 16, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_L': 32, 'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_sz', 'C', 'H', 'L', 'P', 'N'],
)
@triton.jit
def kernel_states(
    B_sz: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B*C*H, ceil(P/BLOCK_P), ceil(N/BLOCK_N))
    pid_bch = tl.program_id(0)
    pid_p   = tl.program_id(1)
    pid_n   = tl.program_id(2)

    b = pid_bch // (C * H)
    ch = pid_bch % (C * H)
    c = ch // H
    h = ch % H

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    offs_p = p_start + tl.arange(0, BLOCK_P)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_p = offs_p < P
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    # A_cumsum[b,h,c,-1]: last element
    Acum_base = b*(H*C*L) + h*(C*L) + c*L
    A_last = tl.load(Acum_ptr + Acum_base + L - 1)

    B_base = b*(C*L*H*N) + c*(L*H*N) + h*N
    X_base = b*(C*L*H*P) + c*(L*H*P) + h*P

    for l_start in range(0, L, BLOCK_L):
        offs_l = l_start + tl.arange(0, BLOCK_L)
        mask_l = offs_l < L

        # decay = exp(A_last - A_cumsum[b,h,c,l])
        A_l = tl.load(Acum_ptr + Acum_base + offs_l, mask=mask_l, other=0.0)
        decay = tl.exp(A_last - A_l)  # [BLOCK_L]

        # X[b,c,l,h,p]: [BLOCK_L, BLOCK_P]
        X_l = tl.load(X_ptr + X_base + offs_l[:, None]*(H*P) + offs_p[None, :],
                      mask=mask_l[:, None] & mask_p[None, :], other=0.0)

        # B[b,c,l,h,n]: [BLOCK_L, BLOCK_N]
        B_l = tl.load(B_ptr + B_base + offs_l[:, None]*(H*N) + offs_n[None, :],
                      mask=mask_l[:, None] & mask_n[None, :], other=0.0)

        # weighted: decay[l] * X[l,p] * B[l,n]
        dX = decay[:, None] * X_l  # [BLOCK_L, BLOCK_P]
        acc += tl.dot(tl.trans(dX), B_l)  # [BLOCK_P, BLOCK_N]

    # states[b,c,h,p,n]: stride (C*H*P*N, H*P*N, P*N, N, 1)
    st_base = b*(C*H*P*N) + c*(H*P*N) + h*(P*N)
    tl.store(states_ptr + st_base + offs_p[:, None]*N + offs_n[None, :],
             acc, mask=mask_p[:, None] & mask_n[None, :])


# ─── Kernel 5: inter-chunk recurrence ──────────────────────────────────────
# decay_chunk[b,h,i,j] = exp(cumsum2[b,h,i] - cumsum2[b,h,j]) with tril mask
# new_states[b,i,h,p,n] = sum_j decay_chunk[b,h,i,j] * states[b,j,h,p,n]
# (where states includes initial_states prepended)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_sz', 'C', 'H', 'P', 'N'],
)
@triton.jit
def kernel_inter_chunk(
    B_sz: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    P: tl.constexpr, N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B*H, C, ceil(P/BLOCK_P)*ceil(N/BLOCK_N))
    pid_bh  = tl.program_id(0)
    pid_c   = tl.program_id(1)
    pid_pn  = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H
    c_idx = pid_c  # output chunk index (0..C-1), corresponds to states index 1..C

    n_pn_blocks = tl.cdiv(N, BLOCK_N)
    pid_p = pid_pn // n_pn_blocks
    pid_n = pid_pn % n_pn_blocks

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N
    offs_p = p_start + tl.arange(0, BLOCK_P)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_p = offs_p < P
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    # cumsum2[b,h,i] where i = c_idx+1 (1-indexed in the padded array)
    cs2_base = b*(H*(C+1)) + h*(C+1)
    cs_i = tl.load(cumsum2_ptr + cs2_base + c_idx + 1)

    # states[b,j,h,p,n]: stride (C+1)*H*P*N, H*P*N, P*N, N, 1
    st_stride_b = (C+1)*H*P*N
    st_stride_c = H*P*N
    st_stride_h = P*N

    for j in range(c_idx + 1):
        # decay = exp(cs_i - cs_j) if j <= c_idx (tril)
        cs_j = tl.load(cumsum2_ptr + cs2_base + j)
        decay = tl.exp(cs_i - cs_j)

        # states[b,j,h,p,n]
        st_base = b*st_stride_b + j*st_stride_c + h*st_stride_h
        st_pn = tl.load(states_ptr + st_base + offs_p[:, None]*N + offs_n[None, :],
                        mask=mask_p[:, None] & mask_n[None, :], other=0.0)
        acc += decay * st_pn

    # out[b,c_idx,h,p,n]
    out_base = b*(C*H*P*N) + c_idx*(H*P*N) + h*(P*N)
    tl.store(out_ptr + out_base + offs_p[:, None]*N + offs_n[None, :],
             acc, mask=mask_p[:, None] & mask_n[None, :])


# ─── Kernel 6: Y_off computation ───────────────────────────────────────────
# Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * new_states[b,c,h,p,n] * exp(A_cumsum[b,h,c,l])
# where exp(A_cumsum[b,h,c,l]) is the decay from start of chunk
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 32, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_P': 16, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_sz', 'C', 'H', 'L', 'P', 'N'],
)
@triton.jit
def kernel_yoff(
    B_sz: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_bch = tl.program_id(0)
    pid_l   = tl.program_id(1)
    pid_p   = tl.program_id(2)

    b = pid_bch // (C * H)
    ch = pid_bch % (C * H)
    c = ch // H
    h = ch % H

    l_start = pid_l * BLOCK_L
    p_start = pid_p * BLOCK_P

    offs_l = l_start + tl.arange(0, BLOCK_L)
    offs_p = p_start + tl.arange(0, BLOCK_P)
    mask_l = offs_l < L
    mask_p = offs_p < P

    # decay_l = exp(A_cumsum[b,h,c,l])
    Acum_base = b*(H*C*L) + h*(C*L) + c*L
    A_l = tl.load(Acum_ptr + Acum_base + offs_l, mask=mask_l, other=0.0)
    decay_l = tl.exp(A_l)  # [BLOCK_L]

    # states[b,c,h,p,n]: stride (C*H*P*N, H*P*N, P*N, N, 1)
    st_base = b*(C*H*P*N) + c*(H*P*N) + h*(P*N)

    acc = tl.zeros([BLOCK_L, BLOCK_P], dtype=tl.float32)

    C_base = b*(C*L*H*N) + c*(L*H*N) + h*N

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # C[b,c,l,h,n]: [BLOCK_L, BLOCK_N]
        C_ln = tl.load(C_ptr + C_base + offs_l[:, None]*(H*N) + offs_n[None, :],
                       mask=mask_l[:, None] & mask_n[None, :], other=0.0)

        # states[b,c,h,p,n]: [BLOCK_P, BLOCK_N]
        st_pn = tl.load(states_ptr + st_base + offs_p[:, None]*N + offs_n[None, :],
                        mask=mask_p[:, None] & mask_n[None, :], other=0.0)

        # acc[l,p] += sum_n C[l,n] * states[p,n]
        acc += tl.dot(C_ln, tl.trans(st_pn))

    # multiply by decay
    acc = acc * decay_l[:, None]

    Y_base = b*(C*L*H*P) + c*(L*H*P) + h*P
    tl.store(Y_ptr + Y_base + offs_l[:, None]*(H*P) + offs_p[None, :],
             acc, mask=mask_l[:, None] & mask_p[None, :])


def kernel_function(p_a, p_b, p_c, x):
    """
    
    
    """
    B, seq_len, H = p_a.shape
    N = p_b.shape[-1]
    P = x.shape[-1]
    block_len = 64
    C = seq_len // block_len  # number of chunks
    L = block_len

    device = p_a.device
    dtype = p_a.dtype

    # ── Step 1: Reshape inputs into blocks ──────────────────────────────────
    # A_blocks: [B, C, L, H]
    A_blocks = p_a.reshape(B, C, L, H)
    # B_blocks: [B, C, L, H, N]
    B_blocks = p_b.reshape(B, C, L, H, N)
    # C_blocks: [B, C, L, H, N]
    C_blocks = p_c.reshape(B, C, L, H, N)
    # X_blocks: [B, C, L, H, P]
    X_blocks = x.reshape(B, C, L, H, P)

    # ── Step 2: Permute A and compute cumsum ─────────────────────────────────
    # A_perm: [B, H, C, L]
    A_perm = A_blocks.permute(0, 3, 1, 2).contiguous()
    # A_cumsum: [B, H, C, L]
    A_cumsum = torch.cumsum(A_perm, dim=-1)

    # ── Step 3: Compute segsum exp (L matrix) ────────────────────────────────
    # L_mat[b,h,c,i,j] = exp(A_cumsum[b,h,c,i] - A_cumsum[b,h,c,j]) with tril
    # Shape: [B, H, C, L, L]
    L_mat = torch.empty(B, H, C, L, L, device=device, dtype=dtype)
    total_bhc = B * H * C
    kernel_segsum_exp[(total_bhc,)](
        A_cumsum, L_mat,
        B, H, C, L,
        BLOCK_L=L,
    )

    # ── Step 4: Y_diag ───────────────────────────────────────────────────────
    # Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n]*B[b,c,s,h,n]*L[b,h,c,l,s]*X[b,c,s,h,p]