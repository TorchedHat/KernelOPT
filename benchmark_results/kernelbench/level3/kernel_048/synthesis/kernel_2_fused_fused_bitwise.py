import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: permute + cumsum for A_blocks
# Input:  p_a [B, L, H]  (B=2048, L=128, H=8)
# Output: A_cumsum [B, H, C, block_len]  (B=2048, H=8, C=2, T=64)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64, 'BLOCK_ROW': 1}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_ROW': 2}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_ROW': 4}, num_warps=8),
        triton.Config({'BLOCK_T': 64, 'BLOCK_ROW': 8}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'T'],
)
@triton.jit
def cumsum_permute_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length T
    pid = tl.program_id(0)
    total_rows = B * H * C
    row_idx = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = row_idx < total_rows

    b_idx = row_idx // (H * C)
    hc_idx = row_idx % (H * C)
    h_idx = hc_idx // C
    c_idx = hc_idx % C

    # Load T elements and compute cumsum
    t_offs = tl.arange(0, BLOCK_T)  # T must be constexpr power of 2

    # Input layout: [B, L, H] with L = C*T
    # a[b, c*T + t, h] -> out[b, h, c, t]
    in_idx = (b_idx[:, None] * (C * T * H) +
              (c_idx[:, None] * T + t_offs[None, :]) * H +
              h_idx[:, None])
    vals = tl.load(a_ptr + in_idx, mask=row_mask[:, None] & (t_offs[None, :] < T), other=0.0)

    # Compute cumsum along T dimension (axis=1 of the 2D tile)
    # Use associative scan
    cumsum_vals = tl.cumsum(vals, axis=1)

    out_idx = (b_idx[:, None] * (H * C * T) +
               h_idx[:, None] * (C * T) +
               c_idx[:, None] * T +
               t_offs[None, :])
    tl.store(out_ptr + out_idx, cumsum_vals, mask=row_mask[:, None] & (t_offs[None, :] < T))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: segsum + exp for L matrix [B, H, C, T, T]
# A_cumsum[b,h,c,i] - A_cumsum[b,h,c,j], masked to lower triangular, then exp
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64, 'BLOCK_BHC': 1}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_BHC': 2}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_BHC': 4}, num_warps=8),
        triton.Config({'BLOCK_T': 64, 'BLOCK_BHC': 8}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'T'],
)
@triton.jit
def segsum_exp_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_BHC: tl.constexpr,
):
    # Each program handles BLOCK_BHC (b,h,c) triples, full T x T matrix
    pid = tl.program_id(0)
    total = B * H * C
    bhc_start = pid * BLOCK_BHC
    bhc_offs = bhc_start + tl.arange(0, BLOCK_BHC)
    bhc_mask = bhc_offs < total

    b_idx = bhc_offs // (H * C)
    h_idx = (bhc_offs // C) % H
    c_idx = bhc_offs % C

    # Load cumsum row [BLOCK_BHC, T]
    t_offs = tl.arange(0, BLOCK_T)
    cs_base = b_idx[:, None] * (H * C * T) + h_idx[:, None] * (C * T) + c_idx[:, None] * T
    cs = tl.load(cumsum_ptr + cs_base + t_offs[None, :],
                 mask=bhc_mask[:, None] & (t_offs[None, :] < T), other=0.0)

    # For each (i, j): val = cs[i] - cs[j], mask upper triangular
    # We need to write T*T values per (b,h,c)
    # Process row by row of the T x T matrix
    out_base = b_idx * (H * C * T * T) + h_idx * (C * T * T) + c_idx * (T * T)

    for i in tl.static_range(0, T):
        cs_i = tl.load(cumsum_ptr + cs_base[:, 0] + i,
                       mask=bhc_mask, other=0.0)  # [BLOCK_BHC]
        diff = cs_i[:, None] - cs  # [BLOCK_BHC, T]
        # mask: j <= i (lower triangular)
        j_offs = t_offs[None, :]  # [1, T]
        mask_tril = j_offs <= i
        val = tl.where(mask_tril, diff, float('-inf'))
        val_exp = libdevice.exp(val)
        out_idx = out_base + i * T + t_offs[None, :]
        tl.store(out_ptr + out_idx, val_exp,
                 mask=bhc_mask[:, None] & (t_offs[None, :] < T))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Y_diag einsum
# Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# = sum_s L[b,h,c,l,s] * X[b,c,s,h,p] * sum_n C[b,c,l,h,n]*B[b,c,s,h,n]
# ─────────────────────────────────────────────────────────────────────────────
# This is complex - implement as a loop over s
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_L': 16, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_L': 16, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_L': 32, 'BLOCK_S': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_L': 32, 'BLOCK_S': 16}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'T', 'P', 'N'],
)
@triton.jit
def ydiag_kernel(
    B_val: tl.constexpr,
    C_val: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B*C*H, ceil(T/BLOCK_L), ceil(P/BLOCK_P))
    pid_bch = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_p = tl.program_id(2)

    b_idx = pid_bch // (C_val * H)
    ch_idx = pid_bch % (C_val * H)
    c_idx = ch_idx // H
    h_idx = ch_idx % H

    l_start = pid_l * BLOCK_L
    p_start = pid_p * BLOCK_P

    l_offs = l_start + tl.arange(0, BLOCK_L)
    p_offs = p_start + tl.arange(0, BLOCK_P)
    l_mask = l_offs < T
    p_mask = p_offs < P

    # Accumulator [BLOCK_L, BLOCK_P]
    acc = tl.zeros([BLOCK_L, BLOCK_P], dtype=tl.float32)

    # C[b,c,l,h,n] base
    C_base = b_idx * (C_val * T * H * N) + c_idx * (T * H * N) + h_idx * N
    # B[b,c,s,h,n] base
    B_base = b_idx * (C_val * T * H * N) + c_idx * (T * H * N) + h_idx * N
    # L[b,h,c,l,s] base
    L_base = b_idx * (H * C_val * T * T) + h_idx * (C_val * T * T) + c_idx * (T * T)
    # X[b,c,s,h,p] base
    X_base = b_idx * (C_val * T * H * P) + c_idx * (T * H * P) + h_idx * P

    n_offs = tl.arange(0, N)

    for s in range(T):
        # Load C[b,c,l,h,:] for all l: [BLOCK_L, N]
        C_idx = C_base + l_offs[:, None] * (H * N) + n_offs[None, :]
        C_vals = tl.load(C_ptr + C_idx,
                         mask=l_mask[:, None] & (n_offs[None, :] < N), other=0.0)

        # Load B[b,c,s,h,:]: [N]
        B_idx = B_base + s * (H * N) + n_offs
        B_vals = tl.load(B_ptr + B_idx, mask=n_offs < N, other=0.0)

        # CB[l] = sum_n C[l,n] * B[s,n]: [BLOCK_L]
        CB = tl.sum(C_vals * B_vals[None, :], axis=1)  # [BLOCK_L]

        # Load L[b,h,c,l,s]: [BLOCK_L]
        L_idx = L_base + l_offs * T + s
        L_vals = tl.load(L_ptr + L_idx, mask=l_mask, other=0.0)

        # Load X[b,c,s,h,p]: [BLOCK_P]
        X_idx = X_base + s * (H * P) + p_offs
        X_vals = tl.load(X_ptr + X_idx, mask=p_mask, other=0.0)

        # acc += (CB * L_vals)[:, None] * X_vals[None, :]
        weight = CB * L_vals  # [BLOCK_L]
        acc += weight[:, None] * X_vals[None, :]  # [BLOCK_L, BLOCK_P]

    # Store Y[b,c,l,h,p]
    Y_base = b_idx * (C_val * T * H * P) + c_idx * (T * H * P) + l_offs[:, None] * (H * P) + h_idx * P + p_offs[None, :]
    tl.store(Y_ptr + Y_base, acc.to(tl.float32),
             mask=l_mask[:, None] & p_mask[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: decay_states and states computation
# decay_states[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay_states[b,h,c,l] * X[b,c,l,h,p]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16, 'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16, 'BLOCK_L': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16, 'BLOCK_L': 64}, num_warps=8),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 32, 'BLOCK_L': 64}, num_warps=8),
    ],
    key=['B_val', 'C_val', 'H', 'T', 'P', 'N'],
)
@triton.jit
def states_kernel(
    B_val: tl.constexpr,
    C_val: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Grid: (B*C*H, ceil(P/BLOCK_P), ceil(N/BLOCK_N))
    pid_bch = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    b_idx = pid_bch // (C_val * H)
    ch_idx = pid_bch % (C_val * H)
    c_idx = ch_idx // H
    h_idx = ch_idx % H

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offs = p_start + tl.arange(0, BLOCK_P)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    # A_cumsum last value: A_cumsum[b,h,c,T-1]
    A_last_idx = b_idx * (H * C_val * T) + h_idx * (C_val * T) + c_idx * T + (T - 1)
    a_last = tl.load(A_cumsum_ptr + A_last_idx)

    B_base = b_idx * (C_val * T * H * N) + c_idx * (T * H * N) + h_idx * N
    X_base = b_idx * (C_val * T * H * P) + c_idx * (T * H * P) + h_idx * P
    A_base = b_idx * (H * C_val * T) + h_idx * (C_val * T) + c_idx * T

    for l in range(T):
        # decay = exp(a_last - A_cumsum[b,h,c,l])
        a_l = tl.load(A_cumsum_ptr + A_base + l)
        decay = libdevice.exp(a_last - a_l)

        # B[b,c,l,h,n]: [BLOCK_N]
        B_idx = B_base + l * (H * N) + n_offs
        B_vals = tl.load(B_ptr + B_idx, mask=n_mask, other=0.0)

        # X[b,c,l,h,p]: [BLOCK_P]
        X_idx = X_base + l * (H * P) + p_offs
        X_vals = tl.load(X_ptr + X_idx, mask=p_mask, other=0.0)

        # acc[p,n] += decay * X[p] * B[n]
        acc += (decay * X_vals)[:, None] * B_vals[None, :]

    # Store states[b,c,h,p,n]
    S_base = b_idx * (C_val * H * P * N) + c_idx * (H * P * N) + h_idx * (P * N)
    S_idx = S_base + p_offs[:, None] * N + n_offs[None, :]
    tl.store(states_ptr + S_idx, acc.to(tl.float32),
             mask=p_mask[:, None] & n_mask[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 5: inter-chunk decay_chunk (segsum on chunk-level cumsum)
# Input: A_cumsum_last [B, H, C] (last element of each chunk)
# Padded with 0 at front -> [B, H, C+1], cumsum -> [B, H, C+1]
# Then segsum + exp -> [B, H, C+1, C+1]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BH': 16, 'BLOCK_C': 4}, num_warps=2),
        triton.Config({'BLOCK_BH': 32, 'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_BH': 64, 'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_BH': 128, 'BLOCK_C': 4}, num_warps=8),
    ],
    key=['B_val', 'H', 'C_val'],
)
@triton.jit
def chunk_decay_kernel(
    B_val: tl.constexpr,
    H: tl.constexpr,
    C_val: tl.constexpr,
    T: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # C+1 = 3 in our case
    C1 = C_val + 1  # 3

    pid = tl.program_id(0)
    total_bh = B_val * H
    bh_start = pid * BLOCK_BH
    bh_offs = bh_start + tl.arange(0, BLOCK_BH)
    bh_mask = bh_offs < total_bh

    b_idx = bh_offs // H
    h_idx = bh_offs % H

    # Load A_cumsum last values for each chunk: [BLOCK_BH, C_val]
    c_offs = tl.arange(0, BLOCK_C)  # BLOCK_C >= C_val
    A_last_base = b_idx[:, None] * (H * C_val * T) + h_idx[:, None] * (C_val * T) + c_offs[None, :] * T + (T - 1)
    a_last = tl.load(A_cumsum_ptr + A_last_base,
                     mask=bh_mask[:, None] & (c_offs[None, :] < C_val), other=0.0)

    # Compute cumsum with 0 prepended: [BLOCK_BH, C+1]
    # cs[0] = 0, cs[1] = a_last[0], cs[2] = a_last[0]+a_last[1]
    # We do this manually for small C
    # cs[i] = sum_{k<i} a_last[k]
    # For C=2, C1=3: cs = [0, a_last[0], a_last[0]+a_last[1]]

    # Build cumsum array [BLOCK_BH, C1]
    # cs[b,0] = 0
    # cs[b,i] = cs[b,i-1] + a_last[b,i-1]  for i>=1
    cs0 = tl.zeros([BLOCK_BH], dtype=tl.float32)
    cs1 = tl.load(A_cumsum_ptr + b_idx * (H * C_val * T) + h_idx * (C_val * T) + 0 * T + (T - 1),
                  mask=bh_mask, other=0.0)
    cs2 = cs1 + tl.load(A_cumsum_ptr + b_idx * (H * C_val * T) + h_idx * (C_val * T) + 1 * T + (T - 1),
                        mask=bh_mask, other=0.0)

    # cs[BLOCK_BH, C1] = stack [cs0, cs1, cs2]
    # For each (i, j) in C1 x C1: val = cs[i] - cs[j], mask j <= i
    out_base = b_idx * (H * C1 * C1) + h_idx * (C1 * C1)

    # Unroll for C1=3
    cs_arr = [cs0, cs1, cs2]
    for i in tl.static_range(3):
        for j in tl.static_range(3):
            if j <= i:
                diff = cs_arr[i] - cs_arr[j]
                val = libdevice.exp(diff)
            else:
                val = tl.zeros([BLOCK_BH], dtype=tl.float32)
            out_idx = out_base + i * C1 + j
            tl.store(decay_chunk_ptr + out_idx, val, mask=bh_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6: new_states einsum + Y_off computation
# new_states[b,c,h,p,n] = sum_{c2} decay_chunk[b,h,c+1,c2] * states_cat[b,c2,h,p,n]
# Y_off[b,c,l,h,p] = sum_{n} C[b,c,l,h,n] * sum_{c2<c} decay * exp(A_cumsum[b,h,c,l]-A_cumsum_last[b,h,c2]) * states[b,c2,h,p,n]
# ─────────────────────────────────────────────────────────────────────────────
# This is the most complex part. Let me implement it step by step.

# Actually, let me re-read the original code more carefully:
# states = cat([initial_states, states], dim=1)  -> [B, C+1, H, P, N]
# decay_chunk = exp(segsum(A_cumsum_last_padded))  -> [B, H, C+1, C+1]
# new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states)  -> [B, C+1, H, P, N]
# Y_off = einsum("bclhn,bchpn,bhcl->bclhp", C_blocks, new_states[:, :C], exp(A_cumsum), X... wait

# Let me re-read more carefully from the original:
# decay_chunk[b,h,z,c] * states[b,c,h,p,n] -> new_states[b,z,h,p,n]
# Then Y_off = einsum("bclhn,bchpn,bhcl->bclhp", C, new_states[:C], exp(A_cumsum))

# Kernel for new_states: [B, C+1, H, P, N] from decay_chunk [B, H, C+1, C+1] and states_cat [B, C+1, H, P, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_val', 'H', 'C1', 'P', 'N'],
)
@triton.jit
def new_states_kernel(
    B_val: tl.constexpr,
    H: tl.constexpr,
    C1: tl.constexpr,  # C+1 = 3
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B*C1*H, ceil(P/BLOCK_P), ceil(N/BLOCK_N))
    pid_bzh = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    b_idx = pid_bzh // (C1 * H)
    zh_idx = pid_bzh % (C1 * H)
    z_idx = zh_idx // H
    h_idx = zh_idx % H

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N
    p_offs = p_start + tl.arange(0, BLOCK_P)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    # decay_chunk[b,h,z,c] for c in range(C1)
    dc_base = b_idx * (H * C1 * C1) + h_idx * (C1 * C1) + z_idx * C1

    # states_cat[b,c,h,p,n]
    sc_base = b_idx * (C1 * H * P * N) + h_idx * (P * N)

    for c in range(C1):
        dc = tl.load(decay_chunk_ptr + dc_base + c)

        sc_idx = sc_base + c * (H * P * N) + p_offs[:, None] * N + n_offs[None, :]
        sc_vals = tl.load(states_cat_ptr + sc_idx,
                          mask=p_mask[:, None] & n_mask[None, :], other=0.0)
        acc += dc * sc_vals

    out_base = b_idx * (C1 * H * P * N) + z_idx * (H * P * N) + h_idx * (P * N)
    out_idx = out_base + p_offs[:, None] * N + n_offs[None, :]
    tl.store(new_states_ptr + out_