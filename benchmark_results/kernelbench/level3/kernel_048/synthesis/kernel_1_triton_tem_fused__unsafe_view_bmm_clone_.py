import torch
import triton
import triton.language as tl

# Constants for this problem
BATCH = 2048
SEQ_LEN = 128
N_HEADS = 8
D_HEAD = 64
D_STATE = 16
BLOCK_LEN = 64
N_CHUNKS = 2  # SEQ_LEN // BLOCK_LEN

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Permute + CumSum for A_blocks
# Input:  p_a [B, 128, 8]  ->  reshape [B, 2, 64, 8]  ->  permute [B, 8, 2, 64]
# Output: A_cumsum [B, 8, 2, 64]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_X': 128}, num_warps=8),
        triton.Config({'BLOCK_X': 256}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def cumsum_permute_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length L
    pid = tl.program_id(0)
    total_rows = B * H * C
    if pid >= total_rows:
        return
    b = pid // (H * C)
    h = (pid // C) % H
    c = pid % C

    # Load L elements from a_ptr with permuted indexing
    # a_ptr layout: [B, C*L, H] -> a[b, c*L + l, h]
    offs = tl.arange(0, L)
    mask = offs < L
    src_idx = b * (C * L * H) + (c * L + offs) * H + h
    vals = tl.load(a_ptr + src_idx, mask=mask, other=0.0)

    # Compute cumulative sum
    # Use associative scan pattern
    acc = tl.zeros([L], dtype=tl.float32)
    # Manual prefix sum since L=64 fits in registers
    # We'll do it iteratively
    cumsum = tl.cumsum(vals, axis=0)

    # Store to out_ptr layout: [B, H, C, L]
    dst_idx = b * (H * C * L) + h * (C * L) + c * L + offs
    tl.store(out_ptr + dst_idx, cumsum, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Compute L matrix = exp(segsum(A_blocks))
# segsum: x_segsum[..., i, j] = cumsum[i] - cumsum[j], masked to lower tri
# Input:  A_cumsum [B, H, C, L]
# Output: L_mat [B, H, C, L, L]  (lower triangular exp differences)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 32}, num_warps=4),
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 64}, num_warps=8),
        triton.Config({'BLOCK_L': 128}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def segsum_exp_kernel(
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Each program handles one (b, h, c, row_i) -> L columns
    pid = tl.program_id(0)
    total = B * H * C * L
    if pid >= total:
        return
    b = pid // (H * C * L)
    h = (pid // (C * L)) % H
    c = (pid // L) % C
    i = pid % L

    # Load cumsum[b, h, c, i]
    base = b * (H * C * L) + h * (C * L) + c * L
    cs_i = tl.load(cumsum_ptr + base + i)

    # For each j in [0, L), compute exp(cs_i - cs_j) if j <= i else -inf
    offs_j = tl.arange(0, BLOCK_L)
    mask_j = offs_j < L
    cs_j = tl.load(cumsum_ptr + base + offs_j, mask=mask_j, other=0.0)

    diff = cs_i - cs_j
    # Lower triangular mask: j <= i
    tril_mask = offs_j <= i
    val = tl.where(tril_mask & mask_j, tl.exp(diff), 0.0)

    # Store to out[b, h, c, i, j]
    out_base = b * (H * C * L * L) + h * (C * L * L) + c * (L * L) + i * L
    tl.store(out_ptr + out_base + offs_j, val, mask=mask_j)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L_mat, X)
# This is a batched matmul. For each (b, c, l, h):
#   Y_diag[b,c,l,h,p] = sum_s sum_n C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# 
# Rewrite: for each (b, c, l, h):
#   Y[b,c,l,h,p] = sum_s L[b,h,c,l,s] * (sum_n C[b,c,l,h,n]*B[b,c,s,h,n]) * X[b,c,s,h,p]
#
# Step A: CB[b,c,l,s,h] = sum_n C[b,c,l,h,n]*B[b,c,s,h,n]  -> dot product over n
# Step B: Y[b,c,l,h,p] = sum_s L[b,h,c,l,s] * CB[b,c,l,s,h] * X[b,c,s,h,p]
#
# We'll fuse steps A and B into one kernel using tl.dot
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 64, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 64, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 64, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 64, 'BLOCK_N': 16}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L', 'P', 'N'],
)
@triton.jit
def ydiag_kernel(
    B_: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B_ * C * L * H, cdiv(P, BLOCK_P))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    total0 = B_ * C * L * H
    if pid0 >= total0:
        return

    b = pid0 // (C * L * H)
    clh = pid0 % (C * L * H)
    c = clh // (L * H)
    lh = clh % (L * H)
    l = lh // H
    h = lh % H

    p_offs = pid1 * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # Accumulator for Y[b,c,l,h,p]
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    # C[b,c,l,h,:] shape [N]
    c_base = b * (C * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
    # X[b,c,s,h,p] for varying s
    x_base_bch = b * (C * L * H * P) + c * (L * H * P) + h * P  # + s*(H*P) + p

    # L_mat[b,h,c,l,s] for varying s
    lmat_base = b * (H * C * L * L) + h * (C * L * L) + c * (L * L) + l * L

    # Loop over s (sequence positions within chunk)
    for s in range(L):
        # Load L_mat[b,h,c,l,s]
        lval = tl.load(L_ptr + lmat_base + s)

        if lval == 0.0:
            continue

        # Compute CB = sum_n C[b,c,l,h,n] * B[b,c,s,h,n]
        b_base = b * (C * L * H * N) + c * (L * H * N) + s * (H * N) + h * N
        cb = tl.zeros([1], dtype=tl.float32)
        for n_off in range(0, N, BLOCK_N):
            n_offs = n_off + tl.arange(0, BLOCK_N)
            n_mask = n_offs < N
            c_vals = tl.load(C_ptr + c_base + n_offs, mask=n_mask, other=0.0)
            b_vals = tl.load(B_ptr + b_base + n_offs, mask=n_mask, other=0.0)
            cb += tl.sum(c_vals * b_vals, axis=0)

        # Load X[b,c,s,h,p]
        x_base = x_base_bch + s * (H * P)
        x_vals = tl.load(X_ptr + x_base + p_offs, mask=p_mask, other=0.0)

        acc += lval * cb[0] * x_vals

    # Store Y[b,c,l,h,p]
    y_base = b * (C * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
    tl.store(Y_ptr + y_base + p_offs, acc, mask=p_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Compute decay_states and states
# decay_states[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay_states[b,h,c,l] * X[b,c,l,h,p]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L', 'P', 'N'],
)
@triton.jit
def states_kernel(
    B_: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B_ * C * H, cdiv(P, BLOCK_P), cdiv(N, BLOCK_N))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    b = pid0 // (C * H)
    ch = pid0 % (C * H)
    c = ch // H
    h = ch % H

    p_offs = pid1 * BLOCK_P + tl.arange(0, BLOCK_P)
    n_offs = pid2 * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    # Load A_cumsum last element: A_cumsum[b,h,c,-1]
    acum_base = b * (H * C * L) + h * (C * L) + c * L
    acum_last = tl.load(Acumsum_ptr + acum_base + (L - 1))

    # Accumulate states[b,c,h,p,n] = sum_l decay[l] * B[b,c,l,h,n] * X[b,c,l,h,p]
    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    for l in range(L):
        # decay = exp(acum_last - A_cumsum[b,h,c,l])
        acum_l = tl.load(Acumsum_ptr + acum_base + l)
        decay = tl.exp(acum_last - acum_l)

        # B[b,c,l,h,n]
        b_base = b * (C * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
        b_vals = tl.load(B_ptr + b_base + n_offs, mask=n_mask, other=0.0)  # [BLOCK_N]

        # X[b,c,l,h,p]
        x_base = b * (C * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
        x_vals = tl.load(X_ptr + x_base + p_offs, mask=p_mask, other=0.0)  # [BLOCK_P]

        # outer product scaled by decay
        acc += decay * x_vals[:, None] * b_vals[None, :]

    # Store states[b,c,h,p,n]
    # states layout: [B, C, H, P, N]
    st_base = b * (C * H * P * N) + c * (H * P * N) + h * (P * N)
    st_offs = p_offs[:, None] * N + n_offs[None, :]
    st_mask = p_mask[:, None] & n_mask[None, :]
    tl.store(states_ptr + st_base + st_offs, acc, mask=st_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 5: Inter-chunk recurrence
# Compute decay_chunk[b,h,i,j] = exp(segsum(chunk_sums)[b,h,i,j])
# Then new_states[b,c,h,p,n] = sum_{c'<c} decay_chunk[b,h,c,c'] * states[b,c',h,p,n]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'P', 'N'],
)
@triton.jit
def inter_chunk_kernel(
    B_: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B_ * C * H, cdiv(P, BLOCK_P), cdiv(N, BLOCK_N))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    b = pid0 // (C * H)
    ch = pid0 % (C * H)
    c = ch // H  # output chunk index [0, C)
    h = ch % H

    p_offs = pid1 * BLOCK_P + tl.arange(0, BLOCK_P)
    n_offs = pid2 * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    # states has C+1 entries (index 0 = initial, 1..C = computed)
    # decay_chunk[b,h,c+1,j] for j in [0, c+1)
    # out[b,c,h,p,n] = sum_{j=0}^{c} decay_chunk[b,h,c+1,j] * states[b,j,h,p,n]
    Cp1 = C + 1
    dc_base = b * (H * Cp1 * Cp1) + h * (Cp1 * Cp1) + (c + 1) * Cp1

    for j in range(Cp1):
        # decay_chunk[b,h,c+1,j]
        dc = tl.load(decay_chunk_ptr + dc_base + j)

        # states[b,j,h,p,n]
        st_base = b * (Cp1 * H * P * N) + j * (H * P * N) + h * (P * N)
        st_offs = p_offs[:, None] * N + n_offs[None, :]
        st_mask = p_mask[:, None] & n_mask[None, :]
        st_vals = tl.load(states_ptr + st_base + st_offs, mask=st_mask, other=0.0)

        acc += dc * st_vals

    # Store out[b,c,h,p,n]
    out_base = b * (C * H * P * N) + c * (H * P * N) + h * (P * N)
    out_offs = p_offs[:, None] * N + n_offs[None, :]
    out_mask = p_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_base + out_offs, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6: Y_off = einsum("bchpn,bclhn->bclhp", new_states, C_blocks) * decay
# For each (b,c,l,h,p):
#   Y_off[b,c,l,h,p] = sum_n new_states[b,c,h,p,n] * C[b,c,l,h,n] * exp(A_cumsum[b,h,c,l] - A_cumsum[b,h,c,-1] + A_cumsum[b,h,c,-1])
# Wait, let me re-read the original code...
#
# Actually from the original:
#   decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)  -> [B,H,C,L]
#   states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
#   ...inter-chunk recurrence gives new_states [B,C,H,P,N]...
#   Y_off = einsum("bchpn,bclhn->bclhp", new_states, C_blocks) * exp(A_cumsum)
#
# So Y_off[b,c,l,h,p] = exp(A_cumsum[b,h,c,l]) * sum_n new_states[b,c,h,p,n] * C[b,c,l,h,n]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L', 'P', 'N'],
)
@triton.jit
def yoff_kernel(
    B_: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B_ * C * L * H, cdiv(P, BLOCK_P))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // (C * L * H)
    clh = pid0 % (C * L * H)
    c = clh // (L * H)
    lh = clh % (L * H)
    l = lh // H
    h = lh % H

    p_offs = pid1 * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # Load exp(A_cumsum[b,h,c,l])
    acum = tl.load(Acumsum_ptr + b * (H * C * L) + h * (C * L) + c * L + l)
    decay = tl.exp(acum)

    # Compute sum_n new_states[b,c,h,p,n] * C[b,c,l,h,n]
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    for n_off in range(0, N, 16):
        n_offs = n_off + tl.arange(0, 16)
        n_mask = n_offs < N

        # new_states[b,c,h,p,n]
        st_base = b * (C * H * P * N) + c * (H * P * N) + h * (P * N)
        st_offs = p_offs[:, None] * N + n_offs[None, :]
        st_mask = p_mask[:, None] & n_mask[None, :]
        st_vals = tl.load(new_states_ptr + st_base + st_offs, mask=st_mask, other=0.0)  # [BLOCK_P, 16]

        # C[b,c,l,h,n]
        c_base = b * (C * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
        c_vals = tl.load(C_ptr + c_base + n_offs, mask=n_mask, other=0.0)  # [16]

        acc += tl.sum(st_vals * c_vals[None, :], axis=1)

    # Y_off[b,c,l,h,p] = decay * acc
    y_base = b * (C * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
    tl.store(Y_off_ptr + y_base + p_offs, decay * acc, mask=p_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 7: Compute decay_chunk matrix for inter-chunk recurrence
# chunk_sums = A_cumsum[:,:,:,-1]  [B, H, C]
# pad to [B, H, C+1] with 0 prepended
# cumsum over last dim -> [B, H, C+1]
# segsum -> decay_chunk[b,h,i,j] = exp(cumsum[i] - cumsum[j]) if j<=i else 0
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_X': 128}, num_warps=8),
        triton.Config({'BLOCK_X': 256}, num_warps=8),
    ],
    key=['B', 'H', 'Cp1'],
)
@triton.jit
def decay_chunk_kernel(
    B_: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Each program handles one (b, h) pair, computing the full Cp1 x Cp1 matrix
    pid = tl.program_id(0)
    if pid >= B_ * H:
        return
    b = pid // H
    h = pid % H

    # Build padded cumsum of chunk sums
    # chunk_sums[c] = A_cumsum[b,h,c,-1]
    # padded = [0, chunk_sums[0], chunk_sums[1], ...]
    # cumsum of padded = [0, cs[0], cs[0]+cs[1], ...]
    # Actually the code does:
    #   select last element of A_cumsum -> [B,H,C]
    #   pad with 0 on left -> [B,H,C+1]
    #   cumsum -> [B,H,C+1]