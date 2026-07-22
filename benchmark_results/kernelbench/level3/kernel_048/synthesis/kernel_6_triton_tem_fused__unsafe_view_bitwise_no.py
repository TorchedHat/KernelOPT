import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Constants
BATCH = 2048
SEQ_LEN = 128
N_HEADS = 8
D_HEAD = 64
D_STATE = 16
BLOCK_LEN = 64
N_CHUNKS = SEQ_LEN // BLOCK_LEN  # = 2

# ─── Kernel 1: cumsum along last dim with permute ───────────────────────────

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
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK_X: tl.constexpr,
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

    # Load L elements from inp[b, c, :, h] (shape [B, C, L, H], strides [C*L*H, L*H, H, 1])
    offs = tl.arange(0, BLOCK_X)
    acc = tl.zeros([BLOCK_X], dtype=tl.float32)
    running = tl.zeros([1], dtype=tl.float32)

    # inp stride: [C*L*H, L*H, H, 1]
    inp_base = b * (C * L * H) + c * (L * H) + h

    out_base = b * (H * C * L) + h * (C * L) + c * L

    # Process in chunks of BLOCK_X
    for start in range(0, L, BLOCK_X):
        mask = (start + offs) < L
        vals = tl.load(inp_ptr + inp_base + (start + offs) * H, mask=mask, other=0.0)
        # prefix sum within block
        # Use associative scan pattern
        v = vals.to(tl.float32)
        # manual prefix sum
        v0 = v
        # Triton doesn't have a direct prefix sum, use cumsum via loop unrolling
        # Instead, compute prefix sum manually
        # We'll do it element by element for correctness
        # Actually use tl.cumsum if available, otherwise manual
        cs = tl.cumsum(v, axis=0)
        cs = cs + running
        running = tl.sum(v, axis=0, keep_dims=True)
        # Store
        tl.store(out_ptr + out_base + start + offs, cs, mask=mask)


# ─── Kernel 2: segsum -> exp (L matrix) ─────────────────────────────────────
# Input: A_cumsum [B, H, C, L], Output: L_mat [B, H, C, L, L]

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
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Each program handles one (b, h, c, row_i) -> computes L values
    pid = tl.program_id(0)
    total = B * H * C * L
    if pid >= total:
        return
    b = pid // (H * C * L)
    rem = pid % (H * C * L)
    h = rem // (C * L)
    rem2 = rem % (C * L)
    c = rem2 // L
    i = rem2 % L  # row index

    # cumsum[b, h, c, i]
    cs_base = b * (H * C * L) + h * (C * L) + c * L
    cs_i = tl.load(cumsum_ptr + cs_base + i)

    offs = tl.arange(0, BLOCK_L)
    mask = offs < L

    cs_j = tl.load(cumsum_ptr + cs_base + offs, mask=mask, other=0.0)

    diff = cs_i - cs_j  # [L]

    # Apply tril mask: valid if j <= i (i.e., offs <= i)
    tril_mask = offs <= i
    val = tl.where(tril_mask, diff, float('-inf'))
    result = libdevice.exp(val)

    # Store to out[b, h, c, i, j]
    out_base = b * (H * C * L * L) + h * (C * L * L) + c * (L * L) + i * L
    tl.store(out_ptr + out_base + offs, result, mask=mask)


# ─── Kernel 3: Y_diag einsum ────────────────────────────────────────────────
# Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# = sum_s L[b,h,c,l,s] * (sum_n C[b,c,l,h,n]*B[b,c,s,h,n]) * X[b,c,s,h,p]
# Rewrite: for each (b,c,l,h,p):
#   Y_diag = sum_s L[b,h,c,l,s] * X[b,c,s,h,p] * (C[b,c,l,h,:] · B[b,c,s,h,:])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 64}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 64}, num_warps=8),
    ],
    key=['B', 'H', 'C', 'L', 'P', 'N'],
)
@triton.jit
def ydiag_kernel(
    B_: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # Each program: (b, c, l, h, p_tile)
    pid = tl.program_id(0)
    n_p_tiles = tl.cdiv(P, BLOCK_P)
    total = B_ * C_ * L * H * n_p_tiles
    if pid >= total:
        return

    b = pid // (C_ * L * H * n_p_tiles)
    rem = pid % (C_ * L * H * n_p_tiles)
    c = rem // (L * H * n_p_tiles)
    rem2 = rem % (L * H * n_p_tiles)
    l = rem2 // (H * n_p_tiles)
    rem3 = rem2 % (H * n_p_tiles)
    h = rem3 // n_p_tiles
    p_tile = rem3 % n_p_tiles

    p_offs = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # C[b, c, l, h, :] shape [N]
    C_base = b * (C_ * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
    C_vec = tl.load(C_ptr + C_base + tl.arange(0, N))  # [N]

    # L[b, h, c, l, :] shape [L] (the row for position l)
    L_base = b * (H * C_ * L * L) + h * (C_ * L * L) + c * (L * L) + l * L

    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    for s_start in range(0, L, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < L

        # L[b,h,c,l,s] for s in s_offs
        L_vals = tl.load(L_ptr + L_base + s_offs, mask=s_mask, other=0.0)  # [BLOCK_S]

        # B[b, c, s, h, :] for each s
        # B_base for s: b*(C*L*H*N) + c*(L*H*N) + s*(H*N) + h*N
        B_base_s = b * (C_ * L * H * N) + c * (L * H * N) + h * N
        # For each s in s_offs, load B[b,c,s,h,:] dot C[b,c,l,h,:]
        # This is a [BLOCK_S, N] load
        B_ptrs = B_ptr + B_base_s + s_offs[:, None] * (H * N) + tl.arange(0, N)[None, :]
        B_mat = tl.load(B_ptrs, mask=s_mask[:, None], other=0.0)  # [BLOCK_S, N]

        # dot product C_vec with each row of B_mat -> [BLOCK_S]
        cb_dot = tl.sum(B_mat * C_vec[None, :], axis=1)  # [BLOCK_S]

        # X[b, c, s, h, p] for s in s_offs, p in p_offs
        X_base_s = b * (C_ * L * H * P) + c * (L * H * P) + h * P
        X_ptrs = X_ptr + X_base_s + s_offs[:, None] * (H * P) + p_offs[None, :]
        X_mat = tl.load(X_ptrs, mask=s_mask[:, None] & p_mask[None, :], other=0.0)  # [BLOCK_S, BLOCK_P]

        # acc += sum_s L_vals[s] * cb_dot[s] * X_mat[s, p]
        weight = L_vals * cb_dot  # [BLOCK_S]
        acc += tl.sum(weight[:, None] * X_mat, axis=0)  # [BLOCK_P]

    Y_base = b * (C_ * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
    tl.store(Y_ptr + Y_base + p_offs, acc, mask=p_mask)


# ─── Kernel 4: states einsum ─────────────────────────────────────────────────
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay[b,h,c,l] * X[b,c,l,h,p]
# decay[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_', 'C_', 'L', 'H', 'P', 'N'],
)
@triton.jit
def states_kernel(
    B_: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_p_tiles = tl.cdiv(P, BLOCK_P)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    total = B_ * C_ * H * n_p_tiles * n_n_tiles
    if pid >= total:
        return

    b = pid // (C_ * H * n_p_tiles * n_n_tiles)
    rem = pid % (C_ * H * n_p_tiles * n_n_tiles)
    c = rem // (H * n_p_tiles * n_n_tiles)
    rem2 = rem % (H * n_p_tiles * n_n_tiles)
    h = rem2 // (n_p_tiles * n_n_tiles)
    rem3 = rem2 % (n_p_tiles * n_n_tiles)
    p_tile = rem3 // n_n_tiles
    n_tile = rem3 % n_n_tiles

    p_offs = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    n_offs = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    # Load A_cumsum[b, h, c, -1]
    Acs_base = b * (H * C_ * L) + h * (C_ * L) + c * L
    Acs_last = tl.load(Acumsum_ptr + Acs_base + L - 1)

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    for l in range(L):
        # decay = exp(Acs_last - Acs[l])
        Acs_l = tl.load(Acumsum_ptr + Acs_base + l)
        decay = libdevice.exp(Acs_last - Acs_l)

        # B[b, c, l, h, n_offs]
        B_base = b * (C_ * L * H * N) + c * (L * H * N) + l * (H * N) + h * N
        B_vec = tl.load(B_ptr + B_base + n_offs, mask=n_mask, other=0.0)  # [BLOCK_N]

        # X[b, c, l, h, p_offs]
        X_base = b * (C_ * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
        X_vec = tl.load(X_ptr + X_base + p_offs, mask=p_mask, other=0.0)  # [BLOCK_P]

        # outer product * decay
        acc += decay * X_vec[:, None] * B_vec[None, :]

    # Store to out[b, c, h, p, n]
    out_base = b * (C_ * H * P * N) + c * (H * P * N) + h * (P * N)
    out_ptrs = out_ptr + out_base + p_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, acc, mask=p_mask[:, None] & n_mask[None, :])


# ─── Kernel 5: inter-chunk decay_chunk einsum ────────────────────────────────
# decay_chunk[b,h,i,j] = exp(segsum(A_chunk_cumsum)[b,h,i,j])
# where A_chunk_cumsum[b,h,c] = A_cumsum[b,h,c,-1], padded with 0 at front

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_C': 4}, num_warps=8),
        triton.Config({'BLOCK_C': 8}, num_warps=8),
    ],
    key=['B_', 'H', 'C_'],
)
@triton.jit
def decay_chunk_kernel(
    B_: tl.constexpr, H: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Compute cumsum of A_chunk = [0, Acs[0,-1], Acs[1,-1], ...]
    # Then segsum and exp
    pid = tl.program_id(0)
    total = B_ * H
    if pid >= total:
        return
    b = pid // H
    h = pid % H

    # Build chunk_cumsum: size C_+1
    # chunk_cumsum[0] = 0
    # chunk_cumsum[i+1] = chunk_cumsum[i] + A_cumsum[b,h,i,-1]
    Acs_base = b * (H * C_ * L) + h * (C_ * L)

    # Compute prefix sums
    chunk_cs = tl.zeros([BLOCK_C], dtype=tl.float32)
    # We need C_+1 values; C_=2 so C_+1=3
    # Load all C_ last values
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C_
    last_vals = tl.load(Acumsum_ptr + Acs_base + offs * L + (L - 1), mask=mask, other=0.0)

    # Build chunk_cumsum array of size C_+1
    # chunk_cs[0] = 0, chunk_cs[i] = sum(last_vals[:i])
    # For C_=2: [0, last_vals[0], last_vals[0]+last_vals[1]]
    # We'll compute this inline for the segsum

    # out[b,h,i,j] = exp(chunk_cs[i] - chunk_cs[j]) if i >= j else -inf
    # Size: (C_+1) x (C_+1) = 3x3
    T = C_ + 1
    out_base = b * (H * T * T) + h * (T * T)

    # Compute chunk_cs values
    # chunk_cs[k] = sum of last_vals[0..k-1]
    for i in range(T):
        cs_i = tl.zeros([1], dtype=tl.float32)
        for k in range(i):
            if k < C_:
                cs_i += tl.load(Acumsum_ptr + Acs_base + k * L + (L - 1))
        for j in range(T):
            cs_j = tl.zeros([1], dtype=tl.float32)
            for k in range(j):
                if k < C_:
                    cs_j += tl.load(Acumsum_ptr + Acs_base + k * L + (L - 1))
            diff = cs_i - cs_j
            if i >= j:
                val = libdevice.exp(diff)
            else:
                val = 0.0
            tl.store(out_ptr + out_base + i * T + j, val)


# ─── Kernel 6: new_states einsum ─────────────────────────────────────────────
# new_states[b,i,h,p,n] = sum_j decay_chunk[b,h,i,j] * states[b,j,h,p,n]
# states has shape [B, C+1, H, P, N] (with initial_states prepended)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_', 'C_', 'H', 'P', 'N'],
)
@triton.jit
def new_states_kernel(
    decay_chunk_ptr,  # [B, H, T, T] where T=C+1
    B_: tl.constexpr, C_: tl.constexpr, H: tl.constexpr,
    P: tl.constexpr, N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    T = C_ + 1
    pid = tl.program_id(0)
    n_p_tiles = tl.cdiv(P, BLOCK_P)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    total = B_ * C_ * H * n_p_tiles * n_n_tiles
    if pid >= total:
        return

    b = pid // (C_ * H * n_p_tiles * n_n_tiles)
    rem = pid % (C_ * H * n_p_tiles * n_n_tiles)
    c = rem // (H * n_p_tiles * n_n_tiles)
    rem2 = rem % (H * n_p_tiles * n_n_tiles)
    h = rem2 // (n_p_tiles * n_n_tiles)
    rem3 = rem2 % (n_p_tiles * n_n_tiles)
    p_tile = rem3 // n_n_tiles
    n_tile = rem3 % n_n_tiles

    p_offs = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    n_offs = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    i = c + 1  # output row index in decay_chunk (1-indexed)

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    for j in range(T):
        # decay_chunk[b, h, i, j]
        dc = tl.load(decay_chunk_ptr + b * (H * T * T) + h * (T * T) + i * T + j)

        # states[b, j, h, p, n]
        s_base = b * (T * H * P * N) + j * (H * P * N) + h * (P * N)
        s_ptrs = states_ptr + s_base + p_offs[:, None] * N + n_offs[None, :]
        s_vals = tl.load(s_ptrs, mask=p_mask[:, None] & n_mask[None, :], other=0.0)

        acc += dc * s_vals

    out_base = b * (C_ * H * P * N) + c * (H * P * N) + h * (P * N)
    out_ptrs = out_ptr + out_base + p_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, acc, mask=p_mask[:, None] & n_mask[None, :])


# ─── Kernel 7: Y_off einsum ──────────────────────────────────────────────────
# Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * decay_l[b,h,c,l] * new_states[b,c,h,p,n]
# where decay_l[b,h,c,l] = exp(A_cumsum[b,h,c,l] - 0) ... wait
# Actually: Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * exp(A_cumsum[b,h,c,l]) * new_states[b,c,h,p,n]
# From the original: Y_off = einsum("bclhn,bhcl,bchpn->bclhp", C_blocks, exp(A_cumsum), new_states)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_N': 16}, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['B_', 'C_', 'L', 'H', 'P', 'N'],
)
@triton.jit
def yoff_kernel(
    B_: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_p_tiles = tl.cdiv(P, BLOCK_P)
    total = B_ * C_ * L * H * n_p_tiles
    if pid >= total:
        return

    b = pid // (C_ * L * H * n_p_tiles)
    rem = pid % (C_ * L * H * n_p_tiles)
    c = rem // (L * H * n_p_tiles)
    rem2 = rem % (L * H * n_p_tiles)
    l = rem2 // (H * n_p_tiles)
    rem3 = rem2 % (H * n_p_tiles)
    h = rem3 // n_p_tiles
    p_tile = rem3 % n_p_tiles

    p_offs = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # exp(A_cumsum[b,h,c,l])
    Acs = tl.load(Acumsum_ptr + b * (H * C_ * L) + h * (C_ * L) + c * L + l)
    decay = libdevice.exp(Acs)

    # C[b, c, l, h, :]
    C_base = b * (C_ * L * H * N) + c * (L * H * N) + l * (H * N) + h * N

    # new_states[b, c, h, :, :]
    NS_base = b * (C_ * H * P * N) + c * (H * P * N) + h * (P * N)

    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    n_offs = tl.arange(0, BLOCK_N)
    for n_start in range(0, N, BLOCK_N):
        n_mask = (n_start + n_offs) < N
        C_vec = tl.load(C_ptr + C_base + n_start + n_offs,