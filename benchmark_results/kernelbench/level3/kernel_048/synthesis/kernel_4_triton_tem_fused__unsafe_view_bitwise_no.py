import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: permute + cumsum for A ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_warps=2),
        triton.Config({'XBLOCK': 4}, num_warps=2),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def cumsum_permute_kernel(
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Each program handles one (b, h, c) row of length L
    pid = tl.program_id(0)
    total_rows = B * H * C
    row_idx = pid  # (b, h, c)
    if row_idx >= total_rows:
        return
    b = row_idx // (H * C)
    hc = row_idx % (H * C)
    h = hc // C
    c = hc % C

    # Load L elements from in_ptr[b, c*L + 0..L-1, h] = in_ptr[b*(C*L*H) + (c*L+l)*H + h]
    # in_ptr shape: [B, C*L, H], strides: [C*L*H, H, 1]
    CL = C * L
    base_in = b * (CL * H) + c * L * H + h

    # Compute cumsum
    acc = tl.zeros([1], dtype=tl.float32)
    for l in range(L):
        val = tl.load(in_ptr + base_in + l * H)
        acc = acc + val
        # out_ptr[b, h, c, l] -> b*(H*C*L) + h*(C*L) + c*L + l
        out_idx = b * (H * C * L) + h * (C * L) + c * L + l
        tl.store(out_ptr + out_idx, acc)


# ─── Kernel 2: segsum + exp for diagonal blocks ──────────────────────────────
# Computes L[b,h,c,i,j] = exp(cumsum[b,h,c,i] - cumsum[b,h,c,j]) if i>=j else -inf
# Output shape: [B, H, C, L, L]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=8),
        triton.Config({'BLOCK': 32}, num_warps=4),
        triton.Config({'BLOCK': 16}, num_warps=4),
        triton.Config({'BLOCK': 8}, num_warps=2),
    ],
    key=['B', 'H', 'C', 'L'],
)
@triton.jit
def segsum_exp_kernel(
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program handles one (b, h, c) block -> L x L matrix
    pid = tl.program_id(0)
    total = B * H * C
    if pid >= total:
        return
    b = pid // (H * C)
    hc = pid % (H * C)
    h = hc // C
    c = hc % C

    base_cs = b * (H * C * L) + h * (C * L) + c * L
    base_out = b * (H * C * L * L) + h * (C * L * L) + c * (L * L)

    # Load cumsum row [L]
    cs = tl.zeros([64], dtype=tl.float32)  # L=64 fixed
    for l in range(L):
        cs_val = tl.load(cumsum_ptr + base_cs + l)
        # Store in register - we'll use loop approach
        _ = cs_val  # placeholder

    # Process L x L matrix
    for i in range(L):
        cs_i = tl.load(cumsum_ptr + base_cs + i)
        for j in range(L):
            cs_j = tl.load(cumsum_ptr + base_cs + j)
            diff = cs_i - cs_j
            # mask: i >= j (lower triangular)
            mask = i >= j
            val = tl.where(mask, diff, float('-inf'))
            exp_val = tl.exp(val)
            tl.store(out_ptr + base_out + i * L + j, exp_val)


# ─── Kernel 3: Y_diag einsum ─────────────────────────────────────────────────
# Y_diag[b,c,l,h,p] = sum_s sum_n C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
# = sum_s L[b,h,c,l,s] * (sum_n C[b,c,l,h,n]*B[b,c,s,h,n]) * X[b,c,s,h,p]
# Shapes: C_blocks[B,C,L,H,N], B_blocks[B,C,L,H,N], L_mat[B,H,C,L,L], X_blocks[B,C,L,H,P]
# Output: [B,C,L,H,P]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_P': 16, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_S': 16}, num_warps=8),
    ],
    key=['B', 'C', 'L', 'H', 'N', 'P'],
)
@triton.jit
def ydiag_kernel(
    B: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, N: tl.constexpr, P: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # Grid: (B * C_ * L * H, cdiv(P, BLOCK_P))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    total0 = B * C_ * L * H
    if pid0 >= total0:
        return

    b = pid0 // (C_ * L * H)
    clh = pid0 % (C_ * L * H)
    c = clh // (L * H)
    lh = clh % (L * H)
    l = lh // H
    h = lh % H

    p_start = pid1 * BLOCK_P
    p_offs = p_start + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    # For each s in [0, L):
    # contribution = L_mat[b,h,c,l,s] * (sum_n C[b,c,l,h,n]*B[b,c,s,h,n]) * X[b,c,s,h,p]
    for s in range(L):
        # Load L_mat[b,h,c,l,s]
        l_idx = b * (H * C_ * L * L) + h * (C_ * L * L) + c * (L * L) + l * L + s
        l_val = tl.load(L_ptr + l_idx)

        # Compute dot(C[b,c,l,h,:], B[b,c,s,h,:])
        cb_dot = tl.zeros([1], dtype=tl.float32)
        for n in range(N):
            c_idx = b * (C_ * L * H * N) + c * (L * H * N) + l * (H * N) + h * N + n
            b_idx = b * (C_ * L * H * N) + c * (L * H * N) + s * (H * N) + h * N + n
            c_val = tl.load(C_ptr + c_idx)
            b_val = tl.load(B_ptr + b_idx)
            cb_dot += c_val * b_val

        # Load X[b,c,s,h,p_offs]
        x_base = b * (C_ * L * H * P) + c * (L * H * P) + s * (H * P) + h * P
        x_vals = tl.load(X_ptr + x_base + p_offs, mask=p_mask, other=0.0)

        acc += l_val * cb_dot * x_vals

    out_base = b * (C_ * L * H * P) + c * (L * H * P) + l * (H * P) + h * P
    tl.store(out_ptr + out_base + p_offs, acc, mask=p_mask)


# ─── Kernel 4: decay_states + states einsum ──────────────────────────────────
# decay_states[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
# states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay_states[b,h,c,l] * X[b,c,l,h,p]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 32}, num_warps=4),
        triton.Config({'BLOCK_L': 16}, num_warps=2),
        triton.Config({'BLOCK_L': 8}, num_warps=2),
    ],
    key=['B', 'C_', 'L', 'H', 'N', 'P'],
)
@triton.jit
def states_kernel(
    B: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, N: tl.constexpr, P: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Grid: B * C_ * H * P * N
    pid = tl.program_id(0)
    total = B * C_ * H * P * N
    if pid >= total:
        return

    b = pid // (C_ * H * P * N)
    rest = pid % (C_ * H * P * N)
    c = rest // (H * P * N)
    rest2 = rest % (H * P * N)
    h = rest2 // (P * N)
    rest3 = rest2 % (P * N)
    p = rest3 // N
    n = rest3 % N

    # Load A_cumsum[b,h,c,-1]
    ac_last = tl.load(A_cumsum_ptr + b * (H * C_ * L) + h * (C_ * L) + c * L + (L - 1))

    acc = tl.zeros([1], dtype=tl.float32)
    for l in range(L):
        ac_l = tl.load(A_cumsum_ptr + b * (H * C_ * L) + h * (C_ * L) + c * L + l)
        decay = tl.exp(ac_last - ac_l)

        b_val = tl.load(B_ptr + b * (C_ * L * H * N) + c * (L * H * N) + l * (H * N) + h * N + n)
        x_val = tl.load(X_ptr + b * (C_ * L * H * P) + c * (L * H * P) + l * (H * P) + h * P + p)

        acc += b_val * decay * x_val

    # out[b, c, h, p, n]
    out_idx = b * (C_ * H * P * N) + c * (H * P * N) + h * (P * N) + p * N + n
    tl.store(out_ptr + out_idx, acc)


# ─── Kernel 5: decay_chunk segsum exp ────────────────────────────────────────
# For inter-chunk: A_cumsum_last[b,h,c] = A_cumsum[b,h,c,-1]
# pad with 0 at front -> [b,h,3] then cumsum -> decay_chunk[b,h,i,j] = exp(cs[i]-cs[j]) tril

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=2),
    ],
    key=['B', 'H', 'C_'],
)
@triton.jit
def decay_chunk_kernel(
    out_ptr,       # [B, H, C_+1, C_+1]  (3x3 for C_=2)
    B: tl.constexpr, H: tl.constexpr, C_: tl.constexpr, L: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # C_p1 = C_ + 1 = 3
    C_p1: tl.constexpr = C_ + 1
    pid = tl.program_id(0)
    total = B * H * C_p1 * C_p1
    xoffset = pid * XBLOCK
    for xi in range(XBLOCK):
        idx = xoffset + xi
        if idx >= total:
            break
        b = idx // (H * C_p1 * C_p1)
        rest = idx % (H * C_p1 * C_p1)
        h = rest // (C_p1 * C_p1)
        ij = rest % (C_p1 * C_p1)
        i = ij // C_p1
        j = ij % C_p1

        # cs[k] = cumsum of padded A_cumsum_last
        # padded = [0, A_cumsum_last[b,h,0], A_cumsum_last[b,h,1]]
        # cs[k] = sum of padded[0..k]
        # cs[0] = 0
        # cs[1] = A_cumsum_last[b,h,0]
        # cs[2] = A_cumsum_last[b,h,0] + A_cumsum_last[b,h,1]

        def get_cs(k, b_, h_):
            acc_ = 0.0
            for kk in range(C_):
                if kk < k:
                    v = tl.load(A_cumsum_ptr + b_ * (H * C_ * L) + h_ * (C_ * L) + kk * L + (L - 1))
                    acc_ = acc_ + v
            return acc_

        cs_i = get_cs(i, b, h)
        cs_j = get_cs(j, b, h)
        diff = cs_i - cs_j
        mask = i >= j
        val = tl.where(mask, diff, float('-inf'))
        exp_val = tl.exp(val)
        tl.store(out_ptr + idx, exp_val)


# ─── Kernel 6: Y_off einsum ──────────────────────────────────────────────────
# Y_off[b,c,l,h,p] = sum_{c2<c} decay_chunk[b,h,c,c2+1] * exp(A_cumsum[b,h,c2,-1] - A_cumsum[b,h,c,l]) ... 
# Actually from the code:
# new_states = einsum("bhij,bjhpn->bihpn", decay_chunk, states)  [B,H,C_+1,C_+1] x [B,C_+1,H,P,N]
# then slice [:, :C_, ...] -> [B, C_, H, P, N]
# Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * sum_{c2} decay_chunk_states[b,c,h,p,n] * exp(-A_cumsum[b,h,c,l])
# 
# Let me re-read the original more carefully...
# 
# From the original PyTorch:
# decay_chunk = exp(segsum(A_cumsum_last_padded))  [B,H,C+1,C+1]
# new_states = einsum("bhij,bjhpn->bihpn", decay_chunk, states)  -> [B,C+1,H,P,N]
# Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * sum_{c2} new_states[b,c,h,p,n] * exp(A_cumsum[b,h,c,l] - ?)
# 
# Actually the full forward is:
# decay_states_for_off = exp(A_cumsum[b,h,c,l] - 0) ... hmm
# 
# Let me look at the einsum ops more carefully from the ATen graph:
# einsum "bhij,bjhpn->bihpn" -> [B,3,H,P,N] then slice [:,:2,...] -> [B,2,H,P,N]
# then exp(A_cumsum) -> [B,H,2,64]
# then einsum "bclhn,bhcl,bclhp->bclhp" -> Y_off [B,2,64,H,P]

# This is getting complex. Let me implement the full forward pass using a mix of
# Triton kernels for the key operations and careful tensor manipulation.

def kernel_function(p_a, p_b, p_c, x):
    """
    
        p_a: [B, seq_len, H] = [2048, 128, 8]
        p_b: [B, seq_len, H, N] = [2048, 128, 8, 16]
        p_c: [B, seq_len, H, N] = [2048, 128, 8, 16]
        x:   [B, seq_len, H, P] = [2048, 128, 8, 64]
    
        Y: [B, seq_len, H, P] = [2048, 128, 8, 64]
    """
    B, seq_len, H = p_a.shape
    N = p_b.shape[-1]
    P = x.shape[-1]
    block_len = 64
    C_ = seq_len // block_len  # number of chunks = 2

    device = p_a.device
    dtype = p_a.dtype

    # Reshape into blocks
    # X_blocks: [B, C_, L, H, P]
    X_blocks = x.reshape(B, C_, block_len, H, P)
    # A_blocks: [B, C_, L, H]
    A_blocks = p_a.reshape(B, C_, block_len, H)
    # B_blocks: [B, C_, L, H, N]
    B_blocks = p_b.reshape(B, C_, block_len, H, N)
    # C_blocks: [B, C_, L, H, N]
    C_blocks = p_c.reshape(B, C_, block_len, H, N)

    # A_blocks permuted: [B, H, C_, L]
    A_perm = A_blocks.permute(0, 3, 1, 2).contiguous()

    # A_cumsum: [B, H, C_, L]
    A_cumsum = _triton_cumsum_last_dim(A_perm, B, H, C_, block_len)

    # ── Step 1: Diagonal block outputs ──────────────────────────────────────
    # L_mat[b,h,c,i,j] = exp(A_cumsum[b,h,c,i] - A_cumsum[b,h,c,j]) if i>=j else 0
    L_mat = _triton_segsum_exp(A_cumsum, B, H, C_, block_len)
    # L_mat: [B, H, C_, L, L]

    # Y_diag[b,c,l,h,p] = sum_s sum_n C[b,c,l,h,n]*B[b,c,s,h,n]*L[b,h,c,l,s]*X[b,c,s,h,p]
    Y_diag = _triton_ydiag(C_blocks, B_blocks, L_mat, X_blocks, B, C_, block_len, H, N, P)
    # Y_diag: [B, C_, L, H, P]

    # ── Step 2: Intra-chunk states ───────────────────────────────────────────
    # decay_states[b,h,c,l] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l])
    # states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * decay_states[b,h,c,l] * X[b,c,l,h,p]
    states = _triton_states(B_blocks, A_cumsum, X_blocks, B, C_, block_len, H, N, P)
    # states: [B, C_, H, P, N]

    # ── Step 3: Inter-chunk recurrence ───────────────────────────────────────
    # initial_states = zeros [B, 1, H, P, N]
    initial_states = torch.zeros(B, 1, H, P, N, device=device, dtype=dtype)
    # states_cat: [B, C_+1, H, P, N]
    states_cat = torch.cat([initial_states, states], dim=1)

    # A_cumsum_last[b,h,c] = A_cumsum[b,h,c,-1]  -> [B, H, C_]
    A_cumsum_last = A_cumsum[:, :, :, -1]  # [B, H, C_]

    # decay_chunk: segsum_exp of padded A_cumsum_last
    # pad: [0, A_cumsum_last[b,h,0], A_cumsum_last[b,h,1]] -> cumsum -> [B,H,3]
    # then segsum_exp -> [B,H,3,3]
    decay_chunk = _triton_decay_chunk(A_cumsum_last, B, H, C_, device, dtype)
    # decay_chunk: [B, H, C_+1, C_+1]

    # new_states[b,i,h,p,n] = sum_j decay_chunk[b,h,i,j] * states_cat[b,j,h,p,n]
    # einsum "bhij,bjhpn->bihpn"
    new_states = _triton_new_states(decay_chunk, states_cat, B, H, C_+1, P, N)
    # new_states: [B, C_+1, H, P, N], slice to [B, C_, H, P, N]
    new_states = new_states[:, :C_, :, :, :]  # [B, C_, H, P, N]

    # ── Step 4: Y_off ────────────────────────────────────────────────────────
    # decay_for_off[b,h,c,l] = exp(A_cumsum[b,h,c,l])  (since new_states already has decay from chunk start)
    # Actually: Y_off[b,c,l,h,p] = sum_n C[b,c,l,h,n] * sum_p2 new_states[b,c,h,p2,n] * exp(A_cumsum[b,h,c,l]) * delta(p,p2)
    # Wait, let me re-read...
    # 
    # From original:
    # decay_states_for_off = exp(A_cumsum)  [B,H,C_,L]
    # Y_off = einsum("bclhn,bhcl,bchpn->bclhp", C_blocks, decay_states_for_off, new_states)
    # 
    # Actually looking at the ATen ops:
    # exp.default -> [B,H,2,64]  (this is exp(A_cumsum) for Y_off)
    # einsum -> [B,2,64,H,P]  (Y_off)
    # add -> Y_diag + Y_off
    
    decay_for_off = torch.exp(A_cumsum)  # [B, H, C_, L]
    Y_off = _triton_yoff(C_blocks, decay_for_off, new_states, B, C_, block_len, H, N, P)
    # Y_off: [B, C_, L, H, P]

    # ── Final: combine ───────────────────────────────────────────────────────
    Y = Y_diag + Y_off  # [B, C_, L, H, P]
    Y = Y.reshape(B, seq_len, H, P)

    return Y


# ─── Helper: cumsum along last dim ───────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROW': 32}, num_warps=4),
        triton.Config({'BLOCK_ROW': 16}, num_warps=4),
        triton.Config({'BLOCK_ROW': 8}, num_warps=2),
        triton.Config({'BLOCK_ROW': 4}, num_warps=2),
    ],
    key=['num_rows', 'L'],
)
@triton.jit
def _cumsum_last_dim_kernel(
    in_ptr, out_ptr,
    num_rows: tl.constexpr, L: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW
    rows = row_start + tl.arange(0, BLOCK_ROW)
    row_mask = rows < num_rows

    for l in range(L):
        # Load
        in_offs = rows * L + l
        vals = tl.load(in_ptr + in_offs, mask=row_mask, other=0.0)
        if l == 0:
            acc = vals
        else:
            acc = acc + vals
        tl.store(out_ptr + in_offs, acc, mask=row_mask)


def _triton_cumsum_last_dim(x, B, H, C_, L):
    """x: [B, H, C_, L] contiguous, compute cumsum along last dim"""
    out = torch.empty_like(x)
    num_rows = B * H * C_
    grid = lambda meta: (triton.cdiv(num_rows, meta['BLOCK_ROW']),)
    _cumsum_last_dim_kernel[grid](x, out, num_rows, L)
    return out


# ─── Helper: segsum exp ──────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROW': 8}, num_warps=4),
        triton.Config({'BLOCK_ROW': 4}, num_warps=4),
        triton.Config({'BLOCK_ROW': 2}, num_warps=2),
        triton.Config({'BLOCK_ROW': 1}, num_warps=2),
    ],
    key=['num_rows', 'L'],
)
@triton.jit
def _segsum_exp_kernel(
    cumsum_ptr, out_ptr,
    num_rows: tl.constexpr, L: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW
    # Each row -> L x L output
    for br in range(BLOCK_ROW):
        row = row_start + br
        if row >= num_rows:
            break
        cs_base = row * L
        out_base = row * L * L
        for i in range(L):
            cs_i = tl.load(cumsum_ptr + cs_base + i)
            for j in range(L):
                cs_j = tl.load(cumsum_ptr + cs_base + j)
                diff = cs_i - cs_