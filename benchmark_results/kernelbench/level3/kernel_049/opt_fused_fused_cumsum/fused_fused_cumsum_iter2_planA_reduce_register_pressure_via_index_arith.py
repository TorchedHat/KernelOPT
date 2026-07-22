import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: cumsum over last dim with permute ───────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 8}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def cumsum_permute_kernel(in_ptr, out_ptr, xnumel, L: tl.constexpr, XBLOCK: tl.constexpr):
    # in_ptr: [B, C, L, H] with strides [C*L*H, L*H, H, 1]  (after reshape from [B, C*L, H])
    # We want out: [B, H, C, L] with cumsum over last dim
    # xnumel = B * H * C
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xidx = xoffset + tl.arange(0, XBLOCK)
    xmask = xidx < xnumel

    # decode: xidx -> (b, h, c)
    # xnumel = B * H * C
    H = 8
    C = 2

    c_idx = xidx % C
    h_idx = (xidx // C) % H
    b_idx = xidx // (C * H)

    # Hoist loop-invariant bases (computed once per program)
    in_base = b_idx * (C * L * H) + c_idx * (L * H) + h_idx
    out_base = b_idx * (H * C * L) + h_idx * (C * L) + c_idx * L

    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for l in range(L):
        # input index: b * C*L*H + c * L*H + l * H + h
        val = tl.load(in_ptr + in_base + l * H, mask=xmask, other=0.0)
        acc = acc + val
        # output index: b * H*C*L + h * C*L + c * L + l
        tl.store(out_ptr + out_base + l, acc, mask=xmask)

def segsum_exp_kernel(
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # One program per (b, h, c, row_i)
    pid = tl.program_id(0)
    total = B * H * C * L
    if pid >= total:
        return

    l_idx = pid % L
    c_idx = (pid // L) % C
    h_idx = (pid // (L * C)) % H
    b_idx = pid // (L * C * H)

    # cumsum base for this (b,h,c): offset = b*H*C*L + h*C*L + c*L
    base = b_idx * (H * C * L) + h_idx * (C * L) + c_idx * L

    # Load cumsum[l_idx] for this row
    cs_i = tl.load(cumsum_ptr + base + l_idx)

    # For each column j in [0, L):
    cols = tl.arange(0, BLOCK_L)
    col_mask = cols < L
    cs_j = tl.load(cumsum_ptr + base + cols, mask=col_mask, other=0.0)

    diff = cs_i - cs_j  # shape [BLOCK_L]

    # mask: upper triangle -> j > l_idx means invalid
    upper = cols > l_idx
    diff = tl.where(upper, float('-inf'), diff)

    result = tl.exp(diff)
    result = tl.where(upper, 0.0, result)

    # Store to out[b, h, c, l_idx, j]
    out_base = b_idx * (H * C * L * L) + h_idx * (C * L * L) + c_idx * (L * L) + l_idx * L
    tl.store(out_ptr + out_base + cols, result, mask=col_mask)


# ─── Kernel 3: decay_states = exp(cumsum[:,:,:,-1:] - cumsum) ─────────────────
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def decay_states_kernel(
    numel,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xidx = xoffset + tl.arange(0, XBLOCK)
    xmask = xidx < numel

    l_idx = xidx % L
    c_idx = (xidx // L) % C
    h_idx = (xidx // (L * C)) % H
    b_idx = xidx // (L * C * H)

    base = b_idx * (H * C * L) + h_idx * (C * L) + c_idx * L
    cs_last = tl.load(cumsum_ptr + base + (L - 1), mask=xmask, other=0.0)
    cs_cur = tl.load(cumsum_ptr + base + l_idx, mask=xmask, other=0.0)
    result = tl.exp(cs_last - cs_cur)
    tl.store(decay_ptr + xidx, result, mask=xmask)


# ─── Kernel 4: inter-chunk cumsum with pad ────────────────────────────────────
# select last element of L dim from cumsum [B,H,C,L] -> [B,H,C]
# pad with 0 at front -> [B,H,C+1]
# cumsum over last dim -> [B,H,C+1]
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 2}, num_warps=2),
        triton.Config({'XBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_warps=4),
    ],
    key=['BH'],
)
@triton.jit
def inter_chunk_cumsum_kernel(
    C: tl.constexpr,
    L: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xidx = xoffset + tl.arange(0, XBLOCK)
    xmask = xidx < BH

    h_idx = xidx % 8
    b_idx = xidx // 8

    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    # First element is 0 (padding)
    tl.store(out_ptr + xidx * (C + 1) + 0, acc, mask=xmask)

    for c in range(C):
        # Load cumsum[b, h, c, L-1]
        in_idx = b_idx * (8 * C * L) + h_idx * (C * L) + c * L + (L - 1)
        val = tl.load(cumsum_ptr + in_idx, mask=xmask, other=0.0)
        acc = acc + val
        tl.store(out_ptr + xidx * (C + 1) + (c + 1), acc, mask=xmask)


# ─── Kernel 5: inter-chunk segsum + exp ───────────────────────────────────────
# inter_cumsum: [B, H, C+1]
# segsum: [B, H, C+1, C+1], then exp
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 4}, num_warps=2),
        triton.Config({'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_C': 8}, num_warps=8),
    ],
    key=['Cp1'],
)
@triton.jit
def inter_segsum_exp_kernel(
    BH,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # one program per (bh, row_i)
    total = BH * Cp1
    if pid >= total:
        return

    i_idx = pid % Cp1
    bh_idx = pid // Cp1

    base = bh_idx * Cp1
    cs_i = tl.load(inter_cumsum_ptr + base + i_idx)

    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < Cp1
    cs_j = tl.load(inter_cumsum_ptr + base + cols, mask=col_mask, other=0.0)

    diff = cs_i - cs_j
    upper = cols > i_idx
    diff = tl.where(upper, float('-inf'), diff)
    result = tl.exp(diff)
    result = tl.where(upper, 0.0, result)

    out_base = bh_idx * (Cp1 * Cp1) + i_idx * Cp1
    tl.store(out_ptr + out_base + cols, result, mask=col_mask)


# ─── Main computation using PyTorch for einsum (heavy lifting) ─────────────────
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

    # ── Step 1: Reshape into blocks ──────────────────────────────────────────
    # X_blocks: [B, C, L, H, d_head]
    X_blocks = x.reshape(B, C, L, H, d_head)
    # A_blocks: [B, C, L, H] -> permute to [B, H, C, L]
    A_blocks = p_a.reshape(B, C, L, H).permute(0, 3, 1, 2).contiguous()
    # B_blocks: [B, C, L, H, d_state]
    B_blocks = p_b.reshape(B, C, L, H, d_state)
    # C_blocks: [B, C, L, H, d_state]
    C_blocks = p_c.reshape(B, C, L, H, d_state)

    # ── Step 2: Cumsum of A ──────────────────────────────────────────────────
    # A_blocks is [B, H, C, L], cumsum over last dim
    A_cumsum = torch.cumsum(A_blocks, dim=-1)  # [B, H, C, L]

    # ── Step 3: Diagonal block outputs ──────────────────────────────────────
    # segsum(A_blocks): for each (b,h,c), compute L[i,j] = cumsum[i] - cumsum[j], masked tril, exp
    # L shape: [B, H, C, L, L]
    
    # Compute segsum using PyTorch (numerically correct)
    # A_cumsum: [B, H, C, L]
    x_cumsum = A_cumsum  # [B, H, C, L]
    # x_segsum[..., i, j] = cumsum[i] - cumsum[j]
    x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)  # [B, H, C, L, L]
    mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
    x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
    L_mat = torch.exp(x_segsum)  # [B, H, C, L, L]

    # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L_mat, X_blocks)
    # Let's break this down:
    # C_blocks: [B, C, L, H, N]  (N=d_state)
    # B_blocks: [B, C, S, H, N]  (S=L)
    # L_mat:    [B, H, C, L, S]
    # X_blocks: [B, C, S, H, P]  (P=d_head)
    # Output:   [B, C, L, H, P]
    
    # Rearrange for efficient computation:
    # Y_diag[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    
    # Step: compute (B * X) contracted over n: BX[b,c,s,h,p] = sum_n B[b,c,s,h,n] * ... 
    # Actually let's do it step by step:
    # First: weighted_X[b,c,l,s,h,p] = L[b,h,c,l,s] * X[b,c,s,h,p]
    # Then sum over s: tmp[b,c,l,h,p] = sum_s weighted_X[b,c,l,s,h,p]
    # Then contract with C over n... wait, C doesn't appear in that.
    
    # Let me re-read: einsum("bclhn,bcshn,bhcls,bcshp->bclhp")
    # indices: b,c,l,h,n,s,p
    # C[b,c,l,h,n], B[b,c,s,h,n], L[b,h,c,l,s], X[b,c,s,h,p]
    # = sum_{s,n} C[b,c,l,h,n]*B[b,c,s,h,n]*L[b,h,c,l,s]*X[b,c,s,h,p]
    
    # Group: sum_n C[b,c,l,h,n]*B[b,c,s,h,n] = CB[b,c,l,s,h]  (dot product over n)
    # Then: sum_s CB[b,c,l,s,h] * L[b,h,c,l,s] * X[b,c,s,h,p]
    
    # CB: [B, C, L, S, H] via einsum("bclhn,bcshn->bclsh")
    CB = torch.einsum('bclhn,bcshn->bclsh', C_blocks, B_blocks)  # [B, C, L, S, H]
    
    # L_mat: [B, H, C, L, S] -> permute to [B, C, L, S, H]
    L_mat_perm = L_mat.permute(0, 2, 3, 4, 1)  # [B, C, L, S, H]
    
    # CBL: [B, C, L, S, H]
    CBL = CB * L_mat_perm  # [B, C, L, S, H]
    
    # X_blocks: [B, C, S, H, P]
    # Y_diag[b,c,l,h,p] = sum_s CBL[b,c,l,s,h] * X[b,c,s,h,p]
    # = einsum("bclsh,bcshp->bclhp", CBL, X_blocks)
    Y_diag = torch.einsum('bclsh,bcshp->bclhp', CBL, X_blocks)  # [B, C, L, H, P]

    # ── Step 4: Intra-chunk states ───────────────────────────────────────────
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)  [B, H, C, L]
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, H, C, L]
    
    # states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
    # B_blocks: [B, C, L, H, N]
    # decay_states: [B, H, C, L]
    # X_blocks: [B, C, L, H, P]
    # states[b,c,h,p,n] = sum_{l} B[b,c,l,h,n] * decay[b,h,c,l] * X[b,c,l,h,p]
    
    # decay_states permuted to [B, C, L, H]
    decay_perm = decay_states.permute(0, 2, 3, 1)  # [B, C, L, H]
    
    # BX[b,c,l,h,p,n] = B[b,c,l,h,n] * X[b,c,l,h,p] * decay[b,c,l,h]
    # states[b,c,h,p,n] = sum_l BX[b,c,l,h,p,n]
    states = torch.einsum('bclhn,bclh,bclhp->bchpn', B_blocks, decay_perm, X_blocks)  # [B, C, H, P, N]

    # ── Step 5: Inter-chunk recurrence ───────────────────────────────────────
    # initial_states = zeros_like(states[:, :1])  [B, 1, H, P, N]
    initial_states = torch.zeros_like(states[:, :1])
    
    # states = cat([initial_states, states], dim=1)  [B, C+1, H, P, N]
    states = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N]

    # select last chunk of A_cumsum: A_cumsum[:,:,:,-1] -> [B, H, C]
    # pad with 0 at front -> [B, H, C+1]
    # cumsum -> [B, H, C+1]
    A_tail = A_cumsum[:, :, :, -1]  # [B, H, C]
    A_tail_padded = torch.nn.functional.pad(A_tail, (1, 0), value=0.0)  # [B, H, C+1]
    inter_cumsum = torch.cumsum(A_tail_padded, dim=-1)  # [B, H, C+1]

    # inter segsum: [B, H, C+1, C+1]
    inter_segsum = inter_cumsum.unsqueeze(-1) - inter_cumsum.unsqueeze(-2)  # [B, H, C+1, C+1]
    mask2 = torch.tril(torch.ones(C + 1, C + 1, device=device, dtype=torch.bool))
    inter_segsum = inter_segsum.masked_fill(~mask2, float('-inf'))
    inter_L = torch.exp(inter_segsum)  # [B, H, C+1, C+1]

    # new_states = einsum("bhzc,bchpn->bzhpn", inter_L, states)
    # inter_L: [B, H, C+1, C+1]  (z, c indices)
    # states:  [B, C+1, H, P, N]
    # new_states[b,z,h,p,n] = sum_c inter_L[b,h,z,c] * states[b,c,h,p,n]
    new_states = torch.einsum('bhzc,bchpn->bzhpn', inter_L, states)  # [B, C+1, H, P, N]

    # select last chunk: new_states[:, -1]  [B, H, P, N]  -> but we need [B, H, d_head, d_state]
    # Actually select.int on dim=1 with index C (last): shape [B, H, P, N]
    final_state = new_states[:, -1]  # [B, H, P, N] = [B, H, d_head, d_state]

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
        device = p_a.device

        # Reference implementation
        X_blocks = x.reshape(B, C, L, H, d_head)
        A_blocks = p_a.reshape(B, C, L, H).permute(0, 3, 1, 2).contiguous()
        B_blocks = p_b.reshape(B, C, L, H, d_state)
        C_blocks = p_c.reshape(B, C, L, H, d_state)

        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        x_cumsum = A_cumsum
        x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
        mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
        L_mat = torch.exp(x_segsum)

        CB = torch.einsum('bclhn,bcshn->bclsh', C_blocks, B_blocks)
        L_mat_perm = L_mat.permute(0, 2, 3, 4, 1)
        CBL = CB * L_mat_perm
        Y_diag = torch.einsum('bclsh,bcshp->bclhp', CBL, X_blocks)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        decay_perm = decay_states.permute(0, 2, 3, 1)
        states = torch.einsum('bclhn,bclh,bclhp->bchpn', B_blocks, decay_perm, X_blocks)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        A_tail = A_cumsum[:, :, :, -1]
        A_tail_padded = torch.nn.functional.pad(A_tail, (1, 0), value=0.0)
        inter_cumsum = torch.cumsum(A_tail_padded, dim=-1)

        inter_segsum = inter_cumsum.unsqueeze(-1) - inter_cumsum.unsqueeze(-2)
        mask2 = torch.tril(torch.ones(C + 1, C + 1, device=device, dtype=torch.bool))
        inter_segsum = inter_segsum.masked_fill(~mask2, float('-inf'))
        inter_L = torch.exp(inter_segsum)

        new_states = torch.einsum('bhzc,bchpn->bzhpn', inter_L, states)
        ref = new_states[:, -1]

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")