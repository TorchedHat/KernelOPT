import torch
import triton
import triton.language as tl
import math

# ─── Fused LayerNorm kernel ───────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 768}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    stride_xm,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_xm
    out_ptr = out_ptr + row * stride_xm

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def fused_layernorm(x, weight, bias, eps=1e-5):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)
    grid = (M,)
    layernorm_kernel[grid](
        x, weight, bias, out,
        M, N, eps,
        x.stride(-2),
    )
    return out


# ─── Fused attention bias + masked_fill + softmax kernel ─────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 64}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=8),
        triton.Config({'BLOCK_SEQ': 256}, num_warps=8),
        triton.Config({'BLOCK_SEQ': 512}, num_warps=16),
    ],
    key=['SEQ_LEN'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, bias_ptr, out_ptr,
    B, NH, SEQ_LEN,
    stride_b, stride_h, stride_m,
    scale,
    BLOCK_SEQ: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    pid = tl.program_id(0)
    total_rows = B * NH * SEQ_LEN
    if pid >= total_rows:
        return

    b_idx = pid // (NH * SEQ_LEN)
    h_idx = (pid // SEQ_LEN) % NH
    m_idx = pid % SEQ_LEN

    row_ptr = att_ptr + b_idx * stride_b + h_idx * stride_h + m_idx * stride_m
    out_row_ptr = out_ptr + b_idx * stride_b + h_idx * stride_h + m_idx * stride_m

    # Scale and apply causal mask, then softmax
    # First pass: find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, SEQ_LEN, BLOCK_SEQ):
        cols = off + tl.arange(0, BLOCK_SEQ)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        x = x * scale
        # causal mask: positions > m_idx get -inf
        causal_mask = cols <= m_idx
        x = tl.where(causal_mask & mask, x, float('-inf'))
        # bias mask: load from bias (1,1,SEQ_LEN,SEQ_LEN)
        bias_val = tl.load(bias_ptr + m_idx * 1024 + cols, mask=mask, other=0.0).to(tl.float32)
        is_zero = bias_val == 0.0
        x = tl.where(is_zero & mask, float('-inf'), x)
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, x, float('-inf')), axis=0))

    # Second pass: compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, SEQ_LEN, BLOCK_SEQ):
        cols = off + tl.arange(0, BLOCK_SEQ)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        x = x * scale
        causal_mask = cols <= m_idx
        x = tl.where(causal_mask & mask, x, float('-inf'))
        bias_val = tl.load(bias_ptr + m_idx * 1024 + cols, mask=mask, other=0.0).to(tl.float32)
        is_zero = bias_val == 0.0
        x = tl.where(is_zero & mask, float('-inf'), x)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Third pass: normalize and store
    for off in range(0, SEQ_LEN, BLOCK_SEQ):
        cols = off + tl.arange(0, BLOCK_SEQ)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        x = x * scale
        causal_mask = cols <= m_idx
        x = tl.where(causal_mask & mask, x, float('-inf'))
        bias_val = tl.load(bias_ptr + m_idx * 1024 + cols, mask=mask, other=0.0).to(tl.float32)
        is_zero = bias_val == 0.0
        x = tl.where(is_zero & mask, float('-inf'), x)
        out_val = tl.exp(x - row_max) / row_sum
        tl.store(out_row_ptr + cols, out_val.to(tl.float32), mask=mask)


# ─── Fused GELU kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + triton.language.extra.libdevice.tanh(inner))
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def fused_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ─── Fused residual add kernel ────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def fused_add(a, b):
    out = torch.empty_like(a)
    N = a.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, out, N)
    return out


# ─── Main kernel function ─────────────────────────────────────────────────────
def kernel_function(
    p_ln_1_weight, p_ln_1_bias,
    p_attn_c_attn_weight, p_attn_c_attn_bias,
    p_attn_c_proj_weight, p_attn_c_proj_bias,
    p_ln_2_weight, p_ln_2_bias,
    p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
    p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
    b_attn_bias,
    x,
):
    B, SEQ, C = x.shape  # 128, 512, 768
    n_head = 8
    head_dim = C // n_head  # 96
    scale = 1.0 / math.sqrt(head_dim)

    # ── 1. LayerNorm 1 ────────────────────────────────────────────────────────
    ln1_out = fused_layernorm(x, p_ln_1_weight, p_ln_1_bias)

    # ── 2. QKV projection (cuBLAS) ────────────────────────────────────────────
    ln1_2d = ln1_out.view(B * SEQ, C)
    qkv_2d = ln1_2d @ p_attn_c_attn_weight.t() + p_attn_c_attn_bias
    qkv = qkv_2d.view(B, SEQ, 3 * C)

    # ── 3. Split Q, K, V ──────────────────────────────────────────────────────
    q, k, v = qkv.split(C, dim=2)
    # Reshape to (B, n_head, SEQ, head_dim)
    q = q.view(B, SEQ, n_head, head_dim).transpose(1, 2).contiguous()
    k = k.view(B, SEQ, n_head, head_dim).transpose(1, 2).contiguous()
    v = v.view(B, SEQ, n_head, head_dim).transpose(1, 2).contiguous()

    # ── 4. Attention scores ───────────────────────────────────────────────────
    # (B, n_head, SEQ, SEQ)
    att_raw = q @ k.transpose(-2, -1)  # (B, nh, SEQ, SEQ)

    # ── 5. Scale + causal mask + softmax (fused) ──────────────────────────────
    att_out = torch.empty_like(att_raw)
    # bias shape: (1, 1, 1024, 1024), we need [:, :, :SEQ, :SEQ]
    bias_slice = b_attn_bias[0, 0, :SEQ, :SEQ].contiguous()  # (SEQ, SEQ)

    total_rows = B * n_head * SEQ
    grid_attn = (total_rows,)
    attn_softmax_kernel[grid_attn](
        att_raw, bias_slice, att_out,
        B, n_head, SEQ,
        att_raw.stride(0), att_raw.stride(1), att_raw.stride(2),
        scale,
    )

    # ── 6. Attention output ───────────────────────────────────────────────────
    # (B, nh, SEQ, SEQ) @ (B, nh, SEQ, head_dim) -> (B, nh, SEQ, head_dim)
    y = att_out @ v
    y = y.transpose(1, 2).contiguous().view(B, SEQ, C)

    # ── 7. Output projection ──────────────────────────────────────────────────
    y_2d = y.view(B * SEQ, C)
    proj_out = y_2d @ p_attn_c_proj_weight.t() + p_attn_c_proj_bias
    proj_out = proj_out.view(B, SEQ, C)

    # ── 8. Residual add 1 ─────────────────────────────────────────────────────
    h = fused_add(x, proj_out)

    # ── 9. LayerNorm 2 ────────────────────────────────────────────────────────
    ln2_out = fused_layernorm(h, p_ln_2_weight, p_ln_2_bias)

    # ── 10. MLP FC ────────────────────────────────────────────────────────────
    ln2_2d = ln2_out.view(B * SEQ, C)
    fc_out = ln2_2d @ p_mlpf_c_fc_weight.t() + p_mlpf_c_fc_bias
    fc_out = fc_out.view(B, SEQ, 3072)

    # ── 11. GELU activation ───────────────────────────────────────────────────
    gelu_out = fused_gelu(fc_out)

    # ── 12. MLP projection ────────────────────────────────────────────────────
    gelu_2d = gelu_out.view(B * SEQ, 3072)
    mlp_out = gelu_2d @ p_mlpf_c_proj_weight.t() + p_mlpf_c_proj_bias
    mlp_out = mlp_out.view(B, SEQ, C)

    # ── 13. Residual add 2 ────────────────────────────────────────────────────
    out = fused_add(h, mlp_out)

    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    B, SEQ, C = 128, 512, 768
    n_head = 8
    max_seqlen = 1024

    p_ln_1_weight = torch.randn(C, device=device)
    p_ln_1_bias = torch.randn(C, device=device)
    p_attn_c_attn_weight = torch.randn(3 * C, C, device=device) * 0.02
    p_attn_c_attn_bias = torch.randn(3 * C, device=device) * 0.02
    p_attn_c_proj_weight = torch.randn(C, C, device=device) * 0.02
    p_attn_c_proj_bias = torch.randn(C, device=device) * 0.02
    p_ln_2_weight = torch.randn(C, device=device)
    p_ln_2_bias = torch.randn(C, device=device)
    p_mlpf_c_fc_weight = torch.randn(3072, C, device=device) * 0.02
    p_mlpf_c_fc_bias = torch.randn(3072, device=device) * 0.02
    p_mlpf_c_proj_weight = torch.randn(C, 3072, device=device) * 0.02
    p_mlpf_c_proj_bias = torch.randn(C, device=device) * 0.02
    b_attn_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device=device)).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(B, SEQ, C, device=device) * 0.02

    return [
        p_ln_1_weight, p_ln_1_bias,
        p_attn_c_attn_weight, p_attn_c_attn_bias,
        p_attn_c_proj_weight, p_attn_c_proj_bias,
        p_ln_2_weight, p_ln_2_bias,
        p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
        p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
        b_attn_bias,
        x,
    ]


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
        import torch.nn.functional as F

        (p_ln_1_weight, p_ln_1_bias,
         p_attn_c_attn_weight, p_attn_c_attn_bias,
         p_attn_c_proj_weight, p_attn_c_proj_bias,
         p_ln_2_weight, p_ln_2_bias,
         p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
         p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
         b_attn_bias,
         x) = inputs

        B, SEQ, C = x.shape
        n_head = 8
        head_dim = C // n_head
        scale = 1.0 / math.sqrt(head_dim)

        # LayerNorm 1
        ln1 = F.layer_norm(x, [C], p_ln_1_weight, p_ln_1_bias)

        # QKV
        qkv = F.linear(ln1, p_attn_c_attn_weight, p_attn_c_attn_bias)
        q_r, k_r, v_r = qkv.split(C, dim=2)
        q_r = q_r.view(B, SEQ, n_head, head_dim).transpose(1, 2)
        k_r = k_r.view(B, SEQ, n_head, head_dim).transpose(1, 2)
        v_r = v_r.view(B, SEQ, n_head, head_dim).transpose(1, 2)

        # Attention
        att_r = (q_r @ k_r.transpose(-2, -1)) * scale
        att_r = att_r.masked_fill(b_attn_bias[:, :, :SEQ, :SEQ] == 0, float('-inf'))
        att_r = F.softmax(att_r, dim=-1)
        y_r = att_r @ v_r
        y_r = y_r.transpose(1, 2).contiguous().view(B, SEQ, C)

        # Proj
        proj_r = F.linear(y_r, p_attn_c_proj_weight, p_attn_c_proj_bias)
        h_r = x + proj_r

        # LayerNorm 2
        ln2_r = F.layer_norm(h_r, [C], p_ln_2_weight, p_ln_2_bias)

        # MLP
        fc_r = F.linear(ln2_r, p_mlpf_c_fc_weight, p_mlpf_c_fc_bias)
        # GELU
        gelu_r = 0.5 * fc_r * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (fc_r + 0.044715 * torch.pow(fc_r, 3.0))))
        mlp_r = F.linear(gelu_r, p_mlpf_c_proj_weight, p_mlpf_c_proj_bias)
        ref = h_r + mlp_r

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")