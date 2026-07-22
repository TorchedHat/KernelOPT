import torch
import triton
import triton.language as tl
import math

# ============================================================
# Kernel 1: Fused LayerNorm
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = tl.math.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


# ============================================================
# Kernel 2: Fused GEMM (tl.dot based)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Kernel 3: Fused attention scores: scale + causal mask + softmax
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, bias_ptr, out_ptr,
    B, H, T, scale,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    # program_id(0) = batch * H * T + head * T + row
    pid = tl.program_id(0)
    batch_head = pid // T
    row = pid % T

    att_row_ptr = att_ptr + pid * T
    out_row_ptr = out_ptr + pid * T

    # Load causal mask for this row
    # bias shape: [1, 1, T, T] (already sliced to [T, T])
    bias_row_ptr = bias_ptr + row * T

    # Find max for numerical stability
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        att_val = tl.load(att_row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Apply scale and causal mask
        att_val = att_val * scale
        # bias == 0 means masked (causal mask)
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, att_val, float('-inf')), axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        att_val = tl.load(att_row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        att_val = tl.where(mask, att_val, float('-inf'))
        row_sum += tl.sum(tl.exp(att_val - row_max), axis=0)

    # Store normalized values
    for off in range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        att_val = tl.load(att_row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        att_val = tl.where(mask, att_val, float('-inf'))
        out_val = tl.exp(att_val - row_max) / row_sum
        tl.store(out_row_ptr + cols, out_val.to(tl.float32), mask=mask)


# ============================================================
# Kernel 4: Fused GELU activation (NewGELU)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
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
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


# ============================================================
# Kernel 5: Batched matmul for attention (Q @ K^T and att @ V)
# Uses tl.dot for the inner product
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def kernel_function(
    p_ln_1_weight,
    p_ln_1_bias,
    p_attn_c_attn_weight,
    p_attn_c_attn_bias,
    p_attn_c_proj_weight,
    p_attn_c_proj_bias,
    p_ln_2_weight,
    p_ln_2_bias,
    p_mlpf___closure___0_cell_contents_c_fc_weight,
    p_mlpf___closure___0_cell_contents_c_fc_bias,
    p_mlpf___closure___0_cell_contents_c_proj_weight,
    p_mlpf___closure___0_cell_contents_c_proj_bias,
    b_attn_bias,
    x,
):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    head_dim = C // n_head  # 96
    device = x.device

    # ---- Step 1: LayerNorm 1 ----
    ln1_out = torch.empty_like(x)
    grid_ln = (B * T,)
    layernorm_kernel[grid_ln](
        x, p_ln_1_weight, p_ln_1_bias, ln1_out,
        B * T, C, 1e-5,
    )

    # ---- Step 2: QKV projection ----
    # ln1_out: [B*T, C] @ W^T [C, 3C] + bias [3C] -> [B*T, 3C]
    ln1_flat = ln1_out.view(B * T, C)
    qkv = torch.empty(B * T, 3 * C, device=device, dtype=torch.float32)
    grid_qkv = (triton.cdiv(B * T, 128), triton.cdiv(3 * C, 128))
    matmul_bias_kernel[grid_qkv](
        ln1_flat, p_attn_c_attn_weight, p_attn_c_attn_bias, qkv,
        B * T, 3 * C, C,
        1, C,       # stride_bk, stride_bn  (weight is [3C, C], we want W^T so B[k,n] = weight[n,k])
    )

    # ---- Step 3: Split QKV and reshape ----
    qkv_3d = qkv.view(B, T, 3 * C)
    q = qkv_3d[:, :, :C].view(B, T, n_head, head_dim).transpose(1, 2).contiguous()  # [B, H, T, hs]
    k = qkv_3d[:, :, C:2*C].view(B, T, n_head, head_dim).transpose(1, 2).contiguous()  # [B, H, T, hs]
    v = qkv_3d[:, :, 2*C:].view(B, T, n_head, head_dim).transpose(1, 2).contiguous()  # [B, H, T, hs]

    # ---- Step 4: Q @ K^T -> [B, H, T, T] ----
    # q: [B, H, T, hs], k: [B, H, T, hs] -> k^T: [B, H, hs, T]
    # att: [B, H, T, T]
    BH = B * n_head
    att = torch.empty(BH, T, T, device=device, dtype=torch.float32)
    q_flat = q.reshape(BH, T, head_dim)
    k_flat = k.reshape(BH, T, head_dim)

    grid_att = (triton.cdiv(T, 32), triton.cdiv(T, 32), BH)
    batched_matmul_kernel[grid_att](
        q_flat, k_flat, att,
        T, T, head_dim,
    )

    # ---- Step 5: Scale + causal mask + softmax ----
    scale = 1.0 / math.sqrt(head_dim)
    # b_attn_bias: [1, 1, 1024, 1024] -> slice to [T, T]
    causal_bias = b_attn_bias[0, 0, :T, :T].contiguous()  # [T, T]

    att_softmax = torch.empty(BH, T, T, device=device, dtype=torch.float32)
    grid_softmax = (BH * T,)
    attn_softmax_kernel[grid_softmax](
        att, causal_bias, att_softmax,
        B, n_head, T, scale,
    )

    # ---- Step 6: att @ V -> [B, H, T, hs] ----
    att_flat = att_softmax  # [BH, T, T]
    v_flat = v.reshape(BH, T, head_dim)
    y_att = torch.empty(BH, T, head_dim, device=device, dtype=torch.float32)

    grid_av = (triton.cdiv(T, 32), triton.cdiv(head_dim, 32), BH)
    batched_matmul_kernel[grid_av](
        att_flat, v_flat, y_att,
        T, head_dim, T,
    )

    # ---- Step 7: Reshape and project ----
    y_att_3d = y_att.reshape(B, n_head, T, head_dim).transpose(1, 2).contiguous().view(B * T, C)

    # c_proj: [B*T, C] @ W^T [C, C] + bias [C] -> [B*T, C]
    proj_out = torch.empty(B * T, C, device=device, dtype=torch.float32)
    grid_proj = (triton.cdiv(B * T, 128), triton.cdiv(C, 128))
    matmul_bias_kernel[grid_proj](
        y_att_3d, p_attn_c_proj_weight, p_attn_c_proj_bias, proj_out,
        B * T, C, C,
        C, 1,
        1, C,
        C, 1,
        True,
    )

    # ---- Step 8: Residual add (attn output + x) ----
    # proj_out: [B*T, C], x: [B, T, C]
    x_flat = x.view(B * T, C)
    residual1 = proj_out + x_flat  # [B*T, C]

    # ---- Step 9: LayerNorm 2 ----
    ln2_out = torch.empty(B * T, C, device=device, dtype=torch.float32)
    grid_ln2 = (B * T,)
    layernorm_kernel[grid_ln2](
        residual1, p_ln_2_weight, p_ln_2_bias, ln2_out,
        B * T, C, 1e-5,
    )

    # ---- Step 10: MLP fc: [B*T, C] @ W^T [C, 4C] + bias [4C] -> [B*T, 4C] ----
    fc_out = torch.empty(B * T, 4 * C, device=device, dtype=torch.float32)
    grid_fc = (triton.cdiv(B * T, 128), triton.cdiv(4 * C, 128))
    matmul_bias_kernel[grid_fc](
        ln2_out, p_mlpf___closure___0_cell_contents_c_fc_weight,
        p_mlpf___closure___0_cell_contents_c_fc_bias, fc_out,
        B * T, 4 * C, C,
        C, 1,
        1, C,
        4 * C, 1,
        True,
    )

    # ---- Step 11: NewGELU activation ----
    gelu_out = torch.empty_like(fc_out)
    N_gelu = B * T * 4 * C
    grid_gelu = (triton.cdiv(N_gelu, 1024),)
    gelu_kernel[grid_gelu](
        fc_out, gelu_out,
        N_gelu,
    )

    # ---- Step 12: MLP proj: [B*T, 4C] @ W^T [4C, C] + bias [C] -> [B*T, C] ----
    mlp_proj_out = torch.empty(B * T, C, device=device, dtype=torch.float32)
    grid_mlp_proj = (triton.cdiv(B * T, 128), triton.cdiv(C, 128))
    matmul_bias_kernel[grid_mlp_proj](
        gelu_out, p_mlpf___closure___0_cell_contents_c_proj_weight,
        p_mlpf___closure___0_cell_contents_c_proj_bias, mlp_proj_out,
        B * T, C, 4 * C,
        4 * C, 1,
        1, 4 * C,
        C, 1,
        True,
    )

    # ---- Step 13: Residual add (mlp output + residual1) ----
    output = (mlp_proj_out + residual1).view(B, T, C)

    return output


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    B, T, C = 128, 512, 768
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
    p_mlp_fc_weight = torch.randn(4 * C, C, device=device) * 0.02
    p_mlp_fc_bias = torch.randn(4 * C, device=device) * 0.02
    p_mlp_proj_weight = torch.randn(C, 4 * C, device=device) * 0.02
    p_mlp_proj_bias = torch.randn(C, device=device) * 0.02
    b_attn_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device=device)).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(B, T, C, device=device) * 0.02

    return [
        p_ln_1_weight, p_ln_1_bias,
        p_attn_c_attn_weight, p_attn_c_attn_bias,
        p_attn_c_proj_weight, p_attn_c_proj_bias,
        p_ln_2_weight, p_ln_2_bias,
        p_mlp_fc_weight, p_mlp_fc_bias,
        p_mlp_proj_weight, p_mlp_proj_bias,
        b_attn_bias, x,
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
        (p_ln_1_weight, p_ln_1_bias,
         p_attn_c_attn_weight, p_attn_c_attn_bias,
         p_attn_c_proj_weight, p_attn_c_proj_bias,
         p_ln_2_weight, p_ln_2_bias,
         p_mlp_fc_weight, p_mlp_fc_bias,
         p_mlp_proj_weight, p_mlp_proj_bias,
         b_attn_bias, x) = inputs

        B, T, C = x.shape
        n_head = 8
        head_dim = C // n_head

        # LayerNorm 1
        import torch.nn.functional as F
        ln1 = F.layer_norm(x, [C], p_ln_1_weight, p_ln_1_bias, eps=1e-5)

        # QKV projection
        qkv = F.linear(ln1, p_attn_c_attn_weight, p_attn_c_attn_bias)
        q_ref, k_ref, v_ref = qkv.split(C, dim=2)
        k_ref = k_ref.view(B, T, n_head, head_dim).transpose(1, 2)
        q_ref = q_ref.view(B, T, n_head, head_dim).transpose(1, 2)
        v_ref = v_ref.view(B, T, n_head, head_dim).transpose(1, 2)

        # Attention
        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att_ref = att_ref.masked_fill(b_attn_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = F.softmax(att_ref, dim=-1)
        y_ref = att_ref @ v_ref
        y_ref = y_ref.transpose(1, 2).contiguous().view(B, T, C)

        # c_proj
        y_ref = F.linear(y_ref, p_attn_c_proj_weight, p_attn_c_proj_bias)
        residual1_ref = y_ref + x

        # LayerNorm 2
        ln2_ref = F.layer_norm(residual1_ref, [C], p_