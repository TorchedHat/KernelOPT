import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: permute+clone (NCHW->seq) ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_Y': 16, 'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_Y': 64, 'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 64}, num_warps=8),
    ],
    key=['SEQ_LEN', 'BATCH', 'DIM'],
)
@triton.jit
def permute_kernel(
    in_ptr, out_ptr,
    SEQ_LEN: tl.constexpr, BATCH: tl.constexpr, DIM: tl.constexpr,
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
):
    # in: (BATCH, DIM, SEQ_LEN) strides=(DIM*SEQ_LEN, SEQ_LEN, 1)
    # out: (SEQ_LEN, BATCH, DIM) strides=(BATCH*DIM, DIM, 1)
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    offs_seq = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_dim = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask_seq = offs_seq < SEQ_LEN
    mask_dim = offs_dim < DIM
    mask = mask_seq[:, None] & mask_dim[None, :]
    for b in range(BATCH):
        in_idx = b * DIM * SEQ_LEN + offs_dim[None, :] * SEQ_LEN + offs_seq[:, None]
        vals = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        out_idx = offs_seq[:, None] * BATCH * DIM + b * DIM + offs_dim[None, :]
        tl.store(out_ptr + out_idx, vals, mask=mask)


# ─── Kernel 2: GEMM (M,K) x (K,N) + bias -> (M,N) ────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc=acc)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ─── Kernel 3: Softmax over last dim ─────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    in_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    row_ptr = in_ptr + row * N
    x = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(out_ptr + row * N + offs, out, mask=mask)


# ─── Kernel 4: BMM: (B, M, K) x (B, K, N) -> (B, M, N) ──────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    A_base = A_ptr + pid_b * stride_ab
    B_base = B_ptr + pid_b * stride_bb
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        a = tl.load(A_base + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_base + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc=acc)
    tl.store(C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ─── Kernel 5: LayerNorm ──────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(in_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / libdevice.sqrt(var + eps)
    x_norm = x_centered * rstd
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b
    tl.store(out_ptr + row * N + offs, out.to(tl.float32), mask=mask)


# ─── Kernel 6: final permute (seq,batch,dim) -> (batch,dim,seq) ───────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_Y': 16, 'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_Y': 64, 'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 64}, num_warps=8),
    ],
    key=['SEQ_LEN', 'BATCH', 'DIM'],
)
@triton.jit
def permute_back_kernel(
    in_ptr, out_ptr,
    SEQ_LEN: tl.constexpr, BATCH: tl.constexpr, DIM: tl.constexpr,
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
):
    # in: (SEQ_LEN, BATCH, DIM) strides=(BATCH*DIM, DIM, 1)
    # out: (BATCH, DIM, SEQ_LEN) strides=(DIM*SEQ_LEN, SEQ_LEN, 1)
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    offs_seq = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_dim = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask_seq = offs_seq < SEQ_LEN
    mask_dim = offs_dim < DIM
    mask = mask_seq[:, None] & mask_dim[None, :]
    for b in range(BATCH):
        in_idx = offs_seq[:, None] * BATCH * DIM + b * DIM + offs_dim[None, :]
        vals = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        out_idx = b * DIM * SEQ_LEN + offs_dim[None, :] * SEQ_LEN + offs_seq[:, None]
        tl.store(out_ptr + out_idx, vals, mask=mask)


def kernel_function(
    p_attn_in_proj_bias,
    p_attn_in_proj_weight,
    p_attn_out_proj_weight,
    p_attn_out_proj_bias,
    p_norm_weight,
    p_norm_bias,
    x,
):
    B, C, H, W = x.shape  # (2, 128, 128, 128)
    SEQ_LEN = H * W       # 16384
    BATCH = B             # 2
    DIM = C               # 128
    embed_dim = DIM       # 128
    num_heads = 4
    head_dim = embed_dim // num_heads  # 32
    scale = head_dim ** -0.5  # 0.1767...

    device = x.device
    dtype = x.dtype

    # Step 1: permute x from (B,C,H,W) -> (SEQ_LEN, B, DIM)
    x_flat = x.reshape(B, C, SEQ_LEN)  # (2, 128, 16384)
    x_seq = torch.empty(SEQ_LEN, BATCH, DIM, device=device, dtype=dtype)
    grid_p = (triton.cdiv(SEQ_LEN, 32), triton.cdiv(DIM, 32))
    permute_kernel[grid_p](
        x_flat, x_seq,
        SEQ_LEN, BATCH, DIM,
    )

    # Step 2: in_proj linear: (SEQ_LEN*BATCH, DIM) @ (DIM, 3*DIM) + bias
    x_2d = x_seq.reshape(SEQ_LEN * BATCH, DIM)  # (32768, 128)
    qkv_out = torch.empty(SEQ_LEN * BATCH, 3 * DIM, device=device, dtype=dtype)
    M_proj = SEQ_LEN * BATCH  # 32768
    N_proj = 3 * DIM           # 384
    K_proj = DIM               # 128
    # weight is (384, 128), we need (32768,128) @ (128,384)
    W_in = p_attn_in_proj_weight  # (384, 128)
    grid_g = (triton.cdiv(M_proj, 64), triton.cdiv(N_proj, 128))
    gemm_bias_kernel[grid_g](
        x_2d, W_in, p_attn_in_proj_bias, qkv_out,
        M_proj, N_proj, K_proj,
        x_2d.stride(0), x_2d.stride(1),
        W_in.stride(1), W_in.stride(0),  # B is transposed: stride_bk=stride(1), stride_bn=stride(0)
        qkv_out.stride(0), qkv_out.stride(1),
        HAS_BIAS=True,
    )

    # Step 3: reshape QKV and split
    # qkv_out: (32768, 384) -> (16384, 2, 3, 128)
    qkv = qkv_out.reshape(SEQ_LEN, BATCH, 3, DIM)
    # Q, K, V each: (SEQ_LEN, BATCH, DIM) -> reshape to (SEQ_LEN, num_heads, head_dim) -> transpose to (num_heads*BATCH, SEQ_LEN, head_dim)
    # Actually: (SEQ_LEN, BATCH, DIM) -> (SEQ_LEN, BATCH*num_heads, head_dim) -> (BATCH*num_heads, SEQ_LEN, head_dim)
    Q = qkv[:, :, 0, :].reshape(SEQ_LEN, BATCH * num_heads, head_dim).permute(1, 0, 2).contiguous()
    K = qkv[:, :, 1, :].reshape(SEQ_LEN, BATCH * num_heads, head_dim).permute(1, 0, 2).contiguous()
    V = qkv[:, :, 2, :].reshape(SEQ_LEN, BATCH * num_heads, head_dim).permute(1, 0, 2).contiguous()
    # Q, K, V: (8, 16384, 32)

    # Scale Q
    Q_scaled = Q * scale  # (8, 16384, 32)

    # Step 4: Attention scores: (8, 16384, 32) @ (8, 32, 16384) -> (8, 16384, 16384)
    num_batches = BATCH * num_heads  # 8
    attn_scores = torch.empty(num_batches, SEQ_LEN, SEQ_LEN, device=device, dtype=dtype)
    # BMM: Q_scaled (8,16384,32) x K^T (8,32,16384)
    K_t = K.permute(0, 2, 1).contiguous()  # (8, 32, 16384)
    grid_bmm1 = (triton.cdiv(SEQ_LEN, 32), triton.cdiv(SEQ_LEN, 32), num_batches)
    bmm_kernel[grid_bmm1](
        Q_scaled, K_t, attn_scores,
        num_batches, SEQ_LEN, SEQ_LEN, head_dim,
        Q_scaled.stride(0), Q_scaled.stride(1), Q_scaled.stride(2),
        K_t.stride(0), K_t.stride(1), K_t.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
    )

    # Step 5: Softmax over last dim
    attn_weights = torch.empty_like(attn_scores)
    total_rows = num_batches * SEQ_LEN
    softmax_kernel[(total_rows,)](
        attn_scores, attn_weights,
        total_rows, SEQ_LEN,
    )

    # Step 6: Weighted sum: (8, 16384, 16384) x (8, 16384, 32) -> (8, 16384, 32)
    attn_out = torch.empty(num_batches, SEQ_LEN, head_dim, device=device, dtype=dtype)
    grid_bmm2 = (triton.cdiv(SEQ_LEN, 32), triton.cdiv(head_dim, 32), num_batches)
    bmm_kernel[grid_bmm2](
        attn_weights, V, attn_out,
        num_batches, SEQ_LEN, head_dim, SEQ_LEN,
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
    )

    # Step 7: Reshape attn_out: (8, 16384, 32) -> (16384, 2, 128) -> (32768, 128)
    attn_out_2d = attn_out.permute(1, 0, 2).contiguous().reshape(SEQ_LEN * BATCH, DIM)

    # Step 8: out_proj linear: (32768, 128) @ (128, 128) + bias
    out_proj_out = torch.empty(SEQ_LEN * BATCH, DIM, device=device, dtype=dtype)
    W_out = p_attn_out_proj_weight  # (128, 128)
    grid_g2 = (triton.cdiv(SEQ_LEN * BATCH, 64), triton.cdiv(DIM, 128))
    gemm_bias_kernel[grid_g2](
        attn_out_2d, W_out, p_attn_out_proj_bias, out_proj_out,
        SEQ_LEN * BATCH, DIM, DIM,
        attn_out_2d.stride(0), attn_out_2d.stride(1),
        W_out.stride(1), W_out.stride(0),
        out_proj_out.stride(0), out_proj_out.stride(1),
        HAS_BIAS=True,
    )

    # Step 9: Compute mean of attention weights for the "add" step
    # attn_weights: (8, 16384, 16384) -> view as (2, 4, 16384, 16384) -> mean over dim=1 -> (2, 16384, 16384)
    # But this is used as: x = attn_output + x (residual)
    # Wait, looking at the computation graph again:
    # add.Tensor -> shape=[16384, 2, 128]: this is attn_output + x_seq (residual connection)
    # The mean.dim is for the attention weights (avg_weights output), but it's not used in the main path
    # Actually looking at the graph: view.default -> shape=[2, 4, 16384, 16384] then mean.dim -> shape=[2, 16384, 16384]
    # This is the avg_attn_weights returned by MultiheadAttention, but in the forward it's discarded (_)
    # The add.Tensor is: attn_output (16384,2,128) + x_seq (16384,2,128)

    # Step 10: Residual add + LayerNorm
    out_proj_seq = out_proj_out.reshape(SEQ_LEN, BATCH, DIM)
    # residual: x_seq + out_proj_seq
    residual = x_seq + out_proj_seq  # (16384, 2, 128)

    # LayerNorm over last dim (DIM=128)
    ln_out = torch.empty_like(residual)
    residual_2d = residual.reshape(SEQ_LEN * BATCH, DIM)
    ln_out_2d = ln_out.reshape(SEQ_LEN * BATCH, DIM)
    layernorm_kernel[(SEQ_LEN * BATCH,)](
        residual_2d, p_norm_weight, p_norm_bias, ln_out_2d,
        SEQ_LEN * BATCH, DIM,
        1e-5,
    )

    # Step 11: permute back (SEQ_LEN, BATCH, DIM) -> (BATCH, DIM, SEQ_LEN) -> reshape to (B,C,H,W)
    ln_out_seq = ln_out_2d.reshape(SEQ_LEN, BATCH, DIM)
    result = torch.empty(BATCH, DIM, SEQ_LEN, device=device, dtype=dtype)
    grid_pb = (triton.cdiv(SEQ_LEN, 32), triton.cdiv(DIM, 32))
    permute_back_kernel[grid_pb](
        ln_out_seq, result,
        SEQ_LEN, BATCH, DIM,
    )
    return result.reshape(B, C, H, W)


def get_inputs():
    B, C, H, W = 2, 128, 128, 128
    embed_dim = 128
    num_heads = 4
    return [
        torch.randn(3 * embed_dim, device='cuda'),           # p_attn_in_proj_bias
        torch.randn(3 * embed_dim, embed_dim, device='cuda'), # p_attn_in_proj_weight
        torch.randn(embed_dim, embed_dim, device='cuda'),     # p_attn_out_proj_weight
        torch.randn(embed_dim, device='cuda'),                # p_attn_out_proj_bias
        torch.randn(embed_dim, device='cuda'),                # p_norm_weight
        torch.randn(embed_dim, device='cuda'),                # p_norm_bias
        torch.randn(B, C, H, W, device='cuda'),              # x
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
        import torch.nn as nn

        p_attn_in_proj_bias, p_attn_in_proj_weight, p_attn_out_proj_weight, \
            p_attn_out_proj_bias, p_norm_weight, p_norm_bias, x = inputs

        embed_dim = 128
        num_heads = 4

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim, num_heads)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                B, C, H, W = x.shape
                x_in = x.view(B, C, H * W).permute(2, 0, 1)
                attn_output, _ = self.attn(x_in, x_in, x_in)
                x_out = self.norm(attn_output + x_in)
                x_out = x_out.permute(1, 2, 0).view(B, C, H, W)
                return x_out

        model = Model().cuda().eval()
        # Set weights
        with torch.no_grad():
            model.attn.in_proj_weight.copy_(p_attn_in_proj_weight)
            model.attn.in_proj_bias.copy_(p_attn_in_proj_bias)
            model.attn.out_proj.weight.copy_(p_attn_out_proj_weight)
            model.attn.out_proj.bias.copy_(p_attn_out_proj_bias)
            model.norm.weight.copy_(p_norm_weight)
            model.norm.bias.copy_(p_norm_bias)

        with torch.no_grad():
            ref = model(x)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")