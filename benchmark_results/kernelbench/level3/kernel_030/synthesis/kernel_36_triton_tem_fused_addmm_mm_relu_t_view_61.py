import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        mask_b = mask_k[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    c_ptrs = C_ptr + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'R'],
)
@triton.jit
def fused_layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, R,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    """
    row = tl.program_id(0)
    if row >= N:
        return

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / R

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / R
    rstd = libdevice.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + r_idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + r_idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_ptr + row * R + r_idx, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'R'],
)
@triton.jit
def fused_add_layernorm_kernel(
    X_ptr, Residual_ptr, W_ptr, B_ptr, Y_ptr,
    N, R,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Fused: Y = LayerNorm(X + Residual)
    """
    row = tl.program_id(0)
    if row >= N:
        return

    # Compute mean of (X + Residual)
    mean = tl.zeros((), dtype=tl.float32)
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Residual_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + res, 0.0)
        mean += tl.sum(val, axis=0)
    mean = mean / R

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Residual_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + res, 0.0)
        diff = tl.where(mask, val - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / R
    rstd = libdevice.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for r_off in range(0, R, BLOCK_R):
        r_idx = r_off + tl.arange(0, BLOCK_R)
        mask = r_idx < R
        x = tl.load(X_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Residual_ptr + row * R + r_idx, mask=mask, other=0.0).to(tl.float32)
        val = x + res
        w = tl.load(W_ptr + r_idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + r_idx, mask=mask, other=0.0).to(tl.float32)
        y = (val - mean) * rstd * w + b
        tl.store(Y_ptr + row * R + r_idx, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    tanh_val = libdevice.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise add: C = A + B
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


def triton_matmul(A, B):
    """A: [M, K], B: [K, N] -> C: [M, N]"""
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def triton_layernorm(X, weight, bias, eps=1e-5):
    """X: [..., R], weight: [R], bias: [R] -> Y: [..., R]"""
    orig_shape = X.shape
    R = orig_shape[-1]
    X_2d = X.reshape(-1, R).contiguous()
    N = X_2d.shape[0]
    Y_2d = torch.empty_like(X_2d)
    grid = (N,)
    fused_layernorm_kernel[grid](
        X_2d, weight, bias, Y_2d,
        N, R, eps,
    )
    return Y_2d.reshape(orig_shape)


def triton_add_layernorm(X, Residual, weight, bias, eps=1e-5):
    """Fused: Y = LayerNorm(X + Residual)"""
    orig_shape = X.shape
    R = orig_shape[-1]
    X_2d = X.reshape(-1, R).contiguous()
    Res_2d = Residual.reshape(-1, R).contiguous()
    N = X_2d.shape[0]
    Y_2d = torch.empty_like(X_2d)
    grid = (N,)
    fused_add_layernorm_kernel[grid](
        X_2d, Res_2d, weight, bias, Y_2d,
        N, R, eps,
    )
    return Y_2d.reshape(orig_shape)


def triton_gelu(X):
    """GELU activation"""
    Y = torch.empty_like(X)
    N = X.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](X, Y, N)
    return Y


def triton_linear(X, weight, bias=None):
    """Linear: Y = X @ weight.T + bias"""
    orig_shape = X.shape
    in_features = orig_shape[-1]
    out_features = weight.shape[0]
    X_2d = X.reshape(-1, in_features).contiguous()
    W_T = weight.t().contiguous()
    Y_2d = triton_matmul(X_2d, W_T)
    if bias is not None:
        N = Y_2d.numel()
        # Add bias using triton
        bias_expanded = bias.unsqueeze(0).expand(Y_2d.shape[0], -1).contiguous()
        Y_2d = Y_2d + bias_expanded
    new_shape = orig_shape[:-1] + (out_features,)
    return Y_2d.reshape(new_shape)


def window_partition_triton(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_triton(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def swin_attention(
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    num_heads,
    attn_mask=None,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads

    # Build qkv bias
    zeros = torch.zeros(C, device=x_windows.device, dtype=x_windows.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])

    # QKV projection
    qkv = triton_linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Cosine attention
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    # Logit scale
    logit_scale_clamped = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=logit_scale.device))).exp()

    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * logit_scale_clamped

    # Relative position bias via CPB MLP
    # cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    rel_coords_flat = relative_coords_table.view(-1, 2)  # [169, 2]
    cpb_hidden = triton_linear(rel_coords_flat, cpb_mlp_0_weight, cpb_mlp_0_bias)
    cpb_hidden = torch.relu(cpb_hidden)
    cpb_out = triton_linear(cpb_hidden, cpb_mlp_2_weight)  # [169, num_heads]

    # Index relative position bias
    rel_pos_idx_flat = relative_position_index.view(-1)  # [N*N]
    rel_pos_bias = cpb_out[rel_pos_idx_flat].view(N, N, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    rel_pos_bias = rel_pos_bias.unsqueeze(0)  # [1, num_heads, N, N]

    attn = attn + rel_pos_bias

    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = torch.softmax(attn, dim=-1)

    x = torch.matmul(attn, v)
    x = x.transpose(1, 2).reshape(B_, N, C)
    x = triton_linear(x, proj_weight, proj_bias)
    return x


def swin_mlp(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    x = triton_linear(x, fc1_weight, fc1_bias)
    x = triton_gelu(x)
    x = triton_linear(x, fc2_weight, fc2_bias)
    return x


def swin_block(
    norm1_weight, norm1_bias,
    norm2_weight, norm2_bias,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    fc1_weight, fc1_bias,
    fc2_weight, fc2_bias,
    num_heads,
    window_size,
    shift_size,
    H, W,
    attn_mask=None,
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x_norm = triton_layernorm(x, norm1_weight, norm1_bias)
    x_norm = x_norm.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_norm, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_norm

    # Window partition
    x_windows = window_partition_triton(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Attention
    attn_windows = swin_attention(
        x_windows, qkv_weight, q_bias, v_bias,
        proj_weight, proj_bias,
        logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias,
        cpb_mlp_2_weight,
        relative_coords_table,
        relative_position_index,
        num_heads,
        attn_mask=attn_mask,
    )

    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse_triton(attn_windows, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x_out = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x_out = shifted_x

    x_out = x_out.view(B, H * W, C)

    # Residual + Norm2 + MLP
    x = shortcut + x_out
    x_norm2 = triton_layernorm(x, norm2_weight, norm2_bias)
    mlp_out = swin_mlp(x_norm2, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    x = x + mlp_out

    return x


def patch_merging(x, reduction_weight, norm_weight, norm_bias, H, W):
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)
    x = x.view(B, -1, 4 * C)

    x = triton_linear(x, reduction_weight)
    x = triton_layernorm(x, norm_weight, norm_bias)
    return x


def kernel_function(
    p_patch_embed_proj_weight,
    p_patch_embed_proj_bias,
    p_patch_embed_norm_weight,
    p_patch_embed_norm_bias,
    p_layers_0_blocks_0_attn_q_bias,
    p_layers_0_blocks_0_attn_v_bias,
    p_layers_0_blocks_0_attn_qkv_weight,
    p_layers_0_blocks_0_attn_logit_scale,
    p_layers_0_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_0_attn_proj_weight,
    p_layers_0_blocks_0_attn_proj_bias,
    p_layers_0_blocks_0_norm1_weight,
    p_layers_0_blocks_0_norm1_bias,
    p_layers_0_blocks_0_mlp_fc1_weight,
    p_layers_0_blocks_0_mlp_fc1_bias,
    p_layers_0_blocks_0_mlp_fc2_weight,
    p_layers_0_blocks_0_mlp_fc2_bias,
    p_layers_0_blocks_0_norm2_weight,
    p_layers_0_blocks_0_norm2_bias,
    p_layers_0_blocks_1_attn_q_bias,
    p_layers_0_blocks_1_attn_v_bias,
    p_layers_0_blocks_1_attn_qkv_weight,
    p_layers_0_blocks_1_attn_logit_scale,
    p_layers_0_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_1_attn_proj_weight,
    p_layers_0_blocks_1_attn_proj_bias,
    p_layers_0_blocks_1_norm1_weight,
    p_layers_0_blocks_1_norm1_bias,
    p_layers_0_blocks_1_mlp_fc1_weight,
    p_layers_0_blocks_1_mlp_fc1_bias,
    p_layers_0_blocks_1_mlp_fc2_weight,
    p_layers_0_blocks_1_mlp_fc2_bias,
    p_layers_0_blocks_1_norm2_weight,
    p_layers_0_blocks_1_norm2_bias,
    p_layers_0_downsample_reduction_weight,
    p_layers_0_downsample_norm_weight,
    p_layers_0_downsample_norm_bias,
    p_layers_1_blocks_0_attn_q_bias,
    p_layers_1_blocks_0_attn_v_bias,
    p_layers_1_blocks_0_attn_qkv_weight,
    p_layers_1_blocks_0_attn_logit_scale,
    p_layers_1_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_0_attn_proj_weight,
    p_layers_1_blocks_0_attn_proj_bias,
    p_layers_1_blocks_0_norm1_weight,
    p_layers_1_blocks_0_norm1_bias,
    p_layers_1_blocks_0_mlp_fc1_weight,
    p_layers_1_blocks_0_mlp_fc1_bias,
    p_layers_1_blocks_0_mlp_fc2_weight,
    p_layers_1_blocks_0_mlp_fc2_bias,
    p_layers_1_blocks_0_norm2_weight,
    p_layers_1_blocks_0_norm2_bias,
    p_layers_1_blocks_1_attn_q_bias,
    p_layers_1_blocks_1_attn_v_bias,
    p_layers_1_blocks_1_attn_qkv_weight,
    p_layers_1_blocks_1_attn_logit_scale,