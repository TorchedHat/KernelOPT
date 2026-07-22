import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    
    rstd = libdevice.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](x_2d, weight, bias, y, M, N, eps)
    return y.reshape(orig_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    N = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Fused Add + LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_layer_norm_kernel(
    X_ptr, Residual_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    X_row = X_ptr + row * N
    R_row = Residual_ptr + row * N
    Y_row = Y_ptr + row * N
    
    # Compute mean of (x + residual)
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(tl.where(mask, x + r, 0.0), axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + r - mean, 0.0)
        var += tl.sum(val * val, axis=0)
    var = var / N
    
    rstd = libdevice.rsqrt(var + eps)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = ((x + r) - mean) * rstd * w + b
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_add_layer_norm(x, residual, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    r_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    add_layer_norm_kernel[grid](x_2d, r_2d, weight, bias, y, M, N, eps)
    return y.reshape(orig_shape)


# ============================================================
# Softmax kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.where(mask, libdevice.exp(x - row_max), 0.0), axis=0)
    
    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.where(mask, libdevice.exp(x - row_max) / row_sum, 0.0)
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Only support last dim softmax
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, y, M, N)
    return y.reshape(orig_shape)


# ============================================================
# Sigmoid kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    N = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, y, N)
    return y


# ============================================================
# ReLU inplace kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def relu_inplace_kernel(X_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(X_ptr + offs, y.to(tl.float32), mask=mask)


def triton_relu_inplace(x):
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_inplace_kernel[grid](x, N)
    return x


# ============================================================
# L2 normalize kernel (for cosine attention)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def l2_normalize_kernel(X_ptr, Y_ptr, M, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    
    # Compute L2 norm
    norm_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        norm_sq += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    
    norm = libdevice.sqrt(norm_sq)
    norm = tl.maximum(norm, eps)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = x / norm
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_l2_normalize(x, dim=-1, eps=1e-12):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    l2_normalize_kernel[grid](x_2d, y, M, N, eps)
    return y.reshape(orig_shape)


# ============================================================
# Main model implementation
# ============================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_bias(
    relative_coords_table, relative_position_index, cpb_mlp_0_weight, cpb_mlp_0_bias,
):
    # MLP on relative coords table
    x = relative_coords_table.view(-1, 2)  # (169, 2)
    x = torch.mm(x, cpb_mlp_0_weight.t()) + cpb_mlp_0_bias  # (169, 512)
    # ReLU
    x = triton_relu_inplace(x)
    x = torch.mm(x, cpb_mlp_2_weight.t())  # (169, num_heads)
    x = x.view(13, 13, num_heads)  # (13, 13, num_heads)
    
    # Index by relative_position_index
    x_flat = x.view(-1, num_heads)  # (169, num_heads)
    idx_flat = relative_position_index.view(-1)  # (2401,)
    relative_position_bias = x_flat[idx_flat]  # (2401, num_heads)
    relative_position_bias = relative_position_bias.view(49, 49, num_heads)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, 49, 49)
    
    # sigmoid * 16
    relative_position_bias = triton_sigmoid(relative_position_bias) * 16.0
    return relative_position_bias


def swin_window_attention(
    q_bias, v_bias, qkv_weight,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    proj_weight, proj_bias,
    relative_coords_table, relative_position_index,
    num_heads, attn_mask=None,
):
    B_, N, C = x.shape
    head_dim = C // num_heads
    
    # Build qkv bias
    k_bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias])
    
    # QKV projection
    qkv = torch.nn.functional.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # Wait, the reshape is (B_, N, 3, num_heads, head_dim) -> permute (2, 0, 3, 1, 4)
    # Actually from the graph: reshape to (B_, N, 3, num_heads, head_dim), permute (2, 0, num_heads_dim, N, head_dim)
    # From graph: reshape [640, 49, 3, 3, 32] -> permute [3, 640, 3, 49, 32]
    # So permute is (2, 0, 3, 1, 4) -> wait that gives [3, 640, 3, 49, 32] from [640, 49, 3, 3, 32]
    # Actually permute(2,0,3,1,4) on shape [640,49,3,3,32]:
    # dim0=640, dim1=49, dim2=3, dim3=3, dim4=32
    # permute(2,0,3,1,4) -> [3, 640, 3, 49, 32] YES
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # each (B_, num_heads, N, head_dim)
    
    # Cosine attention: normalize q and k
    q = triton_l2_normalize(q, dim=-1, eps=1e-6)
    k = triton_l2_normalize(k, dim=-1, eps=1e-6)
    
    # Logit scale
    logit_scale_clamped = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=x.device))).exp()
    
    # Attention: q @ k^T * scale
    attn = torch.matmul(q, k.transpose(-2, -1)) * logit_scale_clamped
    
    # Relative position bias
    rel_pos_bias = get_relative_position_bias(
        relative_coords_table, relative_position_index,
    )
    attn = attn + rel_pos_bias.unsqueeze(0)
    
    # Attention mask for shifted windows
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)
    
    attn = triton_softmax(attn, dim=-1)
    # dropout is identity at inference
    
    x = torch.matmul(attn, v)  # (B_, num_heads, N, head_dim)
    x = x.transpose(1, 2).reshape(B_, N, C)
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


def swin_block(
    x, H, W, window_size, shift_size,
    q_bias, v_bias, qkv_weight, logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    proj_weight, proj_bias,
    norm1_weight, norm1_bias,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    norm2_weight, norm2_bias,
    relative_coords_table, relative_position_index,
    num_heads, attn_mask=None,
):
    B, L, C = x.shape
    
    shortcut = x
    x_norm = triton_layer_norm(x, norm1_weight, norm1_bias)
    
    x_2d = x_norm.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_2d
    
    # Window partition
    x_windows = window_partition(shifted_x, window_size)  # (nW*B, ws, ws, C)
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # Attention
    attn_windows = swin_window_attention(
        x_windows, q_bias, v_bias, qkv_weight, logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        proj_weight, proj_bias,
        relative_coords_table, relative_position_index,
    )
    
    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)
    
    # Reverse cyclic shift
    if shift_size > 0:
        x_out = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x_out = shifted_x
    
    x_out = x_out.view(B, H * W, C)
    
    # Residual + norm2
    x = shortcut + x_out
    
    # MLP
    x_mlp = triton_layer_norm(x, norm2_weight, norm2_bias)
    x_mlp = torch.nn.functional.linear(x_mlp, mlp_fc1_weight, mlp_fc1_bias)
    x_mlp = triton_gelu(x_mlp)
    # dropout identity
    x_mlp = torch.nn.functional.linear(x_mlp, mlp_fc2_weight, mlp_fc2_bias)
    # dropout identity
    
    x = x + x_mlp
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
    x = x.view(B, -1, 4 * C)
    
    x = torch.nn.functional.linear(x, reduction_weight)
    x = triton_layer_norm(x, norm_weight, norm_bias)
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