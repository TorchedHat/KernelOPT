import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_N': 256}, num_warps=16, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles BLOCK_SIZE rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < M

    # For each row, compute mean and variance
    # We process N elements per row
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # First pass: compute mean
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        # Load x[row, col]
        ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=1)

    mean = mean / N

    # Second pass: compute variance
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean[:, None]
        var += tl.sum(diff * diff, axis=1)

    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Third pass: normalize and apply weight/bias
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

        x_norm = (x - mean[:, None]) * rstd[:, None]
        out = x_norm * w[None, :] + b[None, :]

        out_ptrs = out_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        tl.store(out_ptrs, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_gelu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the exact formula via erf
    # gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    out = x * cdf

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def window_partition_kernel(
    x_ptr,
    out_ptr,
    B,
    H,
    W,
    C,
    window_size,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # x: (B, H, W, C) -> windows: (num_windows*B, window_size, window_size, C)
    # num_windows = (H // window_size) * (W // window_size)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Output index: (batch_win, wh, ww, c)
    # batch_win = b * num_win_h * num_win_w + win_h * num_win_w + win_w
    # wh, ww in [0, window_size)
    # c in [0, C)
    c = offsets % C
    tmp = offsets // C
    ww = tmp % window_size
    tmp = tmp // window_size
    wh = tmp % window_size
    tmp = tmp // window_size
    win_w = tmp % (W // window_size)
    tmp = tmp // (W // window_size)
    win_h = tmp % (H // window_size)
    b = tmp // (H // window_size)

    # Input index: x[b, win_h * window_size + wh, win_w * window_size + ww, c]
    ih = win_h * window_size + wh
    iw = win_w * window_size + ww
    in_idx = b * H * W * C + ih * W * C + iw * C + c

    x = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def layer_norm_triton(x, weight, bias, eps=1e-5):
    """Apply layer norm using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)

    BLOCK_SIZE = 1
    BLOCK_N = min(triton.next_power_of_2(N), 512)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    fused_layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        M, N, eps,
    )
    return out.reshape(orig_shape)


def gelu_triton(x):
    """Apply GELU using Triton kernel."""
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_gelu_kernel[grid](x, out, N)
    return out


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
    p_layers_1_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_1_attn_proj_weight,
    p_layers_1_blocks_1_attn_proj_bias,
    p_layers_1_blocks_1_norm1_weight,
    p_layers_1_blocks_1_norm1_bias,
    p_layers_1_blocks_1_mlp_fc1_weight,
    p_layers_1_blocks_1_mlp_fc1_bias,
    p_layers_1_blocks_1_mlp_fc2_weight,
    p_layers_1_blocks_1_mlp_fc2_bias,
    p_layers_1_blocks_1_norm2_weight,
    p_layers_1_blocks_1_norm2_bias,
    p_layers_1_downsample_reduction_weight,
    p_layers_1_downsample_norm_weight,
    p_layers_1_downsample_norm_bias,
    p_layers_2_blocks_0_attn_q_bias,
    p_layers_2_blocks_0_attn_v_bias,
    p_layers_2_blocks_0_attn_qkv_weight,
    p_layers_2_blocks_0_attn_logit_scale,
    p_layers_2_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_0_attn_proj_weight,
    p_layers_2_blocks_0_attn_proj_bias,
    p_layers_2_blocks_0_norm1_weight,
    p_layers_2_blocks_0_norm1_bias,
    p_layers_2_blocks_0_mlp_fc1_weight,
    p_layers_2_blocks_0_mlp_fc1_bias,
    p_layers_2_blocks_0_mlp_fc2_weight,
    p_layers_2_blocks_0_mlp_fc2_bias,
    p_layers_2_blocks_0_norm2_weight,
    p_layers_2_blocks_0_norm2_bias,
    p_layers_2_blocks_1_attn_q_bias,
    p_layers_2_blocks_1_attn_v_bias,
    p_layers_2_blocks_1_attn_qkv_weight,
    p_layers_2_blocks_1_attn_logit_scale,
    p_layers_2_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_1_attn_proj_weight,
    p_layers_2_blocks_1_attn_proj_bias,
    p_layers_2_blocks_1_norm1_weight,
    p_layers_2_blocks_1_norm1_bias,
    p_layers_2_blocks_1_mlp_fc1_weight,
    p_layers_2_blocks_1_mlp_fc1_bias,
    p_layers_2_blocks_1_mlp_fc2_weight,
    p_layers_2_blocks_1_mlp_fc2_bias,
    p_layers_2_blocks_1_norm2_weight,
    p_layers_2_blocks_1_norm2_bias,
    p_layers_2_blocks_2_attn_q_bias,
    p_layers_2_blocks_2_attn_v_bias,
    p_layers_2_blocks_2_attn_qkv_weight,
    p_layers_2_blocks_2_attn_logit_scale,
    p_layers_2_blocks_2_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_2_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_2_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_2_attn_proj_weight,
    p_layers_2_blocks_2_attn_proj_bias,
    p_layers_2_blocks_2_norm1_weight,
    p_layers_2_blocks_2_norm1_bias,
    p_layers_2_blocks_2_mlp_fc1_weight,
    p_layers_2_blocks_2_mlp_fc1_bias,
    p_layers_2_blocks_2_mlp_fc2_weight,
    p_layers_2_blocks_2_mlp_fc2_bias,
    p_layers_2_blocks_2_norm2_weight,
    p_layers_2_blocks_2_norm2_bias,
    p_layers_2_blocks_3_attn_q_bias,
    p_layers_2_blocks_3_attn_v_bias,
    p_layers_2_blocks_3_attn_qkv_weight,
    p_layers_2_blocks_3_attn_logit_scale,
    p_layers_2_blocks_3_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_3_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_3_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_3_attn_proj_weight,
    p_layers_2_blocks_3_attn_proj_bias,
    p_layers_2_blocks_3_norm1_weight,
    p_layers_2_blocks_3_norm1_bias,
    p_layers_2_blocks_3_mlp_fc1_weight,
    p_layers_2_blocks_3_mlp_fc1_bias,
    p_layers_2_blocks_3_mlp_fc2_weight,
    p_layers_2_blocks_3_mlp_fc2_bias,
    p_layers_2_blocks_3_norm2_weight,
    p_layers_2_blocks_3_norm2_bias,
    p_layers_2_blocks_4_attn_q_bias,
    p_layers_2_blocks_4_attn_v_bias,
    p_layers_2_blocks_4_attn_qkv_weight,
    p_layers_2_blocks_4_attn_logit_scale,
    p_layers_2_blocks_4_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_4_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_4_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_4_attn_proj_weight,
    p_layers_2_blocks_4_attn_proj_bias,
    p_layers_2_blocks_4_norm1_weight,
    p_layers_2_blocks_4_norm1_bias,
    p_layers_2_blocks_4_mlp_fc1_weight,
    p_layers_2_blocks_4_mlp_fc1_bias,
    p_layers_2_blocks_4_mlp_fc2_weight,
    p_layers_2_blocks_4_mlp_fc2_bias,
    p_layers_2_blocks_4_norm2_weight,
    p_layers_2_blocks_4_norm2_bias,
    p_layers_2_blocks_5_attn_q_bias,
    p_layers_2_blocks_5_attn_v_bias,
    p_layers_2_blocks_5_attn_qkv_weight,
    p_layers_2_blocks_5_attn_logit_scale,
    p_layers_2_blocks_5_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_5_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_5_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_5_attn_proj_weight,
    p_layers_2_blocks_5_attn_proj_bias,
    p_layers_2_blocks_5_norm1_weight,
    p_layers_2_blocks_5_norm1_bias,
    p_layers_2_blocks_5_mlp_fc1_weight,
    p_layers_2_blocks_5_mlp_fc1_bias,
    p_layers_2_blocks_5_mlp_fc2_weight,
    p_layers_2_blocks_5_mlp_fc2_bias,
    p_layers_2_blocks_5_norm2_weight,
    p_layers_2_blocks_5_norm2_bias,
    p_layers_2_downsample_reduction_weight,
    p_layers_2_downsample_norm_weight,
    p_layers_2_downsample_norm_bias,
    p_layers_3_blocks_0_attn_q_bias,
    p_layers_3_blocks_0_attn_v_bias,
    p_layers_3_blocks_0_attn_qkv_weight,
    p_layers_3_blocks_0_attn_logit_scale,
    p_layers_3_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_3_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_0_attn_proj_weight,
    p_layers_3_blocks_0_attn_proj_bias,
    p_layers_3_blocks_0_norm1_weight,
    p_layers_3_blocks_0_norm1_bias,
    p_layers_3_blocks_0_mlp_fc1_weight,
    p_layers_3_blocks_0_mlp_fc1_bias,
    p_layers_3_blocks_0_mlp_fc2_weight,
    p_layers_3_blocks_0_mlp_fc2_bias,
    p_layers_3_blocks_0_norm2_weight,
    p_layers_3_blocks_0_norm2_bias,
    p_layers_3_blocks_1_attn_q_bias,
    p_layers_3_blocks_1_attn_v_bias,
    p_layers_3_blocks_1_attn_qkv_weight,
    p_layers_3_blocks_1_attn_logit_scale,
    p_layers_3_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_3_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_1_attn_proj_weight,
    p_layers_3_blocks_1_attn_proj_bias,
    p_layers_3_blocks_1_norm1_weight,
    p_layers_3_blocks_1_norm1_bias,
    p_layers_3_blocks_1_mlp_fc1_weight,
    p_layers_3_blocks_1_mlp_fc1_bias,
    p_layers_3_blocks_1_mlp_fc2_weight,
    p_layers_3_blocks_1_mlp_fc2_bias,
    p_layers_3_blocks_1_norm2_weight,
    p_layers_3_blocks_1_norm2_bias,
    p_norm_weight,
    p_norm_bias,
    p_head_weight,
    p_head_bias,
    b_layers_0_blocks_0_attn_relative_coords_table,
    b_layers_0_blocks_0_attn_relative_position_index,
    b_layers_0_blocks_1_attn_mask,
    b_layers_0_blocks_1_attn_relative_coords_table,
    b_layers_0_blocks_1_attn_relative_position_index,
    b_layers_1_blocks_0_attn_relative_coords_table,
    b_layers_1_blocks_0_attn_relative_position_index,
    b_layers_1_blocks_1_attn_mask,
    b_layers_1_blocks_1_attn_relative_coords_table,
    b_layers_1_blocks_1_attn_relative_position_index,
    b_layers_2_blocks_0_attn_relative_coords_table,
    b_layers_2_blocks_0_attn_relative_position_index,
    b_layers_2_blocks_1_attn_mask,
    b_layers_2_blocks_1_attn_relative_coords_table,
    b_layers_2_blocks_1_attn_relative_position_index,
    b_layers_2_blocks_2_attn_relative_coords_table,
    b_layers_2_blocks_2_attn_relative_position_index,
    b_layers_2_blocks_3_attn_mask,
    b_layers_2_blocks_3_attn_relative_coords_table,
    b_layers_2_blocks_3_attn_relative_position_index,
    b_layers_2_blocks_4_attn_relative_coords_table,
    b_layers_2_blocks_4_attn_relative_position_index,
    b_layers_2_blocks_5_attn_mask,
    b_layers_2_blocks_5_attn_relative_coords_table,
    b_layers_2_blocks_5_attn_relative_position_index,
    b_layers_3_blocks_0_attn_relative_coords_table,
    b_layers_3_blocks_0_attn_relative_position_index,
    b_layers_3_blocks_1_attn_relative_coords_table,
    b_layers_3_blocks_1_attn_relative_position_index,
    c_layers_0_blocks_0_attn_lifted_tensor_0,
    c_layers_0_blocks_1_attn_lifted_tensor_1,
    c_layers_1_blocks_0_attn_lifted_tensor_2,
    c_layers_1_blocks_1_attn_lifted_tensor_3,
    c_layers_2_blocks_0_attn_lifted_tensor_4,
    c_layers_2_blocks_1_attn_lifted_tensor_5,
    c_layers_2_blocks_2_attn_lifted_tensor_6,
    c_layers_2_blocks_3_attn_lifted_tensor_7,
    c_layers_2_blocks_4_attn_lifted_tensor_8,
    c_layers_2_blocks_5_attn_lifted_tensor_9,