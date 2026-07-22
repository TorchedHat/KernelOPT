import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def lstm_gates_kernel(
    # x @ W_ih^T: input contribution (already computed as xW)
    # h @ W_hh^T: hidden contribution
    # biases
    # cell state in
    # outputs
    # dims
    M, H,
    # strides
    stride_xw_m, stride_xw_n,
    stride_hw_m, stride_hw_n,
    stride_c_m, stride_c_h,
    stride_hout_m, stride_hout_h,
    stride_cout_m, stride_cout_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of [BLOCK_M rows, BLOCK_N gates]
    # We process gates in chunks of BLOCK_N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_n < (4 * H)
    
    # Load xW values
    xw_ptrs = xW_ptr + offs_m[:, None] * stride_xw_m + offs_n[None, :] * stride_xw_n
    hw_ptrs = hW_ptr + offs_m[:, None] * stride_hw_m + offs_n[None, :] * stride_hw_n
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < 4 * H)
    
    xw = tl.load(xw_ptrs, mask=mask, other=0.0).to(tl.float32)
    hw = tl.load(hw_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    bih = tl.load(bias_ih_ptr + offs_n, mask=(offs_n < 4 * H), other=0.0).to(tl.float32)
    bhh = tl.load(bias_hh_ptr + offs_n, mask=(offs_n < 4 * H), other=0.0).to(tl.float32)
    
    gates = xw + hw + bih[None, :] + bhh[None, :]
    
    # gates layout: [i, f, g, o] each of size H
    # offs_n tells us which gate column we're in
    gate_idx = offs_n % H  # position within gate
    gate_type = offs_n // H  # 0=i, 1=f, 2=g, 3=o
    
    # Apply activations
    # i = sigmoid(gates[:, 0:H])
    # f = sigmoid(gates[:, H:2H])
    # g = tanh(gates[:, 2H:3H])
    # o = sigmoid(gates[:, 3H:4H])
    
    # We need to load c for this tile
    # gate_idx gives the H-dimension index
    c_col_offs = gate_idx  # [BLOCK_N]
    c_ptrs = c_ptr + offs_m[:, None] * stride_c_m + c_col_offs[None, :] * stride_c_h
    c_mask = (offs_m[:, None] < M) & (c_col_offs[None, :] < H)
    c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    
    # Compute sigmoid and tanh
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig_gates = 1.0 / (1.0 + tl.exp(-gates))
    tanh_gates = tl.extra.libdevice.tanh(gates)
    
    # Select activation based on gate type
    # gate_type: 0,1,3 -> sigmoid; 2 -> tanh
    is_tanh = (gate_type == 2)
    activated = tl.where(is_tanh[None, :], tanh_gates, sig_gates)
    
    # Now compute c_new and h_new
    # c_new = f * c + i * g
    # h_new = o * tanh(c_new)
    # But we need all 4 gates simultaneously for each row...
    # This tile approach doesn't work well for the gate combination step
    # We'll store intermediate activated gates and do a second pass
    # Actually, let's store the activated gates to a temp buffer
    # and do the combination in a separate kernel
    
    # Store activated gates
    tl.store(xw_ptrs, activated.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256}, num_warps=8),
    ],
    key=['M', 'H'],
)
@triton.jit
def lstm_combine_kernel(
    M, H,
    stride_g_m, stride_g_h,
    stride_c_m, stride_c_h,
    stride_ho_m, stride_ho_h,
    stride_co_m, stride_co_h,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    # Process H elements at a time (loop over H)
    for h_off in range(0, H, 1):
        h_idx = h_off
        
        # Load gates for this h position
        i_gate = tl.load(gates_ptr + offs_m * stride_g_m + (0 * H + h_idx) * stride_g_h, 
                         mask=mask_m, other=0.0).to(tl.float32)
        f_gate = tl.load(gates_ptr + offs_m * stride_g_m + (1 * H + h_idx) * stride_g_h,
                         mask=mask_m, other=0.0).to(tl.float32)
        g_gate = tl.load(gates_ptr + offs_m * stride_g_m + (2 * H + h_idx) * stride_g_h,
                         mask=mask_m, other=0.0).to(tl.float32)
        o_gate = tl.load(gates_ptr + offs_m * stride_g_m + (3 * H + h_idx) * stride_g_h,
                         mask=mask_m, other=0.0).to(tl.float32)
        
        c_val = tl.load(c_ptr + offs_m * stride_c_m + h_idx * stride_c_h,
                        mask=mask_m, other=0.0).to(tl.float32)
        
        c_new = f_gate * c_val + i_gate * g_gate
        h_new = o_gate * tl.extra.libdevice.tanh(c_new)
        
        tl.store(c_out_ptr + offs_m * stride_co_m + h_idx * stride_co_h,
                 c_new.to(tl.float32), mask=mask_m)
        tl.store(h_out_ptr + offs_m * stride_ho_m + h_idx * stride_ho_h,
                 h_new.to(tl.float32), mask=mask_m)


# This approach with a loop over H=256 is too slow. Let me redesign.
# Better approach: use a 2D kernel where each program handles [BLOCK_M, BLOCK_H]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_H': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 256}, num_warps=8),
    ],
    key=['M', 'H'],
)
@triton.jit
def lstm_cell_combine_kernel(
    M, H,
    stride_g_m,
    stride_c_m,
    stride_ho_m,
    stride_co_m,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    
    # Load i, f, g, o gates
    i_ptrs = gates_ptr + offs_m[:, None] * stride_g_m + (0 * H + offs_h[None, :])
    f_ptrs = gates_ptr + offs_m[:, None] * stride_g_m + (1 * H + offs_h[None, :])
    g_ptrs = gates_ptr + offs_m[:, None] * stride_g_m + (2 * H + offs_h[None, :])
    o_ptrs = gates_ptr + offs_m[:, None] * stride_g_m + (3 * H + offs_h[None, :])
    
    i_gate = tl.load(i_ptrs, mask=mask, other=0.0).to(tl.float32)
    f_gate = tl.load(f_ptrs, mask=mask, other=0.0).to(tl.float32)
    g_gate = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    o_gate = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_c_m + offs_h[None, :]
    c_val = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    c_new = f_gate * c_val + i_gate * g_gate
    h_new = o_gate * tl.extra.libdevice.tanh(c_new)
    
    ho_ptrs = h_out_ptr + offs_m[:, None] * stride_ho_m + offs_h[None, :]
    co_ptrs = c_out_ptr + offs_m[:, None] * stride_co_m + offs_h[None, :]
    
    tl.store(ho_ptrs, h_new.to(tl.float32), mask=mask)
    tl.store(co_ptrs, c_new.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        
        acc = tl.dot(a, b, acc)
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        
        acc = tl.dot(x, w, acc)
    
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_H': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 256}, num_warps=8),
    ],
    key=['M', 'H'],
)
@triton.jit
def lstm_gates_activate_kernel(
    M, H,
    stride_gm,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Apply sigmoid to i,f,o gates and tanh to g gate in-place."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    
    # Process i gate (offset 0)
    i_ptrs = gates_ptr + offs_m[:, None] * stride_gm + offs_h[None, :]
    i_vals = tl.load(i_ptrs, mask=mask, other=0.0).to(tl.float32)
    i_act = 1.0 / (1.0 + tl.exp(-i_vals))
    tl.store(i_ptrs, i_act, mask=mask)
    
    # Process f gate (offset H)
    f_ptrs = gates_ptr + offs_m[:, None] * stride_gm + (H + offs_h[None, :])
    f_vals = tl.load(f_ptrs, mask=mask, other=0.0).to(tl.float32)
    f_act = 1.0 / (1.0 + tl.exp(-f_vals))
    tl.store(f_ptrs, f_act, mask=mask)
    
    # Process g gate (offset 2H) - tanh
    g_ptrs = gates_ptr + offs_m[:, None] * stride_gm + (2 * H + offs_h[None, :])
    g_vals = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    g_act = tl.extra.libdevice.tanh(g_vals)
    tl.store(g_ptrs, g_act, mask=mask)
    
    # Process o gate (offset 3H)
    o_ptrs = gates_ptr + offs_m[:, None] * stride_gm + (3 * H + offs_h[None, :])
    o_vals = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    o_act = 1.0 / (1.0 + tl.exp(-o_vals))
    tl.store(o_ptrs, o_act, mask=mask)


def lstm_cell_forward(x_t, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    """
    """
    batch = x_t.shape[0]
    H = h_prev.shape[1]
    input_size = x_t.shape[1]
    four_H = 4 * H
    
    # Compute xW: [batch, 4H] = x @ W_ih^T
    xW = torch.empty(batch, four_H, device=x_t.device, dtype=torch.float32)
    grid_xw = (triton.cdiv(batch, 16), triton.cdiv(four_H, 64))
    matmul_kernel[grid_xw](
        x_t, weight_ih, xW,
        batch, four_H, input_size,
        x_t.stride(0), x_t.stride(1),
        weight_ih.stride(1), weight_ih.stride(0),  # weight_ih is [4H, input_size], we want x @ W^T
        xW.stride(0), xW.stride(1),
    )
    
    # Compute hW: [batch, 4H] = h @ W_hh^T
    hW = torch.empty(batch, four_H, device=x_t.device, dtype=torch.float32)
    grid_hw = (triton.cdiv(batch, 16), triton.cdiv(four_H, 64))
    matmul_kernel[grid_hw](
        h_prev, weight_hh, hW,
        batch, four_H, H,
        h_prev.stride(0), h_prev.stride(1),
        weight_hh.stride(1), weight_hh.stride(0),
        hW.stride(0), hW.stride(1),
    )
    
    # gates = xW + hW + bias_ih + bias_hh, then activate
    gates = xW + hW + bias_ih.unsqueeze(0) + bias_hh.unsqueeze(0)
    
    # Activate gates in-place
    grid_act = (triton.cdiv(batch, 16), triton.cdiv(H, 64))
    lstm_gates_activate_kernel[grid_act](
        gates, batch, H,
        gates.stride(0),
    )
    
    # Combine gates to get h_new, c_new
    h_new = torch.empty(batch, H, device=x_t.device, dtype=torch.float32)
    c_new = torch.empty(batch, H, device=x_t.device, dtype=torch.float32)
    
    grid_comb = (triton.cdiv(batch, 16), triton.cdiv(H, 64))
    lstm_cell_combine_kernel[grid_comb](
        gates, c_prev, h_new, c_new,
        batch, H,
        gates.stride(0),
        c_prev.stride(0),
        h_new.stride(0),
        c_new.stride(0),
    )
    
    return h_new, c_new


def kernel_function(
    p_lstm_weight_ih_l0, p_lstm_weight_hh_l0, p_lstm_bias_ih_l0, p_lstm_bias_hh_l0,
    p_lstm_weight_ih_l1, p_lstm_weight_hh_l1, p_lstm_bias_ih_l1, p_lstm_bias_hh_l1,
    p_lstm_weight_ih_l2, p_lstm_weight_hh_l2, p_lstm_bias_ih_l2, p_lstm_bias_hh_l2,
    p_lstm_weight_ih_l3, p_lstm_weight_hh_l3, p_lstm_bias_ih_l3, p_lstm_bias_hh_l3,
    p_lstm_weight_ih_l4, p_lstm_weight_hh_l4, p_lstm_bias_ih_l4, p_lstm_bias_hh_l4,
    p_lstm_weight_ih_l5, p_lstm_weight_hh_l5, p_lstm_bias_ih_l5, p_lstm_bias_hh_l5,
    p_fc_weight, p_fc_bias,
):
    """
    """
    batch_size, seq_len, input_size = x.shape
    num_layers = 6
    hidden_size = 256
    
    # Layer weights and biases
    layer_params = [
        (p_lstm_weight_ih_l0, p_lstm_weight_hh_l0, p_lstm_bias_ih_l0, p_lstm_bias_hh_l0),
        (p_lstm_weight_ih_l1, p_lstm_weight_hh_l1, p_lstm_bias_ih_l1, p_lstm_bias_hh_l1),
        (p_lstm_weight_ih_l2, p_lstm_weight_hh_l2, p_lstm_bias_ih_l2, p_lstm_bias_hh_l2),
        (p_lstm_weight_ih_l3, p_lstm_weight_hh_l3, p_lstm_bias_ih_l3, p_lstm_bias_hh_l3),
        (p_lstm_weight_ih_l4, p_lstm_weight_hh_l4, p_lstm_bias_ih_l4, p_lstm_bias_hh_l4),
        (p_lstm_weight_ih_l5, p_lstm_weight_hh_l5, p_lstm_bias_ih_l5, p_lstm_bias_hh_l5),
    ]
    
    # Initialize hidden and cell states
    # h0, c0: [num_layers, batch_size, hidden_size]
    h_states = [h0[l].clone() for l in range(num_layers)]
    c_states = [c0[l].clone() for l in range(