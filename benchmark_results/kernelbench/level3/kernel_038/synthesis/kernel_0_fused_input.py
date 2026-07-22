import torch
import triton
import triton.language as tl

# [Full implementation below]

import torch
import triton
import triton.language as tl
import argparse

# LSTM gate kernel: computes gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
# Then applies sigmoid/tanh activations and updates cell/hidden state
# We implement this as a fused kernel operating on one timestep at a time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['batch_size', 'input_size', 'hidden_size'],
)
@triton.jit
def lstm_gemm_kernel(
    # Input: x_t [batch, input_size]
    x_ptr, x_stride_b, x_stride_i,
    # Weight ih: [4*hidden, input_size]
    wih_ptr, wih_stride_r, wih_stride_c,
    # Weight hh: [4*hidden, hidden_size]
    whh_ptr, whh_stride_r, whh_stride_c,
    # Bias ih: [4*hidden]
    bih_ptr,
    # Bias hh: [4*hidden]
    bhh_ptr,
    # Hidden state h: [batch, hidden_size]
    h_ptr, h_stride_b, h_stride_h,
    # Cell state c: [batch, hidden_size]
    c_ptr, c_stride_b, c_stride_h,
    # Output h_new: [batch, hidden_size]
    h_new_ptr, hn_stride_b, hn_stride_h,
    # Output c_new: [batch, hidden_size]
    c_new_ptr, cn_stride_b, cn_stride_h,
    # Dimensions
    batch_size, input_size, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
      gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh  [batch, 4*hidden]
      i, f, g, o = split(gates, 4)
      i = sigmoid(i), f = sigmoid(f), g = tanh(g), o = sigmoid(o)
      c_new = f * c + i * g
      h_new = o * tanh(c_new)
    
    """
    pid_m = tl.program_id(0)  # batch tile
    pid_n = tl.program_id(1)  # hidden tile (for one gate)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size

    # Accumulate 4 gates: i, f, g, o
    # gate index: 0=i, 1=f, 2=g, 3=o
    # Each gate corresponds to rows [gate*hidden_size : (gate+1)*hidden_size] of W_ih, W_hh

    # We'll compute all 4 gates in one block
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute x @ W_ih.T for all 4 gates
    # W_ih shape: [4*hidden, input_size], so W_ih.T shape: [input_size, 4*hidden]
    # We want x[batch, :] @ W_ih[gate*H:(gate+1)*H, :].T
    for k in range(0, input_size, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < input_size

        # Load x tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * x_stride_b + offs_k[None, :] * x_stride_i
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W_ih tiles for each gate: [BLOCK_N, BLOCK_K] -> transpose to [BLOCK_K, BLOCK_N]
        # gate i: rows offs_n
        wi_ptrs_i = wih_ptr + offs_n[:, None] * wih_stride_r + offs_k[None, :] * wih_stride_c
        wi_tile_i = tl.load(wi_ptrs_i, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wi_ptrs_f = wih_ptr + (offs_n[:, None] + hidden_size) * wih_stride_r + offs_k[None, :] * wih_stride_c
        wi_tile_f = tl.load(wi_ptrs_f, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wi_ptrs_g = wih_ptr + (offs_n[:, None] + 2 * hidden_size) * wih_stride_r + offs_k[None, :] * wih_stride_c
        wi_tile_g = tl.load(wi_ptrs_g, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wi_ptrs_o = wih_ptr + (offs_n[:, None] + 3 * hidden_size) * wih_stride_r + offs_k[None, :] * wih_stride_c
        wi_tile_o = tl.load(wi_ptrs_o, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # acc += x_tile @ wi_tile.T
        acc_i = tl.dot(x_tile, tl.trans(wi_tile_i), acc=acc_i)
        acc_f = tl.dot(x_tile, tl.trans(wi_tile_f), acc=acc_f)
        acc_g = tl.dot(x_tile, tl.trans(wi_tile_g), acc=acc_g)
        acc_o = tl.dot(x_tile, tl.trans(wi_tile_o), acc=acc_o)

    # Compute h @ W_hh.T for all 4 gates
    for k in range(0, hidden_size, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < hidden_size

        # Load h tile: [BLOCK_M, BLOCK_K]
        h_ptrs = h_ptr + offs_m[:, None] * h_stride_b + offs_k[None, :] * h_stride_h
        h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W_hh tiles for each gate
        wh_ptrs_i = whh_ptr + offs_n[:, None] * whh_stride_r + offs_k[None, :] * whh_stride_c
        wh_tile_i = tl.load(wh_ptrs_i, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wh_ptrs_f = whh_ptr + (offs_n[:, None] + hidden_size) * whh_stride_r + offs_k[None, :] * whh_stride_c
        wh_tile_f = tl.load(wh_ptrs_f, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wh_ptrs_g = whh_ptr + (offs_n[:, None] + 2 * hidden_size) * whh_stride_r + offs_k[None, :] * whh_stride_c
        wh_tile_g = tl.load(wh_ptrs_g, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        wh_ptrs_o = whh_ptr + (offs_n[:, None] + 3 * hidden_size) * whh_stride_r + offs_k[None, :] * whh_stride_c
        wh_tile_o = tl.load(wh_ptrs_o, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc_i = tl.dot(h_tile, tl.trans(wh_tile_i), acc=acc_i)
        acc_f = tl.dot(h_tile, tl.trans(wh_tile_f), acc=acc_f)
        acc_g = tl.dot(h_tile, tl.trans(wh_tile_g), acc=acc_g)
        acc_o = tl.dot(h_tile, tl.trans(wh_tile_o), acc=acc_o)

    # Add biases
    bih_i = tl.load(bih_ptr + offs_n, mask=mask_n, other=0.0)
    bih_f = tl.load(bih_ptr + offs_n + hidden_size, mask=mask_n, other=0.0)
    bih_g = tl.load(bih_ptr + offs_n + 2 * hidden_size, mask=mask_n, other=0.0)
    bih_o = tl.load(bih_ptr + offs_n + 3 * hidden_size, mask=mask_n, other=0.0)

    bhh_i = tl.load(bhh_ptr + offs_n, mask=mask_n, other=0.0)
    bhh_f = tl.load(bhh_ptr + offs_n + hidden_size, mask=mask_n, other=0.0)
    bhh_g = tl.load(bhh_ptr + offs_n + 2 * hidden_size, mask=mask_n, other=0.0)
    bhh_o = tl.load(bhh_ptr + offs_n + 3 * hidden_size, mask=mask_n, other=0.0)

    gate_i = acc_i + bih_i[None, :] + bhh_i[None, :]
    gate_f = acc_f + bih_f[None, :] + bhh_f[None, :]
    gate_g = acc_g + bih_g[None, :] + bhh_g[None, :]
    gate_o = acc_o + bih_o[None, :] + bhh_o[None, :]

    # Activations: sigmoid for i, f, o; tanh for g
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig_i = 1.0 / (1.0 + tl.exp(-gate_i))
    sig_f = 1.0 / (1.0 + tl.exp(-gate_f))
    tanh_g = tl.extra.libdevice.tanh(gate_g.to(tl.float32))
    sig_o = 1.0 / (1.0 + tl.exp(-gate_o))

    # Load cell state c: [BLOCK_M, BLOCK_N]
    c_ptrs = c_ptr + offs_m[:, None] * c_stride_b + offs_n[None, :] * c_stride_h
    c_tile = tl.load(c_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Update cell and hidden state
    c_new = sig_f * c_tile + sig_i * tanh_g
    tanh_c_new = tl.extra.libdevice.tanh(c_new.to(tl.float32))
    h_new = sig_o * tanh_c_new

    # Store h_new and c_new
    hn_ptrs = h_new_ptr + offs_m[:, None] * hn_stride_b + offs_n[None, :] * hn_stride_h
    tl.store(hn_ptrs, h_new.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])

    cn_ptrs = c_new_ptr + offs_m[:, None] * cn_stride_b + offs_n[None, :] * cn_stride_h
    tl.store(cn_ptrs, c_new.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
@triton.jit
def linear_kernel(
    x_ptr, x_stride_b, x_stride_f,
    w_ptr, w_stride_o, w_stride_i,
    b_ptr,
    out_ptr, out_stride_b, out_stride_o,
    batch_size, in_features, out_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < in_features

        x_ptrs = x_ptr + offs_m[:, None] * x_stride_b + offs_k[None, :] * x_stride_f
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = w_ptr + offs_n[:, None] * w_stride_o + offs_k[None, :] * w_stride_i
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * out_stride_b + offs_n[None, :] * out_stride_o
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def run_lstm_layer(x, wih, whh, bih, bhh, h, c, reverse=False):
    """
    """
    batch_size, seq_len, input_size = x.shape
    hidden_size = h.shape[1]

    outputs = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)

    h_cur = h.contiguous()
    c_cur = c.contiguous()

    time_steps = range(seq_len) if not reverse else range(seq_len - 1, -1, -1)

    for t in time_steps:
        x_t = x[:, t, :].contiguous()  # [batch, input_size]
        h_new = torch.empty_like(h_cur)
        c_new = torch.empty_like(c_cur)

        grid = lambda meta: (
            triton.cdiv(batch_size, meta['BLOCK_M']),
            triton.cdiv(hidden_size, meta['BLOCK_N']),
        )

        lstm_gemm_kernel[grid](
            x_t, x_t.stride(0), x_t.stride(1),
            wih, wih.stride(0), wih.stride(1),
            whh, whh.stride(0), whh.stride(1),
            bih, bhh,
            h_cur, h_cur.stride(0), h_cur.stride(1),
            c_cur, c_cur.stride(0), c_cur.stride(1),
            h_new, h_new.stride(0), h_new.stride(1),
            c_new, c_new.stride(0), c_new.stride(1),
            batch_size, input_size, hidden_size,
        )

        if not reverse:
            outputs[:, t, :] = h_new
        else:
            outputs[:, t, :] = h_new

        h_cur = h_new
        c_cur = c_new

    return outputs, h_cur, c_cur


def kernel_function(
    p_lstm_weight_ih_l0, p_lstm_weight_hh_l0, p_lstm_bias_ih_l0, p_lstm_bias_hh_l0,
    p_lstm_weight_ih_l0_reverse, p_lstm_weight_hh_l0_reverse, p_lstm_bias_ih_l0_reverse, p_lstm_bias_hh_l0_reverse,
    p_lstm_weight_ih_l1, p_lstm_weight_hh_l1, p_lstm_bias_ih_l1, p_lstm_bias_hh_l1,
    p_lstm_weight_ih_l1_reverse, p_lstm_weight_hh_l1_reverse, p_lstm_bias_ih_l1_reverse, p_lstm_bias_hh_l1_reverse,
    p_lstm_weight_ih_l2, p_lstm_weight_hh_l2, p_lstm_bias_ih_l2, p_lstm_bias_hh_l2,
    p_lstm_weight_ih_l2_reverse, p_lstm_weight_hh_l2_reverse, p_lstm_bias_ih_l2_reverse, p_lstm_bias_hh_l2_reverse,
    p_lstm_weight_ih_l3, p_lstm_weight_hh_l3, p_lstm_bias_ih_l3, p_lstm_bias_hh_l3,
    p_lstm_weight_ih_l3_reverse, p_lstm_weight_hh_l3_reverse, p_lstm_bias_ih_l3_reverse, p_lstm_bias_hh_l3_reverse,
    p_lstm_weight_ih_l4, p_lstm_weight_hh_l4, p_lstm_bias_ih_l4, p_lstm_bias_hh_l4,
    p_lstm_weight_ih_l4_reverse, p_lstm_weight_hh_l4_reverse, p_lstm_bias_ih_l4_reverse, p_lstm_bias_hh_l4_reverse,
    p_lstm_weight_ih_l5, p_lstm_weight_hh_l5, p_lstm_bias_ih_l5, p_lstm_bias_hh_l5,
    p_lstm_weight_ih_l5_reverse, p_lstm_weight_hh_l5_reverse, p_lstm_bias_ih_l5_reverse, p_lstm_bias_hh_l5_reverse,
    p_fc_weight, p_fc_bias,
    x, h0, c0,
):
    batch_size, seq_len, input_size = x.shape
    hidden_size = h0.shape[2]
    num_layers = 6

    # Layer parameters: (wih, whh, bih, bhh, wih_rev, whh_rev, bih_rev, bhh_rev)
    layer_params = [
        (p_lstm_weight_ih_l0, p_lstm_weight_hh_l0, p_lstm_bias_ih_l0, p_lstm_bias_hh_l0,
         p_lstm_weight_ih_l0_reverse, p_lstm_weight_hh_l0_reverse, p_lstm_bias_ih_l0_reverse, p_lstm_bias_hh_l0_reverse),
        (p_lstm_weight_ih_l1, p_lstm_weight_hh_l1, p_lstm_bias_ih_l1, p_lstm_bias_hh_l1,
         p_lstm_weight_ih_l1_reverse, p_lstm_weight_hh_l1_reverse, p_lstm_bias_ih_l1_reverse, p_lstm_bias_hh_l1_reverse),
        (p_lstm_weight_ih_l2, p_lstm_weight_hh_l2, p_lstm_bias_ih_l2, p_lstm_bias_hh_l2,
         p_lstm_weight_ih_l2_reverse, p_lstm_weight_hh_l2_reverse, p_lstm_bias_ih_l2_reverse, p_lstm_bias_hh_l2_reverse),
        (p_lstm_weight_ih_l3, p_lstm_weight_hh_l3, p_lstm_bias_ih_l3, p_lstm_bias_hh_l3,
         p_lstm_weight_ih_l3_reverse, p_lstm_weight_hh_l3_reverse, p_lstm_bias_ih_l3_reverse, p_lstm_bias_hh_l3_reverse),
        (p_lstm_weight_ih_l4, p_lstm_weight_hh_l4, p_lstm_bias_ih_l4, p_lstm_bias_hh_l4,
         p_lstm_weight_ih_l4_reverse, p_lstm_weight_hh_l4_reverse, p_lstm_bias_ih_l4_reverse, p_lstm_bias_hh_l4_reverse),
        (p_lstm_weight_ih_l5, p_lstm_weight_hh_l5, p_lstm_bias_ih_l5, p_lstm_bias_hh_l5,
         p_lstm_weight_ih_l5_reverse, p_lstm_weight_hh_l5_reverse, p_lstm_bias_ih_l5_reverse, p_lstm_bias_hh_l5_reverse),
    ]

    # h0, c0 shape: [num_layers*2, batch, hidden_size]
    # For bidirectional: layer l forward uses h0[2*l], reverse uses h0[2*l+1]

    layer_input = x  # [batch, seq, input_size]

    for layer_idx in range(num_layers):
        wih, whh, bih, bhh, wih_r, whh_r, bih_r, bhh_r = layer_params[layer_idx]

        h_fwd = h0[2 * layer_idx].contiguous()   # [batch, hidden]
        c_fwd = c0[2 * layer_idx].contiguous()
        h_rev = h0[2 * layer_idx + 1].contiguous()
        c_rev = c0[2 * layer_idx + 1].contiguous()

        # Forward direction
        out_fwd, h_fwd_n, c_fwd_n = run_lstm_layer(
            layer_input, wih, whh, bih, bhh, h_fwd, c_fwd, reverse=False
        )
        # Reverse direction
        out_rev, h_rev_n, c_rev_n = run_lstm_layer(
            layer_input, wih_r, whh_r, bih_r, bhh_r, h_rev, c_rev, reverse=True
        )

        # Concatenate forward and reverse outputs: [batch, seq, 2*hidden]
        layer_input = torch.cat([out_fwd, out_rev], dim=2)

    # layer_input is now [batch, seq, 2*hidden] = [batch, seq, 512]
    # Select last timestep: [batch, 512]
    last_out = layer_input[:, -1, :].contiguous()

    # Apply linear layer
    out_features = p_fc_weight.shape[0]
    in_features = p_fc_weight.shape[1]
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(batch_size, meta['BLOCK_M']),
        triton.cdiv(out_features, meta['BLOCK_N']),
    )

    linear_kernel[grid](
        last_out, last_out.stride(0), last_out.stride(1),
        p_fc_weight, p_fc_weight.stride(0), p_fc_weight.stride(1),
        p_fc_bias,
        output, output.stride(0), output.stride(1),
        batch_size, in_features, out_features,
    )

    return output


def get_inputs():
    batch_size = 10
    sequence_length = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6

    # Build all weight tensors
    # Layer 0: input_size=128, hidden_size=256
    wih_l0 = torch.rand(1024, 128, dtype=torch.float32, device='cuda')
    whh_l0 = torch.rand(1024, 256, dtype=torch.float32, device='cuda')
    bih_l0 = torch.rand(1024, dtype=torch.float32, device='cuda')
    bhh_l0 = torch.rand(1024, dtype=torch.float32, device='cuda')
    wih_l0r = torch.rand(1024, 128, dtype=torch.float32, device='cuda')
    whh_l0r = torch.rand(1024, 256, dtype=torch.float32, device='cuda')
    bih_l0r = torch.rand(1024, dtype=torch.float32, device='cuda')
    bhh_l0r = torch.rand(1024, dtype=torch.float32, device='cuda')

    # Layers 1-5: input_size=512 (2*hidden from prev layer), hidden_size=256
    layers = []
    for i in range(1, 6):
        wih = torch.rand(1024, 512, dtype=torch.float32, device='cuda')
        whh = torch.rand(1024, 256, dtype=torch.float32, device='cuda')
        bih = torch.rand(1024, dtype=torch.float32, device='cuda')
        bhh = torch.rand(1024, dtype=torch.float32, device='cuda')
        wih_r = torch.rand(1024, 512, dtype=torch.float32, device='cuda')
        whh_r = torch.rand(1024, 256, dtype=torch.float32, device='cuda')
        bih_r = torch.rand(1024, dtype=torch.float32, device='cuda')
        bhh_r = torch.rand(1024, dtype=torch.float32, device='cuda')
        layers.append((wih, whh, bih, bhh, wih_r, whh_r, bih_r, bhh_r))

    fc_weight = torch.rand(10, 512, dtype=torch.float32, device='cuda')
    fc_bias = torch.rand(10, dtype=torch.float32, device='cuda')

    x = torch.rand(batch_size, sequence_length, input_size, dtype=torch.float32, device='cuda')
    h0 = torch