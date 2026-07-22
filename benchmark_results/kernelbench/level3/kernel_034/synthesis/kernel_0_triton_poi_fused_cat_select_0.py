import torch
import triton
import triton.language as tl


# Fused kernel: cat([x_t, hidden], dim=1) @ W_i2h.T + b_i2h -> tanh -> hidden_new
# Then hidden_new @ W_h2o.T + b_h2o -> output
# x_t: [B, input_size], hidden: [B, hidden_size]
# W_i2h: [hidden_size, input_size+hidden_size], b_i2h: [hidden_size]
# W_h2o: [output_size, hidden_size], b_h2o: [output_size]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['BATCH', 'INPUT_SIZE', 'HIDDEN_SIZE'],
)
@triton.jit
def rnn_i2h_kernel(
    BATCH: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (INPUT_SIZE + HIDDEN_SIZE)
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load from concatenated [x_t, hidden]
        # k < INPUT_SIZE: load from x_ptr
        # k >= INPUT_SIZE: load from h_ptr
        k_in_x = offs_k < INPUT_SIZE
        k_in_h = offs_k >= INPUT_SIZE

        x_k = tl.where(k_in_x, offs_k, 0)
        h_k = tl.where(k_in_h, offs_k - INPUT_SIZE, 0)

        m_mask = offs_m < BATCH
        k_mask = offs_k < K_TOTAL

        # Load x part
        x_ptrs = x_ptr + offs_m[:, None] * INPUT_SIZE + x_k[None, :]
        x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_in_x[None, :] & k_mask[None, :], other=0.0)

        # Load h part
        h_ptrs = h_ptr + offs_m[:, None] * HIDDEN_SIZE + h_k[None, :]
        h_vals = tl.load(h_ptrs, mask=m_mask[:, None] & k_in_h[None, :] & k_mask[None, :], other=0.0)

        a_tile = x_vals + h_vals  # one is zero, so this is the concatenated value

        # Load weight tile [BLOCK_K, BLOCK_N] from w_ptr [HIDDEN_SIZE, K_TOTAL]
        n_mask = offs_n < HIDDEN_SIZE
        w_ptrs = w_ptr + offs_k[:, None] * HIDDEN_SIZE + offs_n[None, :]
        # Wait, w is [HIDDEN_SIZE, K_TOTAL] with stride K_TOTAL per row
        # w[n, k] -> w_ptr + n * K_TOTAL + k
        # We want w_tile[k, n] for tl.dot(a[m,k], w[k,n])
        w_ptrs2 = w_ptr + offs_n[None, :] * K_TOTAL + offs_k[:, None]
        w_tile = tl.load(w_ptrs2, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc = tl.dot(a_tile, w_tile, acc=acc)

    # Add bias and apply tanh
    n_mask = offs_n < HIDDEN_SIZE
    m_mask = offs_m < BATCH
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]
    acc = tl.extra.libdevice.tanh(acc)

    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * HIDDEN_SIZE + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['BATCH', 'HIDDEN_SIZE', 'OUTPUT_SIZE'],
)
@triton.jit
def rnn_h2o_kernel(
    BATCH: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < HIDDEN_SIZE
        m_mask = offs_m < BATCH
        n_mask = offs_n < OUTPUT_SIZE

        # Load h tile [BLOCK_M, BLOCK_K]
        h_ptrs = h_ptr + offs_m[:, None] * HIDDEN_SIZE + offs_k[None, :]
        h_tile = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load w tile [BLOCK_K, BLOCK_N]: w[n, k] -> w_ptr + n*HIDDEN_SIZE + k
        # We want w_tile[k, n]
        w_ptrs = w_ptr + offs_n[None, :] * HIDDEN_SIZE + offs_k[:, None]
        w_tile = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc = tl.dot(h_tile, w_tile, acc=acc)

    n_mask = offs_n < OUTPUT_SIZE
    m_mask = offs_m < BATCH
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * OUTPUT_SIZE + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0):
    # x: [seq_len, batch_size, input_size]
    # h0: [batch_size, hidden_size]
    seq_len, batch_size, input_size = x.shape
    hidden_size = h0.shape[1]
    output_size = p_h2o_weight.shape[0]

    BATCH = batch_size
    INPUT_SIZE = input_size
    HIDDEN_SIZE = hidden_size
    OUTPUT_SIZE = output_size
    K_TOTAL = INPUT_SIZE + HIDDEN_SIZE

    hidden = h0.contiguous()
    outputs = torch.empty(seq_len, batch_size, output_size, device=x.device, dtype=x.dtype)

    # Ensure weights are contiguous
    w_i2h = p_i2h_weight.contiguous()  # [hidden_size, K_TOTAL]
    b_i2h = p_i2h_bias.contiguous()
    w_h2o = p_h2o_weight.contiguous()  # [output_size, hidden_size]
    b_h2o = p_h2o_bias.contiguous()

    hidden_buf = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)
    output_buf = torch.empty(batch_size, output_size, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        x_t = x[t].contiguous()  # [batch_size, input_size]

        # i2h: cat([x_t, hidden]) @ W_i2h.T + b_i2h -> tanh
        grid_i2h = (triton.cdiv(BATCH, 8), triton.cdiv(HIDDEN_SIZE, 32))
        rnn_i2h_kernel[grid_i2h](
            x_t, hidden, w_i2h, b_i2h, hidden_buf,
            BATCH=BATCH,
            INPUT_SIZE=INPUT_SIZE,
            HIDDEN_SIZE=HIDDEN_SIZE,
            K_TOTAL=K_TOTAL,
        )

        # h2o: hidden_buf @ W_h2o.T + b_h2o
        grid_h2o = (triton.cdiv(BATCH, 8), triton.cdiv(OUTPUT_SIZE, 32))
        rnn_h2o_kernel[grid_h2o](
            hidden_buf, w_h2o, b_h2o, output_buf,
            BATCH=BATCH,
            HIDDEN_SIZE=HIDDEN_SIZE,
            OUTPUT_SIZE=OUTPUT_SIZE,
        )

        outputs[t] = output_buf
        hidden = hidden_buf.clone()

    return outputs


def get_inputs():
    batch_size = 8
    input_size = 1024
    hidden_size = 256
    output_size = 128
    sequence_length = 256

    p_i2h_weight = torch.randn(hidden_size, input_size + hidden_size, device='cuda')
    p_i2h_bias = torch.randn(hidden_size, device='cuda')
    p_h2o_weight = torch.randn(output_size, hidden_size, device='cuda')
    p_h2o_bias = torch.randn(output_size, device='cuda')
    x = torch.rand(sequence_length, batch_size, input_size, device='cuda')
    h0 = torch.rand(batch_size, hidden_size, device='cuda')

    return [p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0 = inputs

        # Reference implementation
        import torch.nn as nn
        seq_len, batch_size, input_size = x.shape
        hidden_size = h0.shape[1]
        output_size = p_h2o_weight.shape[0]

        hidden = h0.clone()
        ref_outputs = []
        for t in range(seq_len):
            combined = torch.cat((x[t], hidden), dim=1)
            hidden = torch.tanh(combined @ p_i2h_weight.T + p_i2h_bias)
            out = hidden @ p_h2o_weight.T + p_h2o_bias
            ref_outputs.append(out)
        ref = torch.stack(ref_outputs, dim=0)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")