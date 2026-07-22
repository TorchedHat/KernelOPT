import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['batch_size', 'input_size', 'hidden_size', 'output_size'],
)
@triton.jit
def rnn_step_i2h_kernel(
    # x: (seq_len, batch, input_size)
    x_ptr,
    # h: (batch, hidden_size) - current hidden state
    h_ptr,
    # i2h weight: (hidden_size, input_size + hidden_size)
    w_i2h_ptr,
    # i2h bias: (hidden_size,)
    b_i2h_ptr,
    # output hidden: (batch, hidden_size)
    h_out_ptr,
    # dimensions
    batch_size, input_size, hidden_size,
    # strides for x
    x_stride_seq, x_stride_batch, x_stride_feat,
    # timestep index
    t,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a tile of (BLOCK_M rows of batch, BLOCK_N cols of hidden)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # K = input_size + hidden_size = 1280
    K = input_size + hidden_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: rows=batch, cols=K (concatenated [x[t], h])
        # For k < input_size: load from x[t]
        # For k >= input_size: load from h
        k_in_x = offs_k < input_size
        k_in_h = offs_k >= input_size

        # x part: x[t, batch_row, k]
        x_k = offs_k  # column in x (valid when k < input_size)
        h_k = offs_k - input_size  # column in h (valid when k >= input_size)

        # Build A tile [BLOCK_M, BLOCK_K]
        m_mask = offs_m[:, None] < batch_size
        k_mask_x = (offs_k[None, :] < input_size) & m_mask
        k_mask_h = (offs_k[None, :] >= input_size) & (offs_k[None, :] < K) & m_mask

        x_ptrs = x_ptr + t * x_stride_seq + offs_m[:, None] * x_stride_batch + x_k[None, :]
        h_ptrs = h_ptr + offs_m[:, None] * hidden_size + h_k[None, :]

        a_x = tl.load(x_ptrs, mask=k_mask_x, other=0.0)
        a_h = tl.load(h_ptrs, mask=k_mask_h, other=0.0)
        a = tl.where(offs_k[None, :] < input_size, a_x, a_h)

        # Load B tile: w_i2h[hidden_col, k] shape (BLOCK_N, BLOCK_K) -> need (BLOCK_K, BLOCK_N)
        # w_i2h is (hidden_size, K), row=hidden_dim, col=k
        n_mask = offs_n[None, :] < hidden_size
        k_full_mask = offs_k[:, None] < K

        w_ptrs = w_i2h_ptr + offs_n[None, :] * K + offs_k[:, None]
        b_tile = tl.load(w_ptrs, mask=k_full_mask & n_mask, other=0.0)

        acc = tl.dot(a, b_tile, acc=acc)

    # Add bias and apply tanh
    bias = tl.load(b_i2h_ptr + offs_n, mask=offs_n < hidden_size, other=0.0)
    acc = acc + bias[None, :]
    acc = libdevice.tanh(acc)

    # Store result
    m_mask = offs_m[:, None] < batch_size
    n_mask = offs_n[None, :] < hidden_size
    tl.store(h_out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :],
             acc, mask=m_mask & n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['batch_size', 'hidden_size', 'output_size'],
)
@triton.jit
def rnn_step_h2o_kernel(
    # h: (batch, hidden_size)
    h_ptr,
    # h2o weight: (output_size, hidden_size)
    w_h2o_ptr,
    # h2o bias: (output_size,)
    b_h2o_ptr,
    # output: (batch, output_size)
    out_ptr,
    # dimensions
    batch_size, hidden_size, output_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        m_mask = offs_m[:, None] < batch_size
        k_mask = offs_k[None, :] < hidden_size

        a_ptrs = h_ptr + offs_m[:, None] * hidden_size + offs_k[None, :]
        a = tl.load(a_ptrs, mask=m_mask & k_mask, other=0.0)

        n_mask = offs_n[None, :] < output_size
        k_full_mask = offs_k[:, None] < hidden_size

        w_ptrs = w_h2o_ptr + offs_n[None, :] * hidden_size + offs_k[:, None]
        b_tile = tl.load(w_ptrs, mask=k_full_mask & n_mask, other=0.0)

        acc = tl.dot(a, b_tile, acc=acc)

    bias = tl.load(b_h2o_ptr + offs_n, mask=offs_n < output_size, other=0.0)
    acc = acc + bias[None, :]

    m_mask = offs_m[:, None] < batch_size
    n_mask = offs_n[None, :] < output_size
    tl.store(out_ptr + offs_m[:, None] * output_size + offs_n[None, :],
             acc, mask=m_mask & n_mask)


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0):
    seq_len, batch_size, input_size = x.shape
    hidden_size = h0.shape[1]
    output_size = p_h2o_weight.shape[0]

    # Allocate outputs
    outputs = torch.empty(seq_len, batch_size, output_size, device=x.device, dtype=x.dtype)
    hidden = h0.contiguous().clone()

    # Temp buffer for hidden state
    h_buf = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    x_stride_seq = x.stride(0)
    x_stride_batch = x.stride(1)
    x_stride_feat = x.stride(2)

    for t in range(seq_len):
        # i2h step: compute new hidden
        grid_i2h = lambda meta: (
            triton.cdiv(batch_size, meta['BLOCK_M']),
            triton.cdiv(hidden_size, meta['BLOCK_N']),
        )
        rnn_step_i2h_kernel[grid_i2h](
            x, hidden, p_i2h_weight, p_i2h_bias, h_buf,
            batch_size, input_size, hidden_size,
            x_stride_seq, x_stride_batch, x_stride_feat,
            t,
        )

        # h2o step: compute output
        grid_h2o = lambda meta: (
            triton.cdiv(batch_size, meta['BLOCK_M']),
            triton.cdiv(output_size, meta['BLOCK_N']),
        )
        rnn_step_h2o_kernel[grid_h2o](
            h_buf, p_h2o_weight, p_h2o_bias,
            outputs[t],
            batch_size, hidden_size, output_size,
        )

        # Update hidden
        hidden.copy_(h_buf)

    return outputs


def get_inputs():
    seq_len = 256
    batch_size = 8
    input_size = 1024
    hidden_size = 256
    output_size = 128
    return [
        torch.randn(hidden_size, input_size + hidden_size, device='cuda'),  # p_i2h_weight
        torch.randn(hidden_size, device='cuda'),  # p_i2h_bias
        torch.randn(output_size, hidden_size, device='cuda'),  # p_h2o_weight
        torch.randn(output_size, device='cuda'),  # p_h2o_bias
        torch.randn(seq_len, batch_size, input_size, device='cuda'),  # x
        torch.randn(batch_size, hidden_size, device='cuda'),  # h0
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0 = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
                self.h2o = nn.Linear(hidden_size, output_size)
                self.tanh = nn.Tanh()

            def forward(self, x, h0):
                seq_len, batch_size, _ = x.size()
                hidden = h0.clone()
                outputs = []
                for t in range(seq_len):
                    combined = torch.cat((x[t], hidden), dim=1)
                    hidden = self.tanh(self.i2h(combined))
                    out = self.h2o(hidden)
                    outputs.append(out)
                return torch.stack(outputs, dim=0)

        model = Model(1024, 256, 128).cuda()
        with torch.no_grad():
            model.i2h.weight.copy_(p_i2h_weight)
            model.i2h.bias.copy_(p_i2h_bias)
            model.h2o.weight.copy_(p_h2o_weight)
            model.h2o.bias.copy_(p_h2o_bias)
            ref = model(x, h0)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")