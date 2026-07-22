import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N_i2h', 'K_i2h'],
)
@triton.jit
def rnn_i2h_kernel(
    x_ptr, hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N_i2h, K_x, K_h, K_i2h,
    stride_x_batch, stride_x_feat,
    stride_h_batch, stride_h_feat,
    stride_w_out, stride_w_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Computes: out = tanh(cat(x, hidden) @ weight.T + bias)
    # x: [M, K_x], hidden: [M, K_h], weight: [N_i2h, K_i2h] where K_i2h = K_x + K_h
    # out: [M, N_i2h]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop 1: x part, weight columns [0, K_x)
    for k in range(0, tl.cdiv(K_x, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_x)
        inp = tl.load(x_ptr + offs_m[:, None] * stride_x_batch + offs_k[None, :] * stride_x_feat,
                      mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        w_mask = (offs_k[:, None] < K_x) & (offs_n[None, :] < N_i2h)
        w = tl.load(weight_ptr + offs_k[:, None] * stride_w_in + offs_n[None, :] * stride_w_out,
                    mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(inp, w, allow_tf32=False)

    # Loop 2: hidden part, weight columns [K_x, K_i2h)
    for k in range(0, tl.cdiv(K_h, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_h)
        inp = tl.load(hidden_ptr + offs_m[:, None] * stride_h_batch + offs_k[None, :] * stride_h_feat,
                      mask=h_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        k_global = offs_k + K_x  # offset into weight K dimension
        w_mask = (k_global[:, None] < K_i2h) & (offs_n[None, :] < N_i2h)
        w = tl.load(weight_ptr + k_global[:, None] * stride_w_in + offs_n[None, :] * stride_w_out,
                    mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(inp, w, allow_tf32=False)

    # Add bias and tanh
    bias_mask = offs_n < N_i2h
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = libdevice.tanh(acc)

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_i2h)
    tl.store(out_ptr + offs_m[:, None] * N_i2h + offs_n[None, :], acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N_h2o', 'K_h2o'],
)
@triton.jit

def rnn_h2o_kernel(
    hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N_h2o, K_h2o,
    stride_h_batch, stride_h_feat,
    stride_w_out, stride_w_in,
    stride_o_batch, stride_o_feat,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Computes: out = hidden @ weight.T + bias
    # hidden: [M, K_h2o], weight: [N_h2o, K_h2o]
    # out: [M, N_h2o]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_h2o, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load hidden [BLOCK_M, BLOCK_K]
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_h2o)
        h = tl.load(hidden_ptr + offs_m[:, None] * stride_h_batch + offs_k[None, :] * stride_h_feat,
                    mask=h_mask, other=0.0)

        # Load weight [BLOCK_K, BLOCK_N] (weight is [N_h2o, K_h2o], transposed access)
        w_mask = (offs_k[:, None] < K_h2o) & (offs_n[None, :] < N_h2o)
        w = tl.load(weight_ptr + offs_k[:, None] * stride_w_in + offs_n[None, :] * stride_w_out,
                    mask=w_mask, other=0.0)

        acc += tl.dot(h, w, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < N_h2o
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_h2o)
    tl.store(out_ptr + offs_m[:, None] * stride_o_batch + offs_n[None, :] * stride_o_feat,
             acc, mask=out_mask)


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0):
    # x: [seq_len, batch_size, input_size] = [256, 8, 1024]
    # h0: [batch_size, hidden_size] = [8, 256]
    # p_i2h_weight: [256, 1280], p_i2h_bias: [256]
    # p_h2o_weight: [128, 256], p_h2o_bias: [128]

    seq_len, batch_size, input_size = x.shape
    hidden_size = h0.shape[1]
    output_size = p_h2o_weight.shape[0]

    hidden = h0.contiguous()
    outputs = torch.empty(seq_len, batch_size, output_size, device=x.device, dtype=x.dtype)

    M = batch_size
    N_i2h = hidden_size  # 256
    K_x = input_size     # 1024
    K_h = hidden_size    # 256
    K_i2h = K_x + K_h   # 1280

    N_h2o = output_size  # 128
    K_h2o = hidden_size  # 256

    # Precompute strides for weights
    # i2h_weight: [256, 1280] -> stride_w_out=1280 (stride along output dim), stride_w_in=1 (stride along input dim)
    # We access weight[n, k] = weight_ptr + n * 1280 + k * 1
    # For transposed access (k rows, n cols): weight_ptr + k * 1 + n * 1280
    i2h_w_stride_out = p_i2h_weight.stride(0)  # 1280
    i2h_w_stride_in = p_i2h_weight.stride(1)   # 1

    h2o_w_stride_out = p_h2o_weight.stride(0)  # 256
    h2o_w_stride_in = p_h2o_weight.stride(1)   # 1

    hidden_buf = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        x_t = x[t]  # [8, 1024]

        # i2h: cat(x_t, hidden) @ i2h_weight.T + i2h_bias, then tanh
        grid_i2h = (triton.cdiv(M, 8), triton.cdiv(N_i2h, 64))
        # Use a fixed grid based on autotune best guess, let autotune handle it
        grid_i2h = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_i2h, meta['BLOCK_N']))

        rnn_i2h_kernel[grid_i2h](
            x_t, hidden, p_i2h_weight, p_i2h_bias, hidden_buf,
            M, N_i2h, K_x, K_h, K_i2h,
            x_t.stride(0), x_t.stride(1),
            hidden.stride(0), hidden.stride(1),
            i2h_w_stride_out, i2h_w_stride_in,
        )

        # h2o: hidden_buf @ h2o_weight.T + h2o_bias
        grid_h2o = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_h2o, meta['BLOCK_N']))

        rnn_h2o_kernel[grid_h2o](
            hidden_buf, p_h2o_weight, p_h2o_bias, outputs[t],
            M, N_h2o, K_h2o,
            hidden_buf.stride(0), hidden_buf.stride(1),
            h2o_w_stride_out, h2o_w_stride_in,
            outputs[t].stride(0), outputs[t].stride(1),
        )

        hidden = hidden_buf
        hidden_buf = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    return outputs


def get_inputs():
    batch_size = 8
    input_size = 1024
    hidden_size = 256
    output_size = 128
    sequence_length = 256
    return [
        torch.randn(hidden_size, input_size + hidden_size, device='cuda'),  # p_i2h_weight [256, 1280]
        torch.randn(hidden_size, device='cuda'),                             # p_i2h_bias [256]
        torch.randn(output_size, hidden_size, device='cuda'),               # p_h2o_weight [128, 256]
        torch.randn(output_size, device='cuda'),                             # p_h2o_bias [128]
        torch.randn(sequence_length, batch_size, input_size, device='cuda'),# x [256, 8, 1024]
        torch.randn(batch_size, hidden_size, device='cuda'),                 # h0 [8, 256]
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
        p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0 = inputs

        # Reference implementation
        import torch.nn as nn
        hidden = h0.clone()
        ref_outputs = []
        for t in range(x.shape[0]):
            combined = torch.cat((x[t], hidden), dim=1)
            # linear: combined @ i2h_weight.T + i2h_bias
            h_new = torch.tanh(combined @ p_i2h_weight.T + p_i2h_bias)
            out = h_new @ p_h2o_weight.T + p_h2o_bias
            ref_outputs.append(out)
            hidden = h_new
        ref = torch.stack(ref_outputs, dim=0)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")