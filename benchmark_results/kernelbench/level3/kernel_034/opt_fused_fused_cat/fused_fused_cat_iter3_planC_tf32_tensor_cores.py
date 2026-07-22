import torch
import triton
import triton.language as tl

# RNN step kernel: computes hidden = tanh(combined @ W_i2h.T + b_i2h)
# combined shape: [batch, input+hidden] = [8, 1280]
# W_i2h: [hidden, input+hidden] = [256, 1280]
# b_i2h: [hidden] = [256]
# output hidden: [batch, hidden] = [8, 256]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['HIDDEN_SIZE', 'COMBINED_SIZE'],
)
@triton.jit
def rnn_i2h_kernel(
    # combined input [batch, combined_size]
    x_ptr, hidden_ptr,
    # weights and bias
    w_i2h_ptr, b_i2h_ptr,
    # output hidden [batch, hidden_size]
    out_hidden_ptr,
    # dims
    BATCH: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    COMBINED_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # pid_m: batch tile, pid_n: hidden tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over combined dimension in BLOCK_K chunks
    for k in range(0, tl.cdiv(COMBINED_SIZE, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load from combined = [x[t] | hidden], shape [batch, combined_size]
        # First INPUT_SIZE cols come from x_ptr, next HIDDEN_SIZE from hidden_ptr
        k_vals = offs_k  # [BLOCK_K]

        # Mask for valid k
        k_mask = offs_k < COMBINED_SIZE

        # Build combined tile [BLOCK_M, BLOCK_K]
        # For each k: if k < INPUT_SIZE, load from x_ptr, else from hidden_ptr
        m_mask = offs_m < BATCH

        # Load x part
        x_k_mask = k_vals < INPUT_SIZE  # [BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * INPUT_SIZE + k_vals[None, :]
        x_tile = tl.load(x_ptrs,
                         mask=m_mask[:, None] & x_k_mask[None, :] & k_mask[None, :],
                         other=0.0)

        # Load hidden part
        h_k = k_vals - INPUT_SIZE  # offset into hidden
        h_k_mask = (k_vals >= INPUT_SIZE) & (k_vals < COMBINED_SIZE)
        h_ptrs = hidden_ptr + offs_m[:, None] * HIDDEN_SIZE + h_k[None, :]
        h_tile = tl.load(h_ptrs,
                         mask=m_mask[:, None] & h_k_mask[None, :],
                         other=0.0)

        combined_tile = tl.where(x_k_mask[None, :], x_tile, h_tile)  # [BLOCK_M, BLOCK_K]

        # Load weight tile [BLOCK_K, BLOCK_N] from w_i2h [HIDDEN_SIZE, COMBINED_SIZE]
        # w_i2h[n, k] -> we need w^T so we load w[offs_n, offs_k]
        n_mask = offs_n < HIDDEN_SIZE
        w_ptrs = w_i2h_ptr + offs_n[None, :] * COMBINED_SIZE + offs_k[:, None]
        w_tile = tl.load(w_ptrs,
                         mask=n_mask[None, :] & k_mask[:, None],
                         other=0.0)  # [BLOCK_K, BLOCK_N]

        acc = tl.dot(combined_tile, w_tile, acc=acc)

    # Add bias and apply tanh
    offs_n_final = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n_final < HIDDEN_SIZE
    bias = tl.load(b_i2h_ptr + offs_n_final, mask=n_mask, other=0.0)  # [BLOCK_N]

    acc = acc + bias[None, :]

    # Apply tanh
    acc = tl.extra.libdevice.tanh(acc)

    # Store result
    m_mask = offs_m < BATCH
    out_ptrs = out_hidden_ptr + offs_m[:, None] * HIDDEN_SIZE + offs_n_final[None, :]
    tl.store(out_ptrs, acc.to(tl.float32),
             mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['HIDDEN_SIZE', 'OUTPUT_SIZE'],
)
@triton.jit

def rnn_h2o_kernel(
    hidden_ptr,
    w_h2o_ptr, b_h2o_ptr,
    out_ptr,
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
    m_mask = offs_m < BATCH
    n_mask = offs_n < OUTPUT_SIZE

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < HIDDEN_SIZE

        # Load hidden [BLOCK_M, BLOCK_K]
        h_ptrs = hidden_ptr + offs_m[:, None] * HIDDEN_SIZE + offs_k[None, :]
        h_tile = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load weight [BLOCK_K, BLOCK_N] from w_h2o [OUTPUT_SIZE, HIDDEN_SIZE]
        w_ptrs = w_h2o_ptr + offs_n[None, :] * HIDDEN_SIZE + offs_k[:, None]
        w_tile = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc = tl.dot(h_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(b_h2o_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * OUTPUT_SIZE + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0):
    """
    returns: [256, 8, 128]
    """
    seq_len, batch_size, input_size = x.shape
    hidden_size = p_i2h_weight.shape[0]   # 256
    output_size = p_h2o_weight.shape[0]   # 128
    combined_size = input_size + hidden_size  # 1280

    # Make x contiguous and reshape for easy access
    x_cont = x.contiguous()  # [256, 8, 1024]

    hidden = h0.contiguous()  # [8, 256]
    w_i2h = p_i2h_weight.contiguous()  # [256, 1280]
    b_i2h = p_i2h_bias.contiguous()    # [256]
    w_h2o = p_h2o_weight.contiguous()  # [128, 256]
    b_h2o = p_h2o_bias.contiguous()    # [128]

    outputs = torch.empty(seq_len, batch_size, output_size, device=x.device, dtype=x.dtype)

    # Grid for i2h: [ceil(batch/BLOCK_M), ceil(hidden/BLOCK_N)]
    # Grid for h2o: [ceil(batch/BLOCK_M), ceil(output/BLOCK_N)]

    new_hidden = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        x_t = x_cont[t]  # [8, 1024]

        # i2h kernel
        def grid_i2h(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_M']),
                triton.cdiv(hidden_size, meta['BLOCK_N']),
            )

        rnn_i2h_kernel[grid_i2h](
            x_t, hidden,
            w_i2h, b_i2h,
            new_hidden,
            BATCH=batch_size,
            INPUT_SIZE=input_size,
            HIDDEN_SIZE=hidden_size,
            COMBINED_SIZE=combined_size,
        )

        hidden = new_hidden.clone()

        # h2o kernel
        def grid_h2o(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_M']),
                triton.cdiv(output_size, meta['BLOCK_N']),
            )

        rnn_h2o_kernel[grid_h2o](
            hidden,
            w_h2o, b_h2o,
            outputs[t],
            BATCH=batch_size,
            HIDDEN_SIZE=hidden_size,
            OUTPUT_SIZE=output_size,
        )

    return outputs


def get_inputs():
    batch_size = 8
    input_size = 1024
    hidden_size = 256
    output_size = 128
    sequence_length = 256
    return [
        torch.rand(hidden_size, input_size + hidden_size, device='cuda'),  # i2h weight [256, 1280]
        torch.rand(hidden_size, device='cuda'),                             # i2h bias [256]
        torch.rand(output_size, hidden_size, device='cuda'),               # h2o weight [128, 256]
        torch.rand(output_size, device='cuda'),                            # h2o bias [128]
        torch.rand(sequence_length, batch_size, input_size, device='cuda'),# x [256, 8, 1024]
        torch.rand(batch_size, hidden_size, device='cuda'),                # h0 [8, 256]
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
        # Reference implementation
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.i2h = nn.Linear(1024 + 256, 256)
                self.h2o = nn.Linear(256, 128)
                self.tanh = nn.Tanh()

            def forward(self, x, h0):
                seq_len, batch_size, _ = x.size()
                hidden = h0
                outputs = []
                for t in range(seq_len):
                    combined = torch.cat((x[t], hidden), dim=1)
                    hidden = self.tanh(self.i2h(combined))
                    out = self.h2o(hidden)
                    outputs.append(out)
                return torch.stack(outputs, dim=0)

        model = Model().cuda()
        with torch.no_grad():
            model.i2h.weight.data.copy_(p_i2h_weight)
            model.i2h.bias.data.copy_(p_i2h_bias)
            model.h2o.weight.data.copy_(p_h2o_weight)
            model.h2o.bias.data.copy_(p_h2o_bias)
            ref = model(x, h0)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")