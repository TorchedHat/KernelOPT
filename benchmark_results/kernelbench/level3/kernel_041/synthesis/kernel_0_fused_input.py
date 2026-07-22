import torch
import triton
import triton.language as tl
import math

# GRU cell: given input x (batch, input_size) and hidden h (batch, hidden_size)
# Compute: gates_ih = x @ W_ih.T + b_ih  [batch, 3*hidden]
#          gates_hh = h @ W_hh.T + b_hh  [batch, 3*hidden]
# r = sigmoid(gates_ih[:, :H] + gates_hh[:, :H])
# z = sigmoid(gates_ih[:, H:2H] + gates_hh[:, H:2H])
# n = tanh(gates_ih[:, 2H:] + r * gates_hh[:, 2H:])
# h_new = (1 - z) * n + z * h

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['BATCH', 'INPUT_SIZE', 'HIDDEN_SIZE'],
)
@triton.jit
def gru_matmul_kernel(
    # Input matrix A: (M, K)
    A_ptr, stride_am, stride_ak,
    # Weight matrix B: (N, K) -> we compute A @ B.T
    B_ptr, stride_bn, stride_bk,
    # Output matrix C: (M, N)
    C_ptr, stride_cm, stride_cn,
    M, N, K,
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
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a, b, acc)
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BATCH': 16, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_BATCH': 16, 'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_BATCH': 32, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_BATCH': 32, 'BLOCK_H': 128}, num_warps=8),
    ],
    key=['BATCH', 'HIDDEN_SIZE'],
)
@triton.jit
def gru_cell_kernel(
    # gates_ih: (batch, 3*hidden) - input contribution
    gates_ih_ptr,
    # gates_hh: (batch, 3*hidden) - hidden contribution  
    gates_hh_ptr,
    # h_prev: (batch, hidden)
    h_prev_ptr,
    # h_new: (batch, hidden)
    h_new_ptr,
    BATCH, HIDDEN_SIZE,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    mask_b = offs_b < BATCH
    mask_h = offs_h < HIDDEN_SIZE
    mask = mask_b[:, None] & mask_h[None, :]
    
    H = HIDDEN_SIZE
    
    # Load gates_ih for r, z, n gates
    r_ih_ptrs = gates_ih_ptr + offs_b[:, None] * (3 * H) + offs_h[None, :]
    z_ih_ptrs = gates_ih_ptr + offs_b[:, None] * (3 * H) + H + offs_h[None, :]
    n_ih_ptrs = gates_ih_ptr + offs_b[:, None] * (3 * H) + 2 * H + offs_h[None, :]
    
    r_ih = tl.load(r_ih_ptrs, mask=mask, other=0.0)
    z_ih = tl.load(z_ih_ptrs, mask=mask, other=0.0)
    n_ih = tl.load(n_ih_ptrs, mask=mask, other=0.0)
    
    # Load gates_hh for r, z, n gates
    r_hh_ptrs = gates_hh_ptr + offs_b[:, None] * (3 * H) + offs_h[None, :]
    z_hh_ptrs = gates_hh_ptr + offs_b[:, None] * (3 * H) + H + offs_h[None, :]
    n_hh_ptrs = gates_hh_ptr + offs_b[:, None] * (3 * H) + 2 * H + offs_h[None, :]
    
    r_hh = tl.load(r_hh_ptrs, mask=mask, other=0.0)
    z_hh = tl.load(z_hh_ptrs, mask=mask, other=0.0)
    n_hh = tl.load(n_hh_ptrs, mask=mask, other=0.0)
    
    # Load h_prev
    h_prev_ptrs = h_prev_ptr + offs_b[:, None] * H + offs_h[None, :]
    h_prev = tl.load(h_prev_ptrs, mask=mask, other=0.0)
    
    # Compute gates
    r = tl.sigmoid(r_ih + r_hh)
    z = tl.sigmoid(z_ih + z_hh)
    n = tl.math.tanh(n_ih + r * n_hh)
    
    # Compute new hidden state
    h_new = (1.0 - z) * n + z * h_prev
    
    # Store h_new
    h_new_ptrs = h_new_ptr + offs_b[:, None] * H + offs_h[None, :]
    tl.store(h_new_ptrs, h_new, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_bias_kernel(
    out_ptr,
    bias_ptr,
    stride_row,  # stride between rows (= num_cols)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    col_offs = offs % num_cols
    
    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out + bias, mask=mask)


def gru_matmul(A, B):
    """Compute A @ B.T using Triton kernel"""
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    gru_matmul_kernel[grid](
        A, A.stride(0), A.stride(1),
        B, B.stride(0), B.stride(1),
        C, C.stride(0), C.stride(1),
        M, N, K,
        BATCH=M, INPUT_SIZE=K, HIDDEN_SIZE=N,
    )
    return C


def add_bias(out, bias):
    """Add bias to each row of out in-place"""
    M, N = out.shape
    total = M * N
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    add_bias_kernel[grid](
        out, bias,
        N=total,
        stride_row=N,
        num_cols=N,
    )
    return out


def gru_cell(gates_ih, gates_hh, h_prev):
    """
    returns h_new: (batch, hidden)
    """
    batch, hidden3 = gates_ih.shape
    hidden = hidden3 // 3
    
    h_new = torch.empty((batch, hidden), device=h_prev.device, dtype=torch.float32)
    
    grid = lambda meta: (
        triton.cdiv(batch, meta['BLOCK_BATCH']),
        triton.cdiv(hidden, meta['BLOCK_H']),
    )
    
    gru_cell_kernel[grid](
        gates_ih, gates_hh, h_prev, h_new,
        BATCH=batch, HIDDEN_SIZE=hidden,
    )
    return h_new


def run_gru_layer(x, h0, weight_ih, weight_hh, bias_ih, bias_hh, reverse=False):
    """
    returns: output (seq_len, batch, hidden_size), h_n (batch, hidden_size)
    """
    seq_len, batch, input_size = x.shape
    hidden_size = h0.shape[1]
    
    outputs = []
    h = h0.clone()
    
    time_steps = range(seq_len) if not reverse else range(seq_len - 1, -1, -1)
    
    for t in time_steps:
        x_t = x[t]  # (batch, input_size)
        
        # Compute input gates: x_t @ weight_ih.T + bias_ih
        gates_ih = gru_matmul(x_t, weight_ih)  # (batch, 3*hidden)
        add_bias(gates_ih, bias_ih)
        
        # Compute hidden gates: h @ weight_hh.T + bias_hh
        gates_hh = gru_matmul(h, weight_hh)  # (batch, 3*hidden)
        add_bias(gates_hh, bias_hh)
        
        # Compute new hidden state
        h = gru_cell(gates_ih, gates_hh, h)
        outputs.append(h)
    
    if reverse:
        outputs = outputs[::-1]
    
    output = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_size)
    return output, h


def kernel_function(
    p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0,
    p_gru_weight_ih_l0_reverse, p_gru_weight_hh_l0_reverse, p_gru_bias_ih_l0_reverse, p_gru_bias_hh_l0_reverse,
    p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1,
    p_gru_weight_ih_l1_reverse, p_gru_weight_hh_l1_reverse, p_gru_bias_ih_l1_reverse, p_gru_bias_hh_l1_reverse,
    p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2,
    p_gru_weight_ih_l2_reverse, p_gru_weight_hh_l2_reverse, p_gru_bias_ih_l2_reverse, p_gru_bias_hh_l2_reverse,
    p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3,
    p_gru_weight_ih_l3_reverse, p_gru_weight_hh_l3_reverse, p_gru_bias_ih_l3_reverse, p_gru_bias_hh_l3_reverse,
    p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4,
    p_gru_weight_ih_l4_reverse, p_gru_weight_hh_l4_reverse, p_gru_bias_ih_l4_reverse, p_gru_bias_hh_l4_reverse,
    p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5,
    p_gru_weight_ih_l5_reverse, p_gru_weight_hh_l5_reverse, p_gru_bias_ih_l5_reverse, p_gru_bias_hh_l5_reverse,
):
    num_layers = 6
    hidden_size = 256
    
    # Layer parameters
    layer_params = [
        (p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0,
         p_gru_weight_ih_l0_reverse, p_gru_weight_hh_l0_reverse, p_gru_bias_ih_l0_reverse, p_gru_bias_hh_l0_reverse),
        (p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1,
         p_gru_weight_ih_l1_reverse, p_gru_weight_hh_l1_reverse, p_gru_bias_ih_l1_reverse, p_gru_bias_hh_l1_reverse),
        (p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2,
         p_gru_weight_ih_l2_reverse, p_gru_weight_hh_l2_reverse, p_gru_bias_ih_l2_reverse, p_gru_bias_hh_l2_reverse),
        (p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3,
         p_gru_weight_ih_l3_reverse, p_gru_weight_hh_l3_reverse, p_gru_bias_ih_l3_reverse, p_gru_bias_hh_l3_reverse),
        (p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4,
         p_gru_weight_ih_l4_reverse, p_gru_weight_hh_l4_reverse, p_gru_bias_ih_l4_reverse, p_gru_bias_hh_l4_reverse),
        (p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5,
         p_gru_weight_ih_l5_reverse, p_gru_weight_hh_l5_reverse, p_gru_bias_ih_l5_reverse, p_gru_bias_hh_l5_reverse),
    ]
    
    current_input = x
    all_h_n = []
    
    for layer_idx in range(num_layers):
        wih, whh, bih, bhh, wih_r, whh_r, bih_r, bhh_r = layer_params[layer_idx]
        
        h_fwd = h0[layer_idx * 2]      # (batch, hidden)
        h_bwd = h0[layer_idx * 2 + 1]  # (batch, hidden)
        
        # Forward direction
        out_fwd, h_n_fwd = run_gru_layer(
            current_input, h_fwd, wih, whh, bih, bhh, reverse=False
        )
        
        # Backward direction
        out_bwd, h_n_bwd = run_gru_layer(
            current_input, h_bwd, wih_r, whh_r, bih_r, bhh_r, reverse=True
        )
        
        # Concatenate forward and backward outputs
        current_input = torch.cat([out_fwd, out_bwd], dim=2)  # (seq, batch, 2*hidden)
        
        all_h_n.append(h_n_fwd)
        all_h_n.append(h_n_bwd)
    
    output = current_input  # (seq_len, batch, 2*hidden_size)
    h_n = torch.stack(all_h_n, dim=0)  # (num_layers*2, batch, hidden_size)
    
    return output, h_n


def get_inputs():
    batch_size = 10
    seq_len = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6
    
    # Layer 0 weights
    p_gru_weight_ih_l0 = torch.randn(768, 128)
    p_gru_weight_hh_l0 = torch.randn(768, 256)
    p_gru_bias_ih_l0 = torch.randn(768)
    p_gru_bias_hh_l0 = torch.randn(768)
    p_gru_weight_ih_l0_reverse = torch.randn(768, 128)
    p_gru_weight_hh_l0_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l0_reverse = torch.randn(768)
    p_gru_bias_hh_l0_reverse = torch.randn(768)
    
    # Layer 1 weights
    p_gru_weight_ih_l1 = torch.randn(768, 512)
    p_gru_weight_hh_l1 = torch.randn(768, 256)
    p_gru_bias_ih_l1 = torch.randn(768)
    p_gru_bias_hh_l1 = torch.randn(768)
    p_gru_weight_ih_l1_reverse = torch.randn(768, 512)
    p_gru_weight_hh_l1_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l1_reverse = torch.randn(768)
    p_gru_bias_hh_l1_reverse = torch.randn(768)
    
    # Layer 2 weights
    p_gru_weight_ih_l2 = torch.randn(768, 512)
    p_gru_weight_hh_l2 = torch.randn(768, 256)
    p_gru_bias_ih_l2 = torch.randn(768)
    p_gru_bias_hh_l2 = torch.randn(768)
    p_gru_weight_ih_l2_reverse = torch.randn(768, 512)
    p_gru_weight_hh_l2_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l2_reverse = torch.randn(768)
    p_gru_bias_hh_l2_reverse = torch.randn(768)
    
    # Layer 3 weights
    p_gru_weight_ih_l3 = torch.randn(768, 512)
    p_gru_weight_hh_l3 = torch.randn(768, 256)
    p_gru_bias_ih_l3 = torch.randn(768)
    p_gru_bias_hh_l3 = torch.randn(768)
    p_gru_weight_ih_l3_reverse = torch.randn(768, 512)
    p_gru_weight_hh_l3_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l3_reverse = torch.randn(768)
    p_gru_bias_hh_l3_reverse = torch.randn(768)
    
    # Layer 4 weights
    p_gru_weight_ih_l4 = torch.randn(768, 512)
    p_gru_weight_hh_l4 = torch.randn(768, 256)
    p_gru_bias_ih_l4 = torch.randn(768)
    p_gru_bias_hh_l4 = torch.randn(768)
    p_gru_weight_ih_l4_reverse = torch.randn(768, 512)
    p_gru_weight_hh_l4_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l4_reverse = torch.randn(768)
    p_gru_bias_hh_l4_reverse = torch.randn(768)
    
    # Layer 5 weights
    p_gru_weight_ih_l5 = torch.randn(768, 512)
    p_gru_weight_hh_l5 = torch.randn(768, 256)
    p_gru_bias_ih_l5 = torch.randn(768)
    p_gru_bias_hh_l5 = torch.randn(768)
    p_gru_weight_ih_l5_reverse = torch.randn(768, 512)
    p_gru_weight_hh_l5_reverse = torch.randn(768, 256)
    p_gru_bias_ih_l5_reverse = torch.randn(768)
    p_gru_bias_hh_l5_reverse = torch.randn(768)
    
    x = torch.rand(seq_len, batch_size, input_size)
    h0 = torch.rand(num_layers * 2, batch_size, hidden_size)
    
    return [
        p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0,
        p_gru_weight_ih_l0_reverse, p_gru_weight_hh_l0_reverse, p_gru_bias_ih_l0_reverse, p_gru_bias_hh_l0_reverse,
        p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1,
        p_gru_weight_ih_l1_reverse, p_gru_weight_hh_l1_reverse, p_gru_bias_ih_l1_reverse, p_gru_bias_hh_l1_reverse,
        p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2,
        p_gru_weight_ih_l2_reverse, p_gru_weight_hh_l2_reverse, p_gru_bias_ih_l2_reverse, p_gru_bias_hh_l2_reverse,
        p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3,
        p_gru_weight_ih_l3_reverse, p_gru_weight_hh_l3_reverse, p_gru_bias_ih_l3_reverse, p_gru_bias_hh_l3_reverse,
        p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4,
        p_gru_weight_ih_l4_reverse, p_gru_weight_hh_l4_reverse, p_gru_bias_ih_l4_reverse, p_gru_bias_hh_l4_reverse,
        p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5,
        p_gru_weight_ih_l5_reverse, p_gru_weight_hh_l5_reverse, p_gru_bias_ih_l5_reverse, p_gru_bias_hh_l5_reverse,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]
    
    output, h_n = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        batch_size = 10
        seq_len = 512
        input_size = 128
        hidden_size = 256
        num_layers = 6
        
        # Build reference GRU
        gru_ref = nn.GRU(
            input_size, hidden_size, num_layers,
            bias=True, batch_first=False, dropout=0, bidirectional=True
        ).cuda()
        
        # Set weights from inputs
        cpu_inputs = inputs
        
        gru_ref.weight_ih_l0.data.copy_(cpu_inputs[0].cuda())
        gru_ref.weight_hh_l0.data.copy_(cpu_inputs[1].cuda())
        gru_ref.bias_ih_l0.data.copy_(cpu_inputs[2].cuda())
        gru_ref.bias_hh_l0.data.copy_(cpu_inputs[3].cuda())
        gru_ref.weight_ih_l0_reverse.data.copy_(cpu_inputs[4].cuda())
        gru_ref.weight_hh_l0_reverse.data.copy_(cpu_inputs[5].cuda())
        gru_ref.bias_ih_l0_reverse.data.copy_(cpu_inputs[6].cuda())
        gru_ref.bias_hh_l0_reverse.data.copy_(cpu_inputs[7].cuda())
        
        gru_ref.weight_ih_l1.data.copy_(cpu