import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['batch_size', 'hidden_size'],
)
@triton.jit
def gru_cell_kernel(
    # ih_gates: precomputed x @ W_ih.T + b_ih for this timestep, shape [batch, 3*H]
    ih_ptr,
    # hh_bias: b_hh, shape [3*H]
    hh_bias_ptr,
    # W_hh: shape [3*H, H] (stored as [H_out, H_in])
    W_hh_ptr,
    # h_prev: shape [batch, H]
    h_prev_ptr,
    # h_new: output shape [batch, H]
    h_new_ptr,
    # hh_out: W_hh @ h_prev + b_hh, shape [batch, 3*H] - precomputed
    hh_ptr,
    batch_size, hidden_size,
    stride_ih_b, stride_ih_g,
    stride_hh_b, stride_hh_g,
    stride_hp_b, stride_hp_h,
    stride_hn_b, stride_hn_h,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element, iterating over hidden units
    bid = tl.program_id(0)
    if bid >= batch_size:
        return
    
    pid = tl.program_id(1)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    
    # Load ih gates for this batch: r_ih, z_ih, n_ih
    r_ih = tl.load(ih_ptr + bid * stride_ih_b + offs * stride_ih_g, mask=mask, other=0.0)
    z_ih = tl.load(ih_ptr + bid * stride_ih_b + (offs + hidden_size) * stride_ih_g, mask=mask, other=0.0)
    n_ih = tl.load(ih_ptr + bid * stride_ih_b + (offs + 2 * hidden_size) * stride_ih_g, mask=mask, other=0.0)
    
    # Load hh gates for this batch: r_hh, z_hh, n_hh
    r_hh = tl.load(hh_ptr + bid * stride_hh_b + offs * stride_hh_g, mask=mask, other=0.0)
    z_hh = tl.load(hh_ptr + bid * stride_hh_b + (offs + hidden_size) * stride_hh_g, mask=mask, other=0.0)
    n_hh = tl.load(hh_ptr + bid * stride_hh_b + (offs + 2 * hidden_size) * stride_hh_g, mask=mask, other=0.0)
    
    # Load h_prev
    h_prev = tl.load(h_prev_ptr + bid * stride_hp_b + offs * stride_hp_h, mask=mask, other=0.0)
    
    # GRU gates
    # r = sigmoid(r_ih + r_hh)
    r = tl.sigmoid(r_ih + r_hh)
    # z = sigmoid(z_ih + z_hh)
    z = tl.sigmoid(z_ih + z_hh)
    # n = tanh(n_ih + r * n_hh)
    n = tl.extra.libdevice.tanh(n_ih + r * n_hh)
    # h_new = (1 - z) * n + z * h_prev
    h_new = (1.0 - z) * n + z * h_prev
    
    tl.store(h_new_ptr + bid * stride_hn_b + offs * stride_hn_h, h_new, mask=mask)


def triton_matmul(A, B, bias=None):
    """Compute A @ B + bias where A is [M, K], B is [K, N]"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    if bias is not None:
        matmul_bias_kernel[grid](
            A, B, C, bias,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
    else:
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
    return C


def gru_layer_forward(x_seq, h0, W_ih, W_hh, b_ih, b_hh, reverse=False):
    """
    returns: output [seq_len, batch, H], h_n [batch, H]
    """
    seq_len, batch_size, input_size = x_seq.shape
    hidden_size = h0.shape[1]
    
    # Precompute all input projections at once: [seq_len*batch, 3*H]
    x_flat = x_seq.reshape(seq_len * batch_size, input_size)
    # W_ih is [3*H, input_size], we need x @ W_ih.T = [M, input_size] @ [input_size, 3*H]
    W_ih_T = W_ih.t().contiguous()  # [input_size, 3*H]
    ih_all = triton_matmul(x_flat, W_ih_T, b_ih)  # [seq_len*batch, 3*H]
    ih_all = ih_all.reshape(seq_len, batch_size, 3 * hidden_size)
    
    # W_hh is [3*H, H], we need h @ W_hh.T = [batch, H] @ [H, 3*H]
    W_hh_T = W_hh.t().contiguous()  # [H, 3*H]
    
    h = h0.clone()
    outputs = []
    
    time_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    
    for t in time_range:
        ih_t = ih_all[t]  # [batch, 3*H]
        # Compute hh: h @ W_hh.T + b_hh
        hh_t = triton_matmul(h, W_hh_T, b_hh)  # [batch, 3*H]
        
        # Apply GRU cell using Triton kernel
        h_new = torch.empty_like(h)
        
        grid = lambda meta: (batch_size, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
        gru_cell_kernel[grid](
            ih_t, None, None,
            h, h_new, hh_t,
            batch_size, hidden_size,
            ih_t.stride(0), ih_t.stride(1),
            hh_t.stride(0), hh_t.stride(1),
            h.stride(0), h.stride(1),
            h_new.stride(0), h_new.stride(1),
        )
        h = h_new
        outputs.append(h)
    
    if reverse:
        outputs = outputs[::-1]
    
    output = torch.stack(outputs, dim=0)  # [seq_len, batch, H]
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
    x, h0,
):
    seq_len, batch_size, input_size = x.shape
    num_layers = 6
    hidden_size = 256
    
    layers_fwd = [
        (p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0),
        (p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1),
        (p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2),
        (p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3),
        (p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4),
        (p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5),
    ]
    layers_rev = [
        (p_gru_weight_ih_l0_reverse, p_gru_weight_hh_l0_reverse, p_gru_bias_ih_l0_reverse, p_gru_bias_hh_l0_reverse),
        (p_gru_weight_ih_l1_reverse, p_gru_weight_hh_l1_reverse, p_gru_bias_ih_l1_reverse, p_gru_bias_hh_l1_reverse),
        (p_gru_weight_ih_l2_reverse, p_gru_weight_hh_l2_reverse, p_gru_bias_ih_l2_reverse, p_gru_bias_hh_l2_reverse),
        (p_gru_weight_ih_l3_reverse, p_gru_weight_hh_l3_reverse, p_gru_bias_ih_l3_reverse, p_gru_bias_hh_l3_reverse),
        (p_gru_weight_ih_l4_reverse, p_gru_weight_hh_l4_reverse, p_gru_bias_ih_l4_reverse, p_gru_bias_hh_l4_reverse),
        (p_gru_weight_ih_l5_reverse, p_gru_weight_hh_l5_reverse, p_gru_bias_ih_l5_reverse, p_gru_bias_hh_l5_reverse),
    ]
    
    current_input = x
    all_h_n = []
    
    for layer_idx in range(num_layers):
        # h0 layout: [num_layers*2, batch, hidden] with fwd at 2*layer, rev at 2*layer+1
        h0_fwd = h0[2 * layer_idx]      # [batch, hidden]
        h0_rev = h0[2 * layer_idx + 1]  # [batch, hidden]
        
        W_ih_f, W_hh_f, b_ih_f, b_hh_f = layers_fwd[layer_idx]
        W_ih_r, W_hh_r, b_ih_r, b_hh_r = layers_rev[layer_idx]
        
        out_fwd, h_n_fwd = gru_layer_forward(current_input, h0_fwd, W_ih_f, W_hh_f, b_ih_f, b_hh_f, reverse=False)
        out_rev, h_n_rev = gru_layer_forward(current_input, h0_rev, W_ih_r, W_hh_r, b_ih_r, b_hh_r, reverse=True)
        
        # Concatenate forward and reverse outputs: [seq_len, batch, 2*H]
        current_input = torch.cat([out_fwd, out_rev], dim=2)
        
        all_h_n.append(h_n_fwd)   # [batch, H]
        all_h_n.append(h_n_rev)   # [batch, H]
    
    # Stack h_n: [num_layers*2, batch, H]
    h_n = torch.stack(all_h_n, dim=0)
    
    # output is current_input: [seq_len, batch, 2*H] = [512, 10, 512]
    output = current_input
    
    return h_n  # The model returns h_n


def get_inputs():
    batch_size = 10
    seq_len = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6
    
    weights = []
    # Layer 0: W_ih [768, 128], W_hh [768, 256]
    weights.extend([
        torch.randn(768, 128), torch.randn(768, 256), torch.randn(768), torch.randn(768),
        torch.randn(768, 128), torch.randn(768, 256), torch.randn(768), torch.randn(768),
    ])
    # Layers 1-5: W_ih [768, 512], W_hh [768, 256]
    for _ in range(5):
        weights.extend([
            torch.randn(768, 512), torch.randn(768, 256), torch.randn(768), torch.randn(768),
            torch.randn(768, 512), torch.randn(768, 256), torch.randn(768), torch.randn(768),
        ])
    
    x = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(num_layers * 2, batch_size, hidden_size)
    
    return [w.cuda() for w in weights] + [x.cuda(), h0.cuda()]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        # Build reference GRU
        batch_size = 10
        seq_len = 512
        input_size = 128
        hidden_size = 256
        num_layers = 6
        
        gru = nn.GRU(input_size, hidden_size, num_layers, bias=True, 
                     batch_first=False, dropout=0, bidirectional=True).cuda()
        
        # Set weights from inputs
        x = inputs[-2]
        h0 = inputs[-1]
        
        layer_params = [
            (inputs[0], inputs[1], inputs[2], inputs[3]),   # l0 fwd
            (inputs[4], inputs[5], inputs[6], inputs[7]),   # l0 rev
            (inputs[8], inputs[9], inputs[10], inputs[11]), # l1 fwd
            (inputs[12], inputs[13], inputs[14], inputs[15]), # l1 rev
            (inputs[16], inputs[17], inputs[18], inputs[19]), # l2 fwd
            (inputs[20], inputs[21], inputs[22], inputs[23]), # l2 rev
            (inputs[24], inputs[25], inputs[26], inputs[27]), # l3 fwd
            (inputs[28], inputs[29], inputs[30], inputs[31]), # l3 rev
            (inputs[32], inputs[33], inputs[34], inputs[35]), # l4 fwd
            (inputs[36], inputs[37], inputs[38], inputs[39]), # l4 rev
            (inputs[40], inputs[41], inputs[42], inputs[43]), # l5 fwd
            (inputs[44], inputs[45], inputs[46], inputs[47]), # l5 rev
        ]
        
        for layer_idx in range(num_layers):
            fwd_idx = layer_idx * 2
            rev_idx = layer_idx * 2 + 1
            W_ih_f, W_hh_f, b_ih_f, b_hh_f = layer_params[fwd_idx]
            W_ih_r, W_hh_r, b_ih_r, b_hh_r = layer_params[rev_idx]
            
            getattr(gru, f'weight_ih_l{layer_idx}').data.copy_(W_ih_f)
            getattr(gru, f'weight_hh_l{layer_idx}').data.copy_(W_hh_f)
            getattr(gru, f'bias_ih_l{layer_idx}').data.copy_(b_ih_f)
            getattr(gru, f'bias_hh_l{layer_idx}').data.copy_(b_hh_f)
            getattr(gru, f'weight_ih_l{layer_idx}_reverse').data.copy_(W_ih_r)
            getattr(gru, f'weight_hh_l{layer_idx}_reverse').data.copy_(W_hh_r)
            getattr(gru, f'bias_ih_l{layer_idx}_reverse').data.copy_(b_ih_r)
            getattr(gru, f'bias_hh_l{layer_idx}_reverse').data.copy_(b_hh_r)
        
        with torch.no_grad():
            _, ref_h_n = gru(x, h0)
        
        print(f"Output shape: {output.shape}, ref shape: {ref_h_n.shape}")
        print(f"Max diff: {(output - ref_h_n).abs().max().item():.6f}")
        assert torch.allclose(output, ref_h_n, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref_h_n).abs().max().item()):.4e}"
        print("Kernel executed successfully.")