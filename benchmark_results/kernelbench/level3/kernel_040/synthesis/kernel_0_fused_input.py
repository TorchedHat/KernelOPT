import torch
import triton
import triton.language as tl

- r_t = sigmoid(x_t @ W_ir.T + b_ir + h_{t-1} @ W_hr.T + b_hr)
- z_t = sigmoid(x_t @ W_iz.T + b_iz + h_{t-1} @ W_hz.T + b_hz)  
- n_t = tanh(x_t @ W_in.T + b_in + r_t * (h_{t-1} @ W_hn.T + b_hn))
- h_t = (1 - z_t) * n_t + z_t * h_{t-1}



- For layer 0: x is (batch=10, input=128), W_ih is (768, 128), W_hh is (768, 256)
- For layers 1-5: x is (batch=10, hidden=256), W_ih is (768, 256), W_hh is (768, 256)
- Output gates: (batch=10, 768) -> split into 3 * (batch=10, 256)


import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl

# GRU step kernel: computes one time step for one layer
# Fuses: x @ W_ih.T + b_ih + h @ W_hh.T + b_hh -> gates -> new_h
# M = batch_size, K_x = input_size, K_h = hidden_size, N = 3*hidden_size

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K_x', 'K_h'],
)
@triton.jit
def gru_step_kernel(
    # Input x: (M, K_x)
    x_ptr, stride_xm, stride_xk,
    # Weight ih: (N, K_x) - stored as (3*H, K_x)
    wih_ptr, stride_wih_n, stride_wih_k,
    # Bias ih: (N,)
    bih_ptr,
    # Hidden h: (M, K_h)
    h_ptr, stride_hm, stride_hk,
    # Weight hh: (N, K_h) - stored as (3*H, K_h)
    whh_ptr, stride_whh_n, stride_whh_k,
    # Bias hh: (N,)
    bhh_ptr,
    # Output new_h: (M, H)
    new_h_ptr, stride_nhm, stride_nhk,
    # Dimensions
    M, N, K_x, K_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # N = 3 * H, H = hidden_size
    H = N // 3
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Compute x @ W_ih.T: (M, K_x) @ (K_x, N) -> (M, N)
    # W_ih is (N, K_x), so W_ih.T is (K_x, N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # x @ W_ih.T
    for k in range(0, K_x, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_x)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W_ih: (N, K_x), we want W_ih[offs_n, offs_k] -> tile is (BLOCK_N, BLOCK_K)
        # For x @ W_ih.T: need W_ih.T[offs_k, offs_n] = W_ih[offs_n, offs_k]
        wih_ptrs = wih_ptr + offs_n[None, :] * stride_wih_n + offs_k[:, None] * stride_wih_k
        wih_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K_x)
        wih_tile = tl.load(wih_ptrs, mask=wih_mask, other=0.0)
        # wih_tile shape: (BLOCK_K, BLOCK_N)
        acc = tl.dot(x_tile, wih_tile, acc=acc)
    
    # h @ W_hh.T
    for k in range(0, K_h, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_h)
        h_tile = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        whh_ptrs = whh_ptr + offs_n[None, :] * stride_whh_n + offs_k[:, None] * stride_whh_k
        whh_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K_h)
        whh_tile = tl.load(whh_ptrs, mask=whh_mask, other=0.0)
        acc = tl.dot(h_tile, whh_tile, acc=acc)
    
    # Add biases
    bih_vals = tl.load(bih_ptr + offs_n, mask=offs_n < N, other=0.0)
    bhh_vals = tl.load(bhh_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bih_vals[None, :] + bhh_vals[None, :]
    
    # Now acc has shape (BLOCK_M, BLOCK_N) = gates for this tile
    # But we need to handle the GRU gate computation carefully
    # The gates are: [r | z | n] each of size H
    # We need to know which gate each column belongs to
    # For r and z: sigmoid; for n: need r * h_hh_part
    # 
    # Problem: n gate = tanh(x_ih_n + r * h_hh_n)
    # We've computed x_ih_n + h_hh_n together, but need r * h_hh_n separately
    # 
    # We need to split the computation:
    # For r, z: acc_rz = x_ih_rz + h_hh_rz (already done)
    # For n: acc_n = x_ih_n + r * h_hh_n
    # 
    # This means we need to store h_hh_n separately
    # Let's restructure: compute x_ih separately from h_hh
    
    # Actually, let me use a different approach - store intermediate results
    # This kernel computes the full gate values but n gate needs special treatment
    # We'll handle this by computing h_hh_n separately
    
    # For simplicity in this kernel, let's just compute r, z gates from acc
    # and for n gate, we need to recompute with r applied
    # 
    # gate_r = sigmoid(acc[:, 0:H])
    # gate_z = sigmoid(acc[:, H:2H])  
    # gate_n = tanh(x_ih_n + r * h_hh_n)
    #
    # Since BLOCK_N may span multiple gate regions, we need to handle this carefully
    
    # Determine which gate region each column is in
    # gate 0: r (cols 0..H-1), gate 1: z (cols H..2H-1), gate 2: n (cols 2H..3H-1)
    gate_idx = offs_n // H  # 0, 1, or 2
    col_in_gate = offs_n % H  # position within gate
    
    valid_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # For r and z gates: apply sigmoid to acc
    # For n gate: need r * h_hh_n + x_ih_n
    # We already have x_ih_n + h_hh_n in acc for n gate region
    # But we need r * h_hh_n, not h_hh_n
    # 
    # This requires a two-pass approach or storing h_hh_n separately
    # Let's use a separate kernel for the gate activation part
    # For now, store the raw gate values and handle activation separately
    
    # Store raw gates to new_h temporarily - but we need output shape (M, H)
    # This approach won't work cleanly in one kernel for n gate
    # 
    # Better: use a separate kernel for gate activations
    # Store raw gates to a temp buffer, then apply activations
    
    # Actually, let's store to new_h_ptr which we'll use as temp storage
    # We'll need a separate activation kernel
    # For now store raw gates
    out_ptrs = new_h_ptr + offs_m[:, None] * stride_nhm + offs_n[None, :] * stride_nhk
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_mask)


# Separate kernel to compute x_ih contribution only
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    # A: (M, K)
    a_ptr, stride_am, stride_ak,
    # B: (N, K) - we compute A @ B.T
    b_ptr, stride_bn, stride_bk,
    # bias: (N,)
    bias_ptr,
    # C: (M, N)
    c_ptr, stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc=acc)
    
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :]
    
    out_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['M', 'H'],
)
@triton.jit
def gru_gate_kernel(
    # ih_gates: (M, 3H) - x @ W_ih.T + b_ih
    ih_ptr, stride_ihm, stride_ihn,
    # hh_gates: (M, 3H) - h @ W_hh.T + b_hh  
    hh_ptr, stride_hhm, stride_hhn,
    # h_prev: (M, H)
    h_ptr, stride_hm, stride_hk,
    # new_h: (M, H)
    new_h_ptr, stride_nhm, stride_nhk,
    M, H,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element) and BLOCK_SIZE hidden units
    pid = tl.program_id(0)
    row = pid // tl.cdiv(H, BLOCK_SIZE)
    col_block = pid % tl.cdiv(H, BLOCK_SIZE)
    
    offs_h = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_h = (row < M) & (offs_h < H)
    
    # Load r gate: ih[row, offs_h] + hh[row, offs_h]
    r_ih = tl.load(ih_ptr + row * stride_ihm + offs_h * stride_ihn, mask=mask_h, other=0.0)
    r_hh = tl.load(hh_ptr + row * stride_hhm + offs_h * stride_hhn, mask=mask_h, other=0.0)
    r = tl.sigmoid(r_ih + r_hh)
    
    # Load z gate: ih[row, H+offs_h] + hh[row, H+offs_h]
    z_ih = tl.load(ih_ptr + row * stride_ihm + (H + offs_h) * stride_ihn, mask=mask_h, other=0.0)
    z_hh = tl.load(hh_ptr + row * stride_hhm + (H + offs_h) * stride_hhn, mask=mask_h, other=0.0)
    z = tl.sigmoid(z_ih + z_hh)
    
    # Load n gate: ih[row, 2H+offs_h] + r * hh[row, 2H+offs_h]
    n_ih = tl.load(ih_ptr + row * stride_ihm + (2 * H + offs_h) * stride_ihn, mask=mask_h, other=0.0)
    n_hh = tl.load(hh_ptr + row * stride_hhm + (2 * H + offs_h) * stride_hhn, mask=mask_h, other=0.0)
    n = tl.libdevice.tanh(n_ih + r * n_hh)
    
    # Load h_prev
    h_prev = tl.load(h_ptr + row * stride_hm + offs_h * stride_hk, mask=mask_h, other=0.0)
    
    # new_h = (1 - z) * n + z * h_prev
    new_h = (1.0 - z) * n + z * h_prev
    
    tl.store(new_h_ptr + row * stride_nhm + offs_h * stride_nhk, new_h, mask=mask_h)


def gru_step(x, h, weight_ih, weight_hh, bias_ih, bias_hh):
    """
    """
    M = x.shape[0]
    H = h.shape[1]
    N = 3 * H
    K_x = x.shape[1]
    K_h = H
    
    # Compute ih_gates = x @ weight_ih.T + bias_ih: (M, 3H)
    ih_gates = torch.empty(M, N, device=x.device, dtype=torch.float32)
    grid_ih = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_bias_kernel[grid_ih](
        x, x.stride(0), x.stride(1),
        weight_ih, weight_ih.stride(0), weight_ih.stride(1),
        bias_ih,
        ih_gates, ih_gates.stride(0), ih_gates.stride(1),
        M, N, K_x,
    )
    
    # Compute hh_gates = h @ weight_hh.T + bias_hh: (M, 3H)
    hh_gates = torch.empty(M, N, device=h.device, dtype=torch.float32)
    grid_hh = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_bias_kernel[grid_hh](
        h, h.stride(0), h.stride(1),
        weight_hh, weight_hh.stride(0), weight_hh.stride(1),
        bias_hh,
        hh_gates, hh_gates.stride(0), hh_gates.stride(1),
        M, N, K_h,
    )
    
    # Apply GRU gates
    new_h = torch.empty(M, H, device=x.device, dtype=torch.float32)
    H_blocks = triton.cdiv(H, 64)  # will be overridden by autotune
    grid_gate = lambda meta: (M * triton.cdiv(H, meta['BLOCK_SIZE']),)
    gru_gate_kernel[grid_gate](
        ih_gates, ih_gates.stride(0), ih_gates.stride(1),
        hh_gates, hh_gates.stride(0), hh_gates.stride(1),
        h, h.stride(0), h.stride(1),
        new_h, new_h.stride(0), new_h.stride(1),
        M, H,
    )
    
    return new_h


def kernel_function(
    p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0,
    p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1,
    p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2,
    p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3,
    p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4,
    p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5,
):
    """
    x: (seq_len, batch_size, input_size) = (512, 10, 128)
    h0: (num_layers, batch_size, hidden_size) = (6, 10, 256)
    """
    seq_len, batch_size, input_size = x.shape
    num_layers = 6
    hidden_size = h0.shape[2]
    
    weights_ih = [p_gru_weight_ih_l0, p_gru_weight_ih_l1, p_gru_weight_ih_l2,
    weights_hh = [p_gru_weight_hh_l0, p_gru_weight_hh_l1, p_gru_weight_hh_l2,
    biases_ih = [p_gru_bias_ih_l0, p_gru_bias_ih_l1, p_gru_bias_ih_l2,
    biases_hh = [p_gru_bias_hh_l0, p_gru_bias_hh_l1, p_gru_bias_hh_l2,
    
    # h_layers[layer] = current hidden state for that layer: (batch_size, hidden_size)
    h_layers = [h0[i] for i in range(num_layers)]
    
    # Output tensor for all time steps from last layer
    output = torch.empty(seq_len, batch_size, hidden_size, device=x.device, dtype=x.dtype)
    
    # Process each time step
    for t in range(seq_len):
        x_t = x[t]  # (batch_size, input_size)
        
        for layer in range(num_layers):
            x_in = x_t if layer == 0 else h_layers[layer - 1]
            # Wait - after updating layer, we need the new h for next layer
            # h_layers[layer] is updated in place
            new_h = gru_step(
                x_in, h_layers[layer],
                weights_ih[layer], weights_hh[layer],
                biases_ih[layer], biases_hh[layer]
            )
            h_layers[layer] = new_h
        
        output[t] = h_layers[-1]
    
    # Stack final hidden states
    h_n = torch.stack(h_layers, dim=0)  # (num_layers, batch_size, hidden_size)
    
    return h_n


def get_inputs():
    batch_size = 10
    seq_len = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6
    
    # Generate weights matching the model structure
    weights = []
    # Layer 0: weight_ih (768, 128), weight_hh (768, 256), bias_ih (768,), bias_hh (768,)
    weights.append(torch.randn(768, 128))
    weights.append(torch.randn(768, 256))
    weights.append(torch.randn(768))
    weights.append(torch.randn(768))
    # Layers 1-5: weight_ih (768, 256), weight_hh (768, 256), bias_ih (768,), bias_hh (768,)
    for _ in range(5):
        weights.append(torch.randn(768, 256))
        weights.append(torch.randn(768, 256))
        weights.append(torch.randn(768))
        weights.append(torch.randn(768))
    
    x = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(num_layers, batch_size, hidden_size)
    
    return weights + [x, h0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Move to GPU
    inputs = [inp.cuda() for inp in inputs]
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        # Build reference model
        batch_size = 10
        seq_len = 512
        input_size = 128
        hidden_size = 256
        num_layers = 6
        
        gru = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False)
        
        # Set weights from inputs
        (p_gru_weight_ih_l0, p_gru_weight_hh_l0, p_gru_bias_ih_l0, p_gru_bias_hh_l0,
         p_gru_weight_ih_l1, p_gru_weight_hh_l1, p_gru_bias_ih_l1, p_gru_bias_hh_l1,
         p_gru_weight_ih_l2, p_gru_weight_hh_l2, p_gru_bias_ih_l2, p_gru_bias_hh_l2,
         p_gru_weight_ih_l3, p_gru_weight_hh_l3, p_gru_bias_ih_l3, p_gru_bias_hh_l3,
         p_gru_weight_ih_l4, p_gru_weight_hh_l4, p_gru_bias_ih_l4, p_gru_bias_hh_l4,
         p_gru_weight_ih_l5, p_gru_weight_hh_l5, p_gru_bias_ih_l5, p_gru_bias_hh_l5,
         x, h0) = inputs
        
        with torch.no_grad():
            gru.weight_ih_l0.copy_(p_gru_weight_ih_l0)
            gru.weight_hh_l0.copy_(p_gru_weight_hh_l0)
            gru.bias_ih_l0.copy_(p_gru_bias_ih_l0)
            gru.bias_hh_l0.copy_(p_gru_bias_hh_l0)
            gru.weight_ih_l1.copy_(p_gru_weight_ih_l1)
            gru.weight_hh_l1.copy_(p_gru_weight_hh_l1)
            gru.bias_ih_l1.copy_(p_gru_bias_ih_l1)
            gru.bias_hh_l1.copy_(p_gru_bias_hh_l1)
            gru.weight_ih_l2.copy_(p_gru_weight_ih_l2)
            gru.weight_hh_l2.copy_(p_gru_weight_hh_l2)
            gru.bias_ih_l2.copy_(p_gru_bias_ih_l2)
            gru.bias_hh_l2.copy_(p_gru_bias_hh_l2)
            gru.weight_ih_l3.copy_(p_gru_