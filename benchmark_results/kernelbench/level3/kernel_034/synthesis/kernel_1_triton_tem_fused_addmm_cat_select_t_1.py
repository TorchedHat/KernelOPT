import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Fused RNN step: given combined input [x_t | h] (8x1280), compute:
# h_new = tanh(combined @ W_i2h.T + b_i2h)  -> (8, 256)
# out   = h_new @ W_h2o.T + b_h2o           -> (8, 128)

@triton.autotune(
    configs=[
        triton.Config({'BM': 8, 'BN': 32, 'BK': 64}, num_warps=2, num_stages=3),
        triton.Config({'BM': 8, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 8, 'BN': 32, 'BK': 128}, num_warps=2, num_stages=4),
        triton.Config({'BM': 8, 'BN': 64, 'BK': 128}, num_warps=4, num_stages=4),
    ],
    key=['K_i2h', 'N_i2h'],
)
@triton.jit
def rnn_i2h_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N_i2h, K_i2h,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K_i2h, BK)):
        k_offs = k * BK + offs_k
        a_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        b_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_i2h)
        b_mask = (offs_n[None, :] < N_i2h) & (k_offs[:, None] < K_i2h)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    
    offs_n_bias = pid_n * BN + tl.arange(0, BN)
    bias = tl.load(B_ptr + offs_n_bias, mask=offs_n_bias < N_i2h, other=0.0)
    acc = acc + bias[None, :]
    acc = libdevice.tanh(acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_i2h)
    tl.store(Out_ptr + offs_m[:, None] * N_i2h + offs_n[None, :], acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BM2': 8, 'BN2': 32, 'BK2': 64}, num_warps=2, num_stages=3),
        triton.Config({'BM2': 8, 'BN2': 64, 'BK2': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM2': 8, 'BN2': 32, 'BK2': 128}, num_warps=2, num_stages=4),
        triton.Config({'BM2': 8, 'BN2': 64, 'BK2': 128}, num_warps=4, num_stages=4),
    ],
    key=['K_h2o', 'N_h2o'],
)
@triton.jit
def rnn_h2o_kernel(
    H_ptr, W_ptr, B_ptr, Out_ptr,
    M, N_h2o, K_h2o,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    BM2: tl.constexpr, BN2: tl.constexpr, BK2: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BM2 + tl.arange(0, BM2)
    offs_n = pid_n * BN2 + tl.arange(0, BN2)
    offs_k = tl.arange(0, BK2)
    
    acc = tl.zeros((BM2, BN2), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K_h2o, BK2)):
        k_offs = k * BK2 + offs_k
        a_ptrs = H_ptr + offs_m[:, None] * stride_hm + k_offs[None, :] * stride_hk
        b_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_h2o)
        b_mask = (offs_n[None, :] < N_h2o) & (k_offs[:, None] < K_h2o)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    
    offs_n_bias = pid_n * BN2 + tl.arange(0, BN2)
    bias = tl.load(B_ptr + offs_n_bias, mask=offs_n_bias < N_h2o, other=0.0)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_h2o)
    tl.store(Out_ptr + offs_m[:, None] * N_h2o + offs_n[None, :], acc, mask=out_mask)


# Cat kernel: concatenate x_t (8x1024) and hidden (8x256) -> combined (8x1280)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_size'],
)
@triton.jit
def cat_kernel(
    X_ptr, H_ptr, Out_ptr,
    batch_size, input_size, hidden_size,
    x_stride_batch, x_stride_feat,
    h_stride_batch, h_stride_feat,
    total_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_size
    
    combined_size = input_size + hidden_size
    row = offs // combined_size
    col = offs % combined_size
    
    in_x_region = col < input_size
    
    x_val = tl.load(X_ptr + row * x_stride_batch + col * x_stride_feat, 
                    mask=mask & in_x_region, other=0.0)
    h_val = tl.load(H_ptr + row * h_stride_batch + (col - input_size) * h_stride_feat,
                    mask=mask & ~in_x_region, other=0.0)
    
    result = tl.where(in_x_region, x_val, h_val)
    tl.store(Out_ptr + offs, result, mask=mask)


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, x, h0):
    seq_len, batch_size, input_size = x.shape
    hidden_size = p_i2h_weight.shape[0]  # 256
    output_size = p_h2o_weight.shape[0]  # 128
    combined_size = input_size + hidden_size  # 1280
    
    device = x.device
    
    # Allocate buffers
    hidden = h0.contiguous().to(device=device, dtype=torch.float32)
    combined = torch.empty(batch_size, combined_size, device=device, dtype=torch.float32)
    outputs = torch.empty(seq_len, batch_size, output_size, device=device, dtype=torch.float32)
    hidden_buf = torch.empty(batch_size, hidden_size, device=device, dtype=torch.float32)
    output_buf = torch.empty(batch_size, output_size, device=device, dtype=torch.float32)
    
    # W_i2h: (hidden_size, combined_size) = (256, 1280)
    # W_h2o: (output_size, hidden_size) = (128, 256)
    W_i2h = p_i2h_weight.contiguous()  # (256, 1280)
    W_h2o = p_h2o_weight.contiguous()  # (128, 256)
    
    total_cat = batch_size * combined_size
    
    for t in range(seq_len):
        x_t = x[t]  # (8, 1024)
        
        # Cat: [x_t | hidden] -> combined (8, 1280)
        grid_cat = lambda meta: (triton.cdiv(total_cat, meta['BLOCK_SIZE']),)
        cat_kernel[grid_cat](
            x_t, hidden, combined,
            batch_size, input_size, hidden_size,
            x_t.stride(0), x_t.stride(1),
            hidden.stride(0), hidden.stride(1),
            total_cat,
        )
        
        # i2h: combined (8,1280) @ W_i2h.T (1280,256) + bias -> tanh -> hidden (8,256)
        grid_i2h = lambda meta: (triton.cdiv(batch_size, meta['BM']), triton.cdiv(hidden_size, meta['BN']))
        rnn_i2h_kernel[grid_i2h](
            combined, W_i2h, p_i2h_bias, hidden_buf,
            batch_size, hidden_size, combined_size,
            combined.stride(0), combined.stride(1),
            W_i2h.stride(0), W_i2h.stride(1),
        )
        hidden = hidden_buf.clone()
        
        # h2o: hidden (8,256) @ W_h2o.T (256,128) + bias -> output (8,128)
        grid_h2o = lambda meta: (triton.cdiv(batch_size, meta['BM2']), triton.cdiv(output_size, meta['BN2']))
        rnn_h2o_kernel[grid_h2o](
            hidden, W_h2o, p_h2o_bias, output_buf,
            batch_size, output_size, hidden_size,
            hidden.stride(0), hidden.stride(1),
            W_h2o.stride(0), W_h2o.stride(1),
        )
        
        outputs[t] = output_buf
    
    return outputs


def get_inputs():
    batch_size = 8
    input_size = 1024
    hidden_size = 256
    output_size = 128
    sequence_length = 256
    
    p_i2h_weight = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float32, device='cuda')
    p_i2h_bias = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    p_h2o_weight = torch.randn(output_size, hidden_size, dtype=torch.float32, device='cuda')
    p_h2o_bias = torch.randn(output_size, dtype=torch.float32, device='cuda')
    x = torch.randn(sequence_length, batch_size, input_size, dtype=torch.float32, device='cuda')
    h0 = torch.randn(batch_size, hidden_size, dtype=torch.float32, device='cuda')
    
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
        
        import torch.nn as nn
        
        class RefModel(nn.Module):
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
                    output = self.h2o(hidden)
                    outputs.append(output)
                return torch.stack(outputs, dim=0)
        
        model = RefModel().cuda()
        with torch.no_grad():
            model.i2h.weight.data = p_i2h_weight
            model.i2h.bias.data = p_i2h_bias
            model.h2o.weight.data = p_h2o_weight
            model.h2o.bias.data = p_h2o_bias
            ref = model(x, h0)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")