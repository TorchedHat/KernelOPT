import torch
import triton
import triton.language as tl

# Kernel 1: Fused BatchNorm + ReLU (pointwise)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N_ELEMENTS'],
)
@triton.jit
def bn_relu_kernel(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    # Determine channel index
    c = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Kernel 2: 1x1 Conv as GEMM (M x K) x (K x N) where M=N*H*W, K=Cin, N=Cout
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptr + offs_m[:, None] * K + k_offs[None, :], mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


# Kernel 3: AvgPool2d (kernel=2, stride=2)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N_ELEMENTS'],
)
@triton.jit
def avgpool2d_kernel(in_ptr, out_ptr, N, C, H_in, W_in, H_out, W_out,
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    # Decode output index: [N, C, H_out, W_out]
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)
    h_in = h_out * 2
    w_in = w_out * 2
    # Load 4 values for 2x2 pool
    idx00 = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in
    idx01 = idx00 + 1
    idx10 = idx00 + W_in
    idx11 = idx00 + W_in + 1
    v00 = tl.load(in_ptr + idx00, mask=mask, other=0.0)
    v01 = tl.load(in_ptr + idx01, mask=mask, other=0.0)
    v10 = tl.load(in_ptr + idx10, mask=mask, other=0.0)
    v11 = tl.load(in_ptr + idx11, mask=mask, other=0.0)
    out = (v00 + v01 + v10 + v11) * 0.25
    tl.store(out_ptr + offsets, out, mask=mask)


def kernel_function(p_transition_0_weight, p_transition_0_bias,
                    p_transition_2_weight,
                    b_transition_0_running_mean, b_transition_0_running_var,
    N, C, H, W = x.shape
    # Step 1: BN + ReLU
    bn_out = torch.empty_like(x)
    n_elements = x.numel()
    grid_bn = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid_bn](
        x, b_transition_0_running_mean, b_transition_0_running_var,
        p_transition_0_weight, p_transition_0_bias, bn_out,
    )

    # Step 2: 1x1 Conv as GEMM
    # bn_out: [N, C_in, H, W] -> reshape to [N*H*W, C_in]
    Cout = p_transition_2_weight.shape[0]
    Cin = p_transition_2_weight.shape[1]
    M = N * H * W
    # A: [M, Cin] - need to permute bn_out from NCHW to NHWC
    A = bn_out.permute(0, 2, 3, 1).contiguous().reshape(M, Cin)
    # B: [Cin, Cout] - weight is [Cout, Cin, 1, 1]
    B = p_transition_2_weight.reshape(Cout, Cin).t().contiguous()
    conv_out_flat = torch.empty((M, Cout), device=x.device, dtype=x.dtype)
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Cout, meta['BLOCK_N']))
    conv1x1_gemm_kernel[grid_gemm](A, B, conv_out_flat, M, Cout, Cin)
    # Reshape back to NCHW
    conv_out = conv_out_flat.reshape(N, H, W, Cout).permute(0, 3, 1, 2).contiguous()

    # Step 3: AvgPool2d (kernel=2, stride=2)
    H_out = H // 2
    W_out = W // 2
    pool_out = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)
    n_pool = N * Cout * H_out * W_out
    grid_pool = lambda meta: (triton.cdiv(n_pool, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid_pool](
    )
    return pool_out


def get_inputs():
    batch_size = 128
    num_input_features = 32
    num_output_features = 64
    height, width = 256, 256
    p_transition_0_weight = torch.rand(num_input_features, device='cuda')
    p_transition_0_bias = torch.rand(num_input_features, device='cuda')
    p_transition_2_weight = torch.rand(num_output_features, num_input_features, 1, 1, device='cuda')
    b_transition_0_running_mean = torch.rand(num_input_features, device='cuda')
    b_transition_0_running_var = torch.rand(num_input_features, device='cuda') + 0.1
    x = torch.rand(batch_size, num_input_features, height, width, device='cuda')
    return [p_transition_0_weight, p_transition_0_bias, p_transition_2_weight,


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
        p_w, p_b, conv_w, rm, rv, x = inputs
        model = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        model.eval()
        with torch.no_grad():
            model[0].weight.data.copy_(p_w)
            model[0].bias.data.copy_(p_b)
            model[0].running_mean.data.copy_(rm)
            model[0].running_var.data.copy_(rv)
            model[2].weight.data.copy_(conv_w)
            ref = model(x)
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")