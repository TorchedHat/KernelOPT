import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc)
        
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def inception_module_forward(x, branch1x1_w, branch1x1_b,
                              branch3x3_0_w, branch3x3_0_b, branch3x3_1_w, branch3x3_1_b,
                              branch5x5_0_w, branch5x5_0_b, branch5x5_1_w, branch5x5_1_b,
    b1 = F.conv2d(x, branch1x1_w, branch1x1_b)
    b1 = F.relu(b1)
    
    b2 = F.conv2d(x, branch3x3_0_w, branch3x3_0_b)
    b2 = F.relu(b2)
    b2 = F.conv2d(b2, branch3x3_1_w, branch3x3_1_b, padding=1)
    b2 = F.relu(b2)
    
    b3 = F.conv2d(x, branch5x5_0_w, branch5x5_0_b)
    b3 = F.relu(b3)
    b3 = F.conv2d(b3, branch5x5_1_w, branch5x5_1_b, padding=2)
    b3 = F.relu(b3)
    
    b4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    b4 = F.conv2d(b4, pool_proj_w, pool_proj_b)
    b4 = F.relu(b4)
    
    return torch.cat([b1, b2, b3, b4], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
):
    # conv1 -> relu -> maxpool
    x = F.conv2d(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # conv2 -> relu
    x = F.conv2d(x, p_conv2_weight, p_conv2_bias)
    x = F.relu(x)
    
    # conv3 -> relu -> maxpool
    x = F.conv2d(x, p_conv3_weight, p_conv3_bias, padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception3a
    x = inception_module_forward(
        x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )
    
    # inception3b
    x = inception_module_forward(
        x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )
    
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception4a
    x = inception_module_forward(
        x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )
    
    # inception4b
    x = inception_module_forward(
        x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
        p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    )
    
    # inception4c
    x = inception_module_forward(
        x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
        p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    )
    
    # inception4d
    x = inception_module_forward(
        x,
        p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
        p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
        p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
        p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
        p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
        p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    )
    
    # inception4e
    x = inception_module_forward(
        x,
        p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
        p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
        p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
        p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
        p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
        p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    )
    
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception5a
    x = inception_module_forward(
        x,
        p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
        p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
        p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
        p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
        p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
        p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    )
    
    # inception5b
    x = inception_module_forward(
        x,
        p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
        p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
        p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
        p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
        p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
        p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    )
    
    # adaptive avg pool -> flatten -> dropout (no-op at inference) -> linear
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.flatten(1)
    # dropout is identity at inference
    
    # Linear layer via Triton GEMM
    M, K = x.shape
    N = p_fc_weight.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    linear_kernel[grid](
        x, p_fc_weight, p_fc_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_fc_weight.stride(0), p_fc_weight.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def get_inputs():
    torch.manual_seed(42)
    
    def make_conv_params(out_c, in_c, kh, kw):
        w = torch.randn(out_c, in_c, kh, kw, device='cuda')
        b = torch.randn(out_c, device='cuda')
        return w, b
    
    p_conv1_weight, p_conv1_bias = make_conv_params(64, 3, 7, 7)
    p_conv2_weight, p_conv2_bias = make_conv_params(64, 64, 1, 1)
    p_conv3_weight, p_conv3_bias = make_conv_params(192, 64, 3, 3)
    
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias = make_conv_params(64, 192, 1, 1)
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias = make_conv_params(96, 192, 1, 1)
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias = make_conv_params(128, 96, 3, 3)
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias = make_conv_params(16, 192, 1, 1)
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias = make_conv_params(32, 16, 5, 5)
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias = make_conv_params(32, 192, 1, 1)
    
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias = make_conv_params(128, 256, 1, 1)
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias = make_conv_params(128, 256, 1, 1)
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias = make_conv_params(192, 128, 3, 3)
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias = make_conv_params(32, 256, 1, 1)
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias = make_conv_params(96, 32, 5, 5)
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias = make_conv_params(64, 256, 1, 1)
    
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias = make_conv_params(192, 480, 1, 1)
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias = make_conv_params(96, 480, 1, 1)
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias = make_conv_params(208, 96, 3, 3)
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias = make_conv_params(16, 480, 1, 1)
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias = make_conv_params(48, 16, 5, 5)
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias = make_conv_params(64, 480, 1, 1)
    
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias = make_conv_params(160, 512, 1, 1)
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias = make_conv_params(112, 512, 1, 1)
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias = make_conv_params(224, 112, 3, 3)
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias = make_conv_params(24, 512, 1, 1)
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias = make_conv_params(64, 24, 5, 5)
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias = make_conv_params(64, 512, 1, 1)
    
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias = make_conv_params(128, 512, 1, 1)
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias = make_conv_params(128, 512, 1, 1)
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias = make_conv_params(256, 128, 3, 3)
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias = make_conv_params(24, 512, 1, 1)
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias = make_conv_params(64, 24, 5, 5)
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias = make_conv_params(64, 512, 1, 1)
    
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias = make_conv_params(112, 512, 1, 1)
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias = make_conv_params(144, 512, 1, 1)
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias = make_conv_params(288, 144, 3, 3)
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias = make_conv_params(32, 512, 1, 1)
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias = make_conv_params(64, 32, 5, 5)
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias = make_conv_params(64, 512, 1, 1)
    
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias = make_conv_params(256, 528, 1, 1)
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias = make_conv_params(160, 528, 1, 1)
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias = make_conv_params(320, 160, 3, 3)
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias = make_conv_params(32, 528, 1, 1)
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias = make_conv_params(128, 32, 5, 5)