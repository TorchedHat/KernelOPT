import torch
import triton
import triton.language as tl
import math

# BatchNorm + ReLU fused pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode channel index
    c = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_scaled = x_norm * w + b
    x_relu = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + offs, x_relu, mask=mask)


# Implicit GEMM Conv2d kernel (NHWC layout internally)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Strides for X in NHWC: [IN_H*IN_W*IN_C, IN_W*IN_C, IN_C, 1]
    # Strides for W in (OUT_C, KH, KW, IN_C): [KH*KW*IN_C, KW*IN_C, IN_C, 1]
    # Strides for Y in NHWC: [OUT_H*OUT_W*OUT_C, OUT_W*OUT_C, OUT_C, 1]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W
    # N = OUT_C
    # K = KH * KW * IN_C
    M_TOTAL = BATCH * OUT_H * OUT_W
    K_TOTAL = KH * KW * IN_C
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < K_TOTAL
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial & (batch_idx[:, None] < BATCH)
        
        # X in NHWC: index = batch*IN_H*IN_W*IN_C + ih*IN_W*IN_C + iw*IN_C + in_c
        x_idx = batch_idx[:, None] * (IN_H * IN_W * IN_C) + ih * (IN_W * IN_C) + iw * IN_C + in_c_idx[None, :]
        x_tile = tl.load(X_ptr + x_idx, mask=mask_x, other=0.0)
        
        # W in (OUT_C, KH, KW, IN_C) -> NHWC-like: index = oc*KH*KW*IN_C + kh*KW*IN_C + kw*IN_C + in_c
        # We need W as [BLOCK_K, BLOCK_N]
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_idx = offs_n[None, :] * (KH * KW * IN_C) + kh_idx[:, None] * (KW * IN_C) + kw_idx[:, None] * IN_C + in_c_idx[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Store output in NHWC
    offs_m_valid = offs_m < M_TOTAL
    offs_n_valid = offs_n < OUT_C
    mask_out = offs_m_valid[:, None] & offs_n_valid[None, :]
    
    y_idx = batch_idx[:, None] * (OUT_H * OUT_W * OUT_C) + out_h_idx[:, None] * (OUT_W * OUT_C) + out_w_idx[:, None] * OUT_C + offs_n[None, :]
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=mask_out)


def bn_relu(x, mean, var, weight, bias):
    """Apply BatchNorm + ReLU"""
    N_total = x.numel()
    C = mean.shape[0]
    HW = x.shape[2] * x.shape[3]
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, C, HW, N_total)
    return out


def conv2d_triton(x_nhwc, w, OUT_H, OUT_W, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W):
    """Conv2d using implicit GEMM, x in NHWC, w in (OUT_C, IN_C, KH, KW)"""
    BATCH, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C = w.shape[0]
    
    # Reorder w to (OUT_C, KH, KW, IN_C) for our kernel
    # w is (OUT_C, IN_C, KH, KW) -> (OUT_C, KH, KW, IN_C)
    w_reordered = w.permute(0, 2, 3, 1).contiguous()
    
    y_nhwc = torch.empty(BATCH, OUT_H, OUT_W, OUT_C, device=x_nhwc.device, dtype=x_nhwc.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_reordered, y_nhwc,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
    )
    
    return y_nhwc


def kernel_function(
    p_layers_0_0_weight, p_layers_0_0_bias,
    p_layers_0_2_weight,
    p_layers_1_0_weight, p_layers_1_0_bias,
    p_layers_1_2_weight,
    p_layers_2_0_weight, p_layers_2_0_bias,
    p_layers_2_2_weight,
    p_layers_3_0_weight, p_layers_3_0_bias,
    p_layers_3_2_weight,
    p_layers_4_0_weight, p_layers_4_0_bias,
    p_layers_4_2_weight,
    p_layers_5_0_weight, p_layers_5_0_bias,
    p_layers_5_2_weight,
    b_layers_0_0_running_mean, b_layers_0_0_running_var,
    b_layers_1_0_running_mean, b_layers_1_0_running_var,
    b_layers_2_0_running_mean, b_layers_2_0_running_var,
    b_layers_3_0_running_mean, b_layers_3_0_running_var,
    b_layers_4_0_running_mean, b_layers_4_0_running_var,
    b_layers_5_0_running_mean, b_layers_5_0_running_var,
    x,
):
    BATCH, C0, H, W = x.shape  # [10, 32, 224, 224]
    KH, KW = 3, 3
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 1, 1
    OUT_H = (H + 2*PAD_H - KH) // STRIDE_H + 1  # 224
    OUT_W = (W + 2*PAD_W - KW) // STRIDE_W + 1  # 224
    
    features = [x]  # list of NCHW tensors
    
    layer_params = [
        (p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
         b_layers_0_0_running_mean, b_layers_0_0_running_var),
        (p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
         b_layers_1_0_running_mean, b_layers_1_0_running_var),
        (p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
         b_layers_2_0_running_mean, b_layers_2_0_running_var),
        (p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
         b_layers_3_0_running_mean, b_layers_3_0_running_var),
        (p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
         b_layers_4_0_running_mean, b_layers_4_0_running_var),
        (p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
         b_layers_5_0_running_mean, b_layers_5_0_running_var),
    ]
    
    x_cat = x  # current concatenated tensor (NCHW)
    
    for (bn_w, bn_b, conv_w, running_mean, running_var) in layer_params:
        # BatchNorm + ReLU on x_cat
        bn_out = bn_relu(x_cat, running_mean, running_var, bn_w, bn_b)
        
        # Convert to NHWC for conv
        bn_out_nhwc = bn_out.permute(0, 2, 3, 1).contiguous()
        
        # Conv2d
        conv_out_nhwc = conv2d_triton(
        )
        
        # Convert back to NCHW
        conv_out = conv_out_nhwc.permute(0, 3, 1, 2).contiguous()
        
        # Dropout(0.0) is identity
        # Cat
        features.append(conv_out)
        x_cat = torch.cat(features, dim=1)
    
    return x_cat


def get_inputs():
    batch_size = 10
    num_layers = 6
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    
    inputs = []
    # Layer weights and biases
    in_features = num_input_features
    for i in range(num_layers):
        inputs.append(torch.randn(in_features + i * growth_rate, device='cuda'))  # bn weight
        inputs.append(torch.randn(in_features + i * growth_rate, device='cuda'))  # bn bias
        inputs.append(torch.randn(growth_rate, in_features + i * growth_rate, 3, 3, device='cuda'))  # conv weight
    
    # Running stats
    in_features = num_input_features
    for i in range(num_layers):
        inputs.append(torch.zeros(in_features + i * growth_rate, device='cuda'))  # running mean
        inputs.append(torch.ones(in_features + i * growth_rate, device='cuda'))   # running var
    
    # Input x
    inputs.append(torch.randn(batch_size, num_input_features, height, width, device='cuda'))
    
    return inputs


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
        
        # Reconstruct model for reference
        class Model(nn.Module):
            def __init__(self, num_layers, num_input_features, growth_rate):
                super(Model, self).__init__()
                layers = []
                for i in range(num_layers):
                    layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
                self.layers = nn.ModuleList(layers)
            
            def _make_layer(self, in_features, growth_rate):
                return nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.Dropout(0.0)
                )
            
            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feature = layer(x)
                    features.append(new_feature)
                    x = torch.cat(features, 1)
                return x
        
        model = Model(6, 32, 32).cuda().eval()
        
        # Set weights from inputs
        # inputs order: [bn_w0, bn_b0, conv_w0, bn_w1, bn_b1, conv_w1, ..., rm0, rv0, rm1, rv1, ..., x]
        idx = 0
        for i in range(6):
            model.layers[i][0].weight.data = inputs[idx].clone()
            idx += 1
            model.layers[i][0].bias.data = inputs[idx].clone()
            idx += 1
            model.layers[i][2].weight.data = inputs[idx].clone()
            idx += 1
        
        for i in range(6):
            in_f = 32 + i * 32
            model.layers[i][0].running_mean.data = inputs[idx].clone()
            idx += 1
            model.layers[i][0].running_var.data = inputs[idx].clone()
            idx += 1
        
        x_ref = inputs[idx].clone()
        
        with torch.no_grad():
            ref = model(x_ref)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")