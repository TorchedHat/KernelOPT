import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    n_elements, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        n_elements, C, HW,
        eps,
    )
    return out


def kernel_function(*args):
    # Unpack all parameters
    idx = 0
    p_features_0_weight = args[idx]; idx+=1
    p_features_1_weight = args[idx]; idx+=1
    p_features_1_bias = args[idx]; idx+=1
    
    # dense block 0 params (6 layers)
    db0_params = []
    for i in range(6):
        db0_params.append({
            'bn_w': args[idx], 'bn_b': args[idx+1], 'conv_w': args[idx+2]
        })
        idx += 3
    
    # transition 0
    t0_bn_w = args[idx]; idx+=1
    t0_bn_b = args[idx]; idx+=1
    t0_conv_w = args[idx]; idx+=1
    
    # dense block 1 params (12 layers)
    db1_params = []
    for i in range(12):
        db1_params.append({
            'bn_w': args[idx], 'bn_b': args[idx+1], 'conv_w': args[idx+2]
        })
        idx += 3
    
    # transition 1
    t1_bn_w = args[idx]; idx+=1
    t1_bn_b = args[idx]; idx+=1
    t1_conv_w = args[idx]; idx+=1
    
    # dense block 2 params (48 layers)
    db2_params = []
    for i in range(48):
        db2_params.append({
            'bn_w': args[idx], 'bn_b': args[idx+1], 'conv_w': args[idx+2]
        })
        idx += 3
    
    # transition 2
    t2_bn_w = args[idx]; idx+=1
    t2_bn_b = args[idx]; idx+=1
    t2_conv_w = args[idx]; idx+=1
    
    # dense block 3 params (32 layers)
    db3_params = []
    for i in range(32):
        db3_params.append({
            'bn_w': args[idx], 'bn_b': args[idx+1], 'conv_w': args[idx+2]
        })
        idx += 3
    
    # final bn
    p_final_bn_weight = args[idx]; idx+=1
    p_final_bn_bias = args[idx]; idx+=1
    
    # classifier
    p_classifier_weight = args[idx]; idx+=1
    p_classifier_bias = args[idx]; idx+=1
    
    # running stats
    b_features_1_running_mean = args[idx]; idx+=1
    b_features_1_running_var = args[idx]; idx+=1
    
    db0_running = []
    for i in range(6):
        db0_running.append({'mean': args[idx], 'var': args[idx+1]})
        idx += 2
    
    t0_running_mean = args[idx]; idx+=1
    t0_running_var = args[idx]; idx+=1
    
    db1_running = []
    for i in range(12):
        db1_running.append({'mean': args[idx], 'var': args[idx+1]})
        idx += 2
    
    t1_running_mean = args[idx]; idx+=1
    t1_running_var = args[idx]; idx+=1
    
    db2_running = []
    for i in range(48):
        db2_running.append({'mean': args[idx], 'var': args[idx+1]})
        idx += 2
    
    t2_running_mean = args[idx]; idx+=1
    t2_running_var = args[idx]; idx+=1
    
    db3_running = []
    for i in range(32):
        db3_running.append({'mean': args[idx], 'var': args[idx+1]})
        idx += 2
    
    b_final_bn_running_mean = args[idx]; idx+=1
    b_final_bn_running_var = args[idx]; idx+=1
    
    x = args[idx]; idx+=1
    
    # Forward pass
    # Initial conv
    x = torch.ops.aten.convolution(x, p_features_0_weight, None, [2,2], [3,3], [1,1], False, [0,0], 1)
    x = fused_bn_relu(x, p_features_1_weight, p_features_1_bias, b_features_1_running_mean, b_features_1_running_var)
    x = torch.ops.aten.max_pool2d_with_indices(x, [3,3], [2,2], [1,1])[0]
    
    # Dense block 0 (6 layers, growth=32, start=64)
    for i in range(6):
        p = db0_params[i]
        r = db0_running[i]
        feat = fused_bn_relu(x, p['bn_w'], p['bn_b'], r['mean'], r['var'])
        feat = torch.ops.aten.convolution(feat, p['conv_w'], None, [1,1], [1,1], [1,1], False, [0,0], 1)
        x = torch.cat([x, feat], dim=1)
    
    # Transition 0
    x = fused_bn_relu(x, t0_bn_w, t0_bn_b, t0_running_mean, t0_running_var)
    x = torch.ops.aten.convolution(x, t0_conv_w, None, [1,1], [0,0], [1,1], False, [0,0], 1)
    x = torch.ops.aten.avg_pool2d(x, [2,2], [2,2])
    
    # Dense block 1 (12 layers)
    for i in range(12):
        p = db1_params[i]
        r = db1_running[i]
        feat = fused_bn_relu(x, p['bn_w'], p['bn_b'], r['mean'], r['var'])
        feat = torch.ops.aten.convolution(feat, p['conv_w'], None, [1,1], [1,1], [1,1], False, [0,0], 1)
        x = torch.cat([x, feat], dim=1)
    
    # Transition 1
    x = fused_bn_relu(x, t1_bn_w, t1_bn_b, t1_running_mean, t1_running_var)
    x = torch.ops.aten.convolution(x, t1_conv_w, None, [1,1], [0,0], [1,1], False, [0,0], 1)
    x = torch.ops.aten.avg_pool2d(x, [2,2], [2,2])
    
    # Dense block 2 (48 layers)
    for i in range(48):
        p = db2_params[i]
        r = db2_running[i]
        feat = fused_bn_relu(x, p['bn_w'], p['bn_b'], r['mean'], r['var'])
        feat = torch.ops.aten.convolution(feat, p['conv_w'], None, [1,1], [1,1], [1,1], False, [0,0], 1)
        x = torch.cat([x, feat], dim=1)
    
    # Transition 2
    x = fused_bn_relu(x, t2_bn_w, t2_bn_b, t2_running_mean, t2_running_var)
    x = torch.ops.aten.convolution(x, t2_conv_w, None, [1,1], [0,0], [1,1], False, [0,0], 1)
    x = torch.ops.aten.avg_pool2d(x, [2,2], [2,2])
    
    # Dense block 3 (32 layers)
    for i in range(32):
        p = db3_params[i]
        r = db3_running[i]
        feat = fused_bn_relu(x, p['bn_w'], p['bn_b'], r['mean'], r['var'])
        feat = torch.ops.aten.convolution(feat, p['conv_w'], None, [1,1], [1,1], [1,1], False, [0,0], 1)
        x = torch.cat([x, feat], dim=1)
    
    # Final BN + ReLU
    x = fused_bn_relu(x, p_final_bn_weight, p_final_bn_bias, b_final_bn_running_mean, b_final_bn_running_var)
    
    # Adaptive avg pool
    x = torch.ops.aten.adaptive_avg_pool2d(x, [1,1])
    x = x.view(x.shape[0], -1)
    
    # Linear
    x = torch.ops.aten.linear(x, p_classifier_weight, p_classifier_bias)
    
    return x