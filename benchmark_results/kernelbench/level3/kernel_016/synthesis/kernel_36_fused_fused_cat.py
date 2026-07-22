import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # offsets layout: N*C*H*W, channel = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = N * C * H * W
    HW = H * W
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        C, HW,
        eps,
        n_elements,
    )
    return out


def dense_layer(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight):
    """BN + ReLU + Conv2d + (dropout with p=0 is identity)"""
    x = fused_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var)
    x = F.conv2d(x, conv_weight, bias=None, padding=1)
    # dropout with p=0.0 is identity
    return x


def transition_layer(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight):
    """BN + ReLU + Conv2d(1x1) + AvgPool2d"""
    x = fused_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var)
    x = F.conv2d(x, conv_weight, bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def kernel_function(*args):
    idx = 0
    # Parameters in order as listed in model inputs
    p_features_0_weight = args[idx]; idx+=1          # [64, 3, 7, 7]
    p_features_1_weight = args[idx]; idx+=1          # [64]
    p_features_1_bias = args[idx]; idx+=1            # [64]
    
    # dense_blocks_0 layers 0-5
    p_db0_l0_bn_w = args[idx]; idx+=1   # [64]
    p_db0_l0_bn_b = args[idx]; idx+=1   # [64]
    p_db0_l0_conv = args[idx]; idx+=1   # [32, 64, 3, 3]
    p_db0_l1_bn_w = args[idx]; idx+=1   # [96]
    p_db0_l1_bn_b = args[idx]; idx+=1   # [96]
    p_db0_l1_conv = args[idx]; idx+=1   # [32, 96, 3, 3]
    p_db0_l2_bn_w = args[idx]; idx+=1   # [128]
    p_db0_l2_bn_b = args[idx]; idx+=1   # [128]
    p_db0_l2_conv = args[idx]; idx+=1   # [32, 128, 3, 3]
    p_db0_l3_bn_w = args[idx]; idx+=1   # [160]
    p_db0_l3_bn_b = args[idx]; idx+=1   # [160]
    p_db0_l3_conv = args[idx]; idx+=1   # [32, 160, 3, 3]
    p_db0_l4_bn_w = args[idx]; idx+=1   # [192]
    p_db0_l4_bn_b = args[idx]; idx+=1   # [192]
    p_db0_l4_conv = args[idx]; idx+=1   # [32, 192, 3, 3]
    p_db0_l5_bn_w = args[idx]; idx+=1   # [224]
    p_db0_l5_bn_b = args[idx]; idx+=1   # [224]
    p_db0_l5_conv = args[idx]; idx+=1   # [32, 224, 3, 3]
    
    # transition_layers_0
    p_tr0_bn_w = args[idx]; idx+=1   # [256]
    p_tr0_bn_b = args[idx]; idx+=1   # [256]
    p_tr0_conv = args[idx]; idx+=1   # [128, 256, 1, 1]
    
    # dense_blocks_1 layers 0-11
    p_db1_l0_bn_w = args[idx]; idx+=1   # [128]
    p_db1_l0_bn_b = args[idx]; idx+=1   # [128]
    p_db1_l0_conv = args[idx]; idx+=1   # [32, 128, 3, 3]
    p_db1_l1_bn_w = args[idx]; idx+=1   # [160]
    p_db1_l1_bn_b = args[idx]; idx+=1   # [160]
    p_db1_l1_conv = args[idx]; idx+=1   # [32, 160, 3, 3]
    p_db1_l2_bn_w = args[idx]; idx+=1   # [192]
    p_db1_l2_bn_b = args[idx]; idx+=1   # [192]
    p_db1_l2_conv = args[idx]; idx+=1   # [32, 192, 3, 3]
    p_db1_l3_bn_w = args[idx]; idx+=1   # [224]
    p_db1_l3_bn_b = args[idx]; idx+=1   # [224]
    p_db1_l3_conv = args[idx]; idx+=1   # [32, 224, 3, 3]
    p_db1_l4_bn_w = args[idx]; idx+=1   # [256]
    p_db1_l4_bn_b = args[idx]; idx+=1   # [256]
    p_db1_l4_conv = args[idx]; idx+=1   # [32, 256, 3, 3]
    p_db1_l5_bn_w = args[idx]; idx+=1   # [288]
    p_db1_l5_bn_b = args[idx]; idx+=1   # [288]
    p_db1_l5_conv = args[idx]; idx+=1   # [32, 288, 3, 3]
    p_db1_l6_bn_w = args[idx]; idx+=1   # [320]
    p_db1_l6_bn_b = args[idx]; idx+=1   # [320]
    p_db1_l6_conv = args[idx]; idx+=1   # [32, 320, 3, 3]
    p_db1_l7_bn_w = args[idx]; idx+=1   # [352]
    p_db1_l7_bn_b = args[idx]; idx+=1   # [352]
    p_db1_l7_conv = args[idx]; idx+=1   # [32, 352, 3, 3]
    p_db1_l8_bn_w = args[idx]; idx+=1   # [384]
    p_db1_l8_bn_b = args[idx]; idx+=1   # [384]
    p_db1_l8_conv = args[idx]; idx+=1   # [32, 384, 3, 3]
    p_db1_l9_bn_w = args[idx]; idx+=1   # [416]
    p_db1_l9_bn_b = args[idx]; idx+=1   # [416]
    p_db1_l9_conv = args[idx]; idx+=1   # [32, 416, 3, 3]
    p_db1_l10_bn_w = args[idx]; idx+=1  # [448]
    p_db1_l10_bn_b = args[idx]; idx+=1  # [448]
    p_db1_l10_conv = args[idx]; idx+=1  # [32, 448, 3, 3]
    p_db1_l11_bn_w = args[idx]; idx+=1  # [480]
    p_db1_l11_bn_b = args[idx]; idx+=1  # [480]
    p_db1_l11_conv = args[idx]; idx+=1  # [32, 480, 3, 3]
    
    # transition_layers_1
    p_tr1_bn_w = args[idx]; idx+=1   # [512]
    p_tr1_bn_b = args[idx]; idx+=1   # [512]
    p_tr1_conv = args[idx]; idx+=1   # [256, 512, 1, 1]
    
    # dense_blocks_2 layers 0-47
    db2_params = []
    for i in range(48):
        bn_w = args[idx]; idx+=1
        bn_b = args[idx]; idx+=1
        conv_w = args[idx]; idx+=1
        db2_params.append((bn_w, bn_b, conv_w))
    
    # transition_layers_2
    p_tr2_bn_w = args[idx]; idx+=1   # [1792]
    p_tr2_bn_b = args[idx]; idx+=1   # [1792]
    p_tr2_conv = args[idx]; idx+=1   # [896, 1792, 1, 1]
    
    # dense_blocks_3 layers 0-31
    db3_params = []
    for i in range(32):
        bn_w = args[idx]; idx+=1
        bn_b = args[idx]; idx+=1
        conv_w = args[idx]; idx+=1
        db3_params.append((bn_w, bn_b, conv_w))
    
    # final_bn
    p_final_bn_w = args[idx]; idx+=1   # [1920]
    p_final_bn_b = args[idx]; idx+=1   # [1920]
    
    # classifier
    p_classifier_w = args[idx]; idx+=1  # [10, 1920]
    p_classifier_b = args[idx]; idx+=1  # [10]
    
    # running stats for features_1
    b_features_1_mean = args[idx]; idx+=1   # [64]
    b_features_1_var = args[idx]; idx+=1    # [64]
    
    # running stats for dense_blocks_0 layers 0-5
    b_db0_l0_mean = args[idx]; idx+=1
    b_db0_l0_var = args[idx]; idx+=1
    b_db0_l1_mean = args[idx]; idx+=1
    b_db0_l1_var = args[idx]; idx+=1
    b_db0_l2_mean = args[idx]; idx+=1
    b_db0_l2_var = args[idx]; idx+=1
    b_db0_l3_mean = args[idx]; idx+=1
    b_db0_l3_var = args[idx]; idx+=1
    b_db0_l4_mean = args[idx]; idx+=1
    b_db0_l4_var = args[idx]; idx+=1
    b_db0_l5_mean = args[idx]; idx+=1
    b_db0_l5_var = args[idx]; idx+=1
    
    # running stats for transition_layers_0
    b_tr0_mean = args[idx]; idx+=1
    b_tr0_var = args[idx]; idx+=1
    
    # running stats for dense_blocks_1 layers 0-11
    b_db1_stats = []
    for i in range(12):
        m = args[idx]; idx+=1
        v = args[idx]; idx+=1
        b_db1_stats.append((m, v))
    
    # running stats for transition_layers_1
    b_tr1_mean = args[idx]; idx+=1
    b_tr1_var = args[idx]; idx+=1
    
    # running stats for dense_blocks_2 layers 0-47
    b_db2_stats = []
    for i in range(48):
        m = args[idx]; idx+=1
        v = args[idx]; idx+=1
        b_db2_stats.append((m, v))
    
    # running stats for transition_layers_2
    b_tr2_mean = args[idx]; idx+=1
    b_tr2_var = args[idx]; idx+=1
    
    # running stats for dense_blocks_3 layers 0-31
    b_db3_stats = []
    for i in range(32):
        m = args[idx]; idx+=1
        v = args[idx]; idx+=1
        b_db3_stats.append((m, v))
    
    # running stats for final_bn
    b_final_bn_mean = args[idx]; idx+=1
    b_final_bn_var = args[idx]; idx+=1
    
    # input x
    x = args[idx]; idx+=1
    
    # Initial features: Conv2d(3, 64, 7, stride=2, padding=3) + BN + ReLU + MaxPool
    x = F.conv2d(x, p_features_0_weight, bias=None, stride=2, padding=3)
    x = fused_bn_relu(x, p_features_1_weight, p_features_1_bias, b_features_1_mean, b_features_1_var)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Dense Block 0 (6 layers)
    db0_bn_w = [p_db0_l0_bn_w, p_db0_l1_bn_w, p_db0_l2_bn_w, p_db0_l3_bn_w, p_db0_l4_bn_w, p_db0_l5_bn_w]
    db0_bn_b = [p_db0_l0_bn_b, p_db0_l1_bn_b, p_db0_l2_bn_b, p_db0_l3_bn_b, p_db0_l4_bn_b, p_db0_l5_bn_b]
    db0_conv = [p_db0_l0_conv, p_db0_l1_conv, p_db0_l2_conv, p_db0_l3_conv, p_db0_l4_conv, p_db0_l5_conv]
    db0_means = [b_db0_l0_mean, b_db0_l1_mean, b_db0_l2_mean, b_db0_l3_mean, b_db0_l4_mean, b_db0_l5_mean]
    db0_vars = [b_db0_l0_var, b_db0_l1_var, b_db0_l2_var, b_db0_l3_var, b_db0_l4_var, b_db0_l5_var]
    
    features = [x]
    for i in range(6):
        new_feat = dense_layer(x, db0_bn_w[i], db0_bn_b[i], db0_means[i], db0_vars[i], db0_conv[i])
        features.append(new_feat)
        x = torch.cat(features, dim=1)
    
    # Transition 0
    x = transition_layer(x, p_tr0_bn_w, p_tr0_bn_b, b_tr0_mean, b_tr0_var, p_tr0_conv)
    
    # Dense Block 1 (12 layers)
    db1_bn_w = [p_db1_l0_bn_w, p_db1_l1_bn_w, p_db1_l2_bn_w, p_db1_l3_bn_w, p_db1_l4_bn_w, p_db1_l5_bn_w,
    db1_bn_b = [p_db1_l0_bn_b, p_db1_l1_bn_b, p_db1_l2_bn_b, p_db1_l3_bn_b, p_db1_l4_bn_b, p_db1_l5_bn_b,
    db1_conv = [p_db1_l0_conv, p_db1_l1_conv, p_db1_l2_conv, p_db1_l3_conv, p_db1_l4_conv, p_db1_l5_conv,
    
    features = [x]
    for i in range(12):
        m, v = b_db1_stats[i]
        new_feat = dense_layer(x, db1_bn_w[i], db1_bn_b[i], m, v, db1_conv[i])
        features.append(new_feat)
        x = torch.cat(features, dim=1)
    
    # Transition 1
    x = transition_layer(x, p_tr1_bn_w, p_tr1_bn_b, b_tr1_mean, b_tr1_var, p_tr1_conv)
    
    # Dense Block 2 (48 layers)
    features = [x]
    for i in range(48):
        bn_w, bn_b, conv_w = db2_params[i]
        m, v = b_db2_stats[i]
        new_feat = dense_layer(x, bn_w, bn_b, m, v, conv_w)
        features.append(new_feat)
        x = torch.cat(features, dim=1)
    
    # Transition 2
    x = transition_layer(x, p_tr2_bn_w, p_tr2_bn_b, b_tr2_mean, b_tr2_var, p_tr2_conv)
    
    # Dense Block 3 (32 layers)
    features = [x]
    for i in range(32):
        bn_w, bn_b, conv_w = db3_params[i]
        m, v = b_db3_stats[i]
        new_feat = dense_layer(x, bn_w, bn_b, m, v, conv_w)
        features.append(new_feat)
        x = torch.cat(features, dim=1)
    
    # Final BN + ReLU
    x = fused_bn_relu(x, p_final_bn_w, p_final_bn_b, b_final_bn_mean, b_final_bn_var)
    
    # Adaptive avg pool + flatten + linear
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    x = F.linear(x, p_classifier_w, p_classifier_b)
    
    return x


def get_inputs():
    device = 'cuda'
    inputs = []
    
    # p_features_0_weight
    inputs.append(torch.randn(64, 3, 7, 7, device=device))
    # p_features_1_weight, p_features_1_bias
    inputs.append(torch.ones(64, device=device))
    inputs.append(torch.zeros(64, device=device))
    
    # dense_blocks_0 layers 0-5 (6 layers, growth_rate=32, starting from 64)
    in_ch = 64
    for i in range(6):
        inputs.append(torch.ones(in_ch, device=device))   # bn_w
        inputs.append(torch.zeros(in_ch, device=device))  # bn_b
        inputs.append(torch.randn(32, in_ch, 3, 3, device=device))  # conv_w
        in_ch += 32
    # in_ch = 256
    
    # transition_layers_0
    inputs.append(torch.ones(256, device=device))   # bn_w
    inputs.append(torch.zeros(256, device=device))  # bn_b
    inputs.append(torch.randn(128, 256, 1, 1, device=device))  # conv_w
    
    # dense_blocks_1 layers 0-11 (12 layers, starting from 128)
    in_ch = 128
    for i in range(12):
        inputs.append(torch.ones(in_ch, device=device))
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.randn(32, in_ch, 3, 3, device=device))
        in_ch += 32
    # in_ch = 512
    
    # transition_layers_1
    inputs.append(torch.ones(512, device=device))
    inputs.append(torch.zeros(512, device=device))
    inputs.append(torch.randn(256, 512, 1, 1, device=device))
    
    # dense_blocks_2 layers 0-47 (48 layers, starting from 256)
    in_ch = 256
    for i in range(48):
        inputs.append(torch.ones(in_ch, device=device))
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.randn(32, in_ch, 3, 3, device=device))
        in_ch += 32
    # in_ch = 1792
    
    # transition_layers_2
    inputs.append(torch.ones(1792, device=device))
    inputs.append(torch.zeros(1792, device=device))
    inputs.append(torch.randn(896, 1792, 1, 1, device=device))
    
    # dense_blocks_3 layers 0-31 (32 layers, starting from 896)
    in_ch = 896
    for i in range(32):
        inputs.append(torch.ones(in_ch, device=device))
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.randn(32, in_ch, 3, 3, device=device))
        in_ch += 32
    # in_ch = 1920
    
    # final_bn
    inputs.append(torch.ones(1920, device=device))
    inputs.append(torch.zeros(1920, device=device))
    
    # classifier
    inputs.append(torch.randn(10, 1920, device=device))
    inputs.append(torch.zeros(10, device=device))
    
    # Running stats for features_1
    inputs.append(torch.zeros(64, device=device))   # mean
    inputs.append(torch.ones(64, device=device))    # var
    
    # Running stats for dense_blocks_0 layers 0-5
    in_ch = 64
    for i in range(6):
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.ones(in_ch, device=device))
        in_ch += 32
    
    # Running stats for transition_layers_0
    inputs.append(torch.zeros(256, device=device))
    inputs.append(torch.ones(256, device=device))
    
    # Running stats for dense_blocks_1 layers 0-11
    in_ch = 128
    for i in range(12):
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.ones(in_ch, device=device))
        in_ch += 32
    
    # Running stats for transition_layers_1
    inputs.append(torch.zeros(512, device=device))
    inputs.append(torch.ones(512, device=device))
    
    # Running stats for dense_blocks_2 layers 0-47
    in_ch = 256
    for i in range(48):
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.ones(in_ch, device=device))
        in_ch += 32
    
    # Running stats for transition_layers_2
    inputs.append(torch.zeros(1792, device=device))
    inputs.append(torch.ones(1792, device=device))
    
    # Running stats for dense_blocks_3 layers 0-31
    in_ch = 896
    for i in range(32):
        inputs.append(torch.zeros(in_ch, device=device))
        inputs.append(torch.ones(in_ch, device=device))
        in_ch += 32
    
    # Running stats for final_bn
    inputs.append(torch.zeros(1920, device=device))
    inputs.append(torch.ones(1920, device=device))
    
    # Input x
    inputs.append(torch.randn(10, 3, 224, 224, device=device))
    
    return inputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args_parsed = parser.parse_args()
    
    inputs = get_inputs()
    print(f"Number of inputs: {len(inputs)}")
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    print(f"Output shape: {output.shape}")
    
    if not args_parsed.profile_only:
        # Verify correctness against PyTorch reference
        import torch.nn as nn
        
        class DenseBlock(nn.Module):
            def __init__(self, num_layers, num_input_features, growth_rate):
                super().__init__()
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
        
        class TransitionLayer(nn.Module):
            def __init__(self, num_input_features, num_output_features):
                super().__init__()
                self.transition = nn.Sequential(
                    nn.BatchNorm2d(num_input_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
            
            def forward(self, x):
                return self.transition(x)
        
        class Model(nn.Module):
            def __init__(self, growth_rate=32, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
                
                num_features = 64
                block_config = [6, 12, 48, 32]
                
                self.dense_blocks = nn.ModuleList()
                self.transition_layers = nn.ModuleList()
                
                for i, num_layers in enumerate(block_config):
                    block = DenseBlock(num_layers, num_features, growth_rate)
                    self.dense_blocks.append(block)
                    num_features = num_features + num_layers * growth_rate
                    if i != len(block_config) - 1:
                        trans = TransitionLayer(num_features, num_features // 2)
                        self.transition_layers.append(trans)
                        num_features = num_features // 2