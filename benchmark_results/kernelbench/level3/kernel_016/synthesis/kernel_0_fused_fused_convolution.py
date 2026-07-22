import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_channels', 'spatial'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, spatial, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    # Layout: (N, C, H, W) -> flat index = n*C*spatial + c*spatial + hw
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_channels * spatial
    # We need to figure out which channel each element belongs to
    # flat index within (C, spatial) part
    # full index = batch * C * spatial + c * spatial + hw
    # We process all batches together
    mask = offsets < tl.num_programs(0) * BLOCK_SIZE
    # Actually let's use a different approach - process per (batch*spatial) tile per channel
    # Better: flat 1D over all elements, derive channel from index
    pass

# Better approach: one program per (batch, channel) pair, iterate over spatial
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4),
        triton.Config({'BLOCK_S': 512}, num_warps=4),
        triton.Config({'BLOCK_S': 256}, num_warps=2),
        triton.Config({'BLOCK_S': 128}, num_warps=2),
    ],
    key=['spatial'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, spatial,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program_id(0) = batch index
    # program_id(1) = channel index  
    # program_id(2) = spatial tile
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < spatial
    
    # Load BN params for this channel
    mean = tl.load(mean_ptr + pid_c)
    var = tl.load(var_ptr + pid_c)
    weight = tl.load(weight_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    
    # Compute base pointer for this (n, c) slice
    base = pid_n * C * spatial + pid_c * spatial
    
    x = tl.load(x_ptr + base + offs_s, mask=mask_s, other=0.0)
    
    # BN inference: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * weight + bias
    
    # ReLU
    out = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + base + offs_s, out, mask=mask_s)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    spatial = H * W
    out = torch.empty_like(x)
    
    # Make contiguous
    x = x.contiguous()
    
    grid = lambda meta: (N, C, triton.cdiv(spatial, meta['BLOCK_S']))
    fused_bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        N, C, spatial,
        eps,
    )
    return out


def densenet_forward(
    x,
    # Initial conv + BN params
    features_conv_weight,
    features_bn_weight, features_bn_bias,
    features_bn_mean, features_bn_var,
    # Dense block 0 params: list of (bn_w, bn_b, bn_mean, bn_var, conv_w) per layer
    db0_params,
    # Transition 0 params
    tr0_bn_w, tr0_bn_b, tr0_bn_mean, tr0_bn_var, tr0_conv_w,
    # Dense block 1 params
    db1_params,
    # Transition 1 params
    tr1_bn_w, tr1_bn_b, tr1_bn_mean, tr1_bn_var, tr1_conv_w,
    # Dense block 2 params
    db2_params,
    # Transition 2 params
    tr2_bn_w, tr2_bn_b, tr2_bn_mean, tr2_bn_var, tr2_conv_w,
    # Dense block 3 params
    db3_params,
    # Final BN
    final_bn_w, final_bn_b, final_bn_mean, final_bn_var,
    # Classifier
    classifier_w, classifier_b,
):
    # Initial features
    x = F.conv2d(x, features_conv_weight, bias=None, stride=2, padding=3)
    x = fused_bn_relu(x, features_bn_mean, features_bn_var, features_bn_weight, features_bn_bias)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Dense block 0
    features = [x]
    for (bn_w, bn_b, bn_mean, bn_var, conv_w) in db0_params:
        inp = torch.cat(features, dim=1)
        out = fused_bn_relu(inp, bn_mean, bn_var, bn_w, bn_b)
        out = F.conv2d(out, conv_w, bias=None, padding=1)
        # dropout with p=0.0 is identity
        features.append(out)
    x = torch.cat(features, dim=1)
    
    # Transition 0
    x = fused_bn_relu(x, tr0_bn_mean, tr0_bn_var, tr0_bn_w, tr0_bn_b)
    x = F.conv2d(x, tr0_conv_w, bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    
    # Dense block 1
    features = [x]
    for (bn_w, bn_b, bn_mean, bn_var, conv_w) in db1_params:
        inp = torch.cat(features, dim=1)
        out = fused_bn_relu(inp, bn_mean, bn_var, bn_w, bn_b)
        out = F.conv2d(out, conv_w, bias=None, padding=1)
        features.append(out)
    x = torch.cat(features, dim=1)
    
    # Transition 1
    x = fused_bn_relu(x, tr1_bn_mean, tr1_bn_var, tr1_bn_w, tr1_bn_b)
    x = F.conv2d(x, tr1_conv_w, bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    
    # Dense block 2
    features = [x]
    for (bn_w, bn_b, bn_mean, bn_var, conv_w) in db2_params:
        inp = torch.cat(features, dim=1)
        out = fused_bn_relu(inp, bn_mean, bn_var, bn_w, bn_b)
        out = F.conv2d(out, conv_w, bias=None, padding=1)
        features.append(out)
    x = torch.cat(features, dim=1)
    
    # Transition 2
    x = fused_bn_relu(x, tr2_bn_mean, tr2_bn_var, tr2_bn_w, tr2_bn_b)
    x = F.conv2d(x, tr2_conv_w, bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    
    # Dense block 3
    features = [x]
    for (bn_w, bn_b, bn_mean, bn_var, conv_w) in db3_params:
        inp = torch.cat(features, dim=1)
        out = fused_bn_relu(inp, bn_mean, bn_var, bn_w, bn_b)
        out = F.conv2d(out, conv_w, bias=None, padding=1)
        features.append(out)
    x = torch.cat(features, dim=1)
    
    # Final BN + ReLU
    x = fused_bn_relu(x, final_bn_mean, final_bn_var, final_bn_w, final_bn_b)
    
    # Global average pooling
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    
    # Classifier - use matrix multiply
    x = x @ classifier_w.t() + classifier_b
    
    return x


def kernel_function(*args):
    # Unpack all arguments according to the specification
    idx = 0
    p_features_0_weight = args[idx]; idx += 1
    p_features_1_weight = args[idx]; idx += 1
    p_features_1_bias = args[idx]; idx += 1
    
    # Dense block 0: 6 layers
    db0_raw = []
    for i in range(6):
        bn_w = args[idx]; idx += 1
        bn_b = args[idx]; idx += 1
        conv_w = args[idx]; idx += 1
        db0_raw.append((bn_w, bn_b, conv_w))
    
    # Transition 0
    tr0_bn_w = args[idx]; idx += 1
    tr0_bn_b = args[idx]; idx += 1
    tr0_conv_w = args[idx]; idx += 1
    
    # Dense block 1: 12 layers
    db1_raw = []
    for i in range(12):
        bn_w = args[idx]; idx += 1
        bn_b = args[idx]; idx += 1
        conv_w = args[idx]; idx += 1
        db1_raw.append((bn_w, bn_b, conv_w))
    
    # Transition 1
    tr1_bn_w = args[idx]; idx += 1
    tr1_bn_b = args[idx]; idx += 1
    tr1_conv_w = args[idx]; idx += 1
    
    # Dense block 2: 48 layers
    db2_raw = []
    for i in range(48):
        bn_w = args[idx]; idx += 1
        bn_b = args[idx]; idx += 1
        conv_w = args[idx]; idx += 1
        db2_raw.append((bn_w, bn_b, conv_w))
    
    # Transition 2
    tr2_bn_w = args[idx]; idx += 1
    tr2_bn_b = args[idx]; idx += 1
    tr2_conv_w = args[idx]; idx += 1
    
    # Dense block 3: 32 layers
    db3_raw = []
    for i in range(32):
        bn_w = args[idx]; idx += 1
        bn_b = args[idx]; idx += 1
        conv_w = args[idx]; idx += 1
        db3_raw.append((bn_w, bn_b, conv_w))
    
    # Final BN
    p_final_bn_weight = args[idx]; idx += 1
    p_final_bn_bias = args[idx]; idx += 1
    
    # Classifier
    p_classifier_weight = args[idx]; idx += 1
    p_classifier_bias = args[idx]; idx += 1
    
    # Running stats
    b_features_1_running_mean = args[idx]; idx += 1
    b_features_1_running_var = args[idx]; idx += 1
    
    # DB0 running stats
    db0_means = []
    db0_vars = []
    for i in range(6):
        db0_means.append(args[idx]); idx += 1
        db0_vars.append(args[idx]); idx += 1
    
    # TR0 running stats
    tr0_bn_mean = args[idx]; idx += 1
    tr0_bn_var = args[idx]; idx += 1
    
    # DB1 running stats
    db1_means = []
    db1_vars = []
    for i in range(12):
        db1_means.append(args[idx]); idx += 1
        db1_vars.append(args[idx]); idx += 1
    
    # TR1 running stats
    tr1_bn_mean = args[idx]; idx += 1
    tr1_bn_var = args[idx]; idx += 1
    
    # DB2 running stats
    db2_means = []
    db2_vars = []
    for i in range(48):
        db2_means.append(args[idx]); idx += 1
        db2_vars.append(args[idx]); idx += 1
    
    # TR2 running stats
    tr2_bn_mean = args[idx]; idx += 1
    tr2_bn_var = args[idx]; idx += 1
    
    # DB3 running stats
    db3_means = []
    db3_vars = []
    for i in range(32):
        db3_means.append(args[idx]); idx += 1
        db3_vars.append(args[idx]); idx += 1
    
    # Final BN running stats
    b_final_bn_running_mean = args[idx]; idx += 1
    b_final_bn_running_var = args[idx]; idx += 1
    
    # Input
    x = args[idx]; idx += 1
    
    # Build param lists with running stats
    db0_params = [
        (db0_raw[i][0], db0_raw[i][1], db0_means[i], db0_vars[i], db0_raw[i][2])
        for i in range(6)
    ]
    db1_params = [
        (db1_raw[i][0], db1_raw[i][1], db1_means[i], db1_vars[i], db1_raw[i][2])
        for i in range(12)
    ]
    db2_params = [
        (db2_raw[i][0], db2_raw[i][1], db2_means[i], db2_vars[i], db2_raw[i][2])
        for i in range(48)
    ]
    db3_params = [
        (db3_raw[i][0], db3_raw[i][1], db3_means[i], db3_vars[i], db3_raw[i][2])
        for i in range(32)
    ]
    
    return densenet_forward(
        x,
        p_features_0_weight,
        p_features_1_weight, p_features_1_bias,
        b_features_1_running_mean, b_features_1_running_var,
        db0_params,
        tr0_bn_w, tr0_bn_b, tr0_bn_mean, tr0_bn_var, tr0_conv_w,
        db1_params,
        tr1_bn_w, tr1_bn_b, tr1_bn_mean, tr1_bn_var, tr1_conv_w,
        db2_params,
        tr2_bn_w, tr2_bn_b, tr2_bn_mean, tr2_bn_var, tr2_conv_w,
        db3_params,
        p_final_bn_weight, p_final_bn_bias, b_final_bn_running_mean, b_final_bn_running_var,
        p_classifier_weight, p_classifier_bias,
    )


def get_inputs():
    # Create all the inputs matching the specification
    inputs = []
    
    # Initial conv weight
    inputs.append(torch.randn(64, 3, 7, 7, device='cuda'))
    # Initial BN weight, bias
    inputs.append(torch.ones(64, device='cuda'))
    inputs.append(torch.zeros(64, device='cuda'))
    
    # Dense block 0: 6 layers, starting channels: 64, growth=32
    # Layer i has input channels: 64 + i*32
    in_ch = 64
    for i in range(6):
        inputs.append(torch.ones(in_ch, device='cuda'))   # bn weight
        inputs.append(torch.zeros(in_ch, device='cuda'))  # bn bias
        inputs.append(torch.randn(32, in_ch, 3, 3, device='cuda'))  # conv weight
        in_ch += 32
    # in_ch = 256
    
    # Transition 0: 256 -> 128
    inputs.append(torch.ones(256, device='cuda'))   # bn weight
    inputs.append(torch.zeros(256, device='cuda'))  # bn bias
    inputs.append(torch.randn(128, 256, 1, 1, device='cuda'))  # conv weight
    
    # Dense block 1: 12 layers, starting channels: 128
    in_ch = 128
    for i in range(12):
        inputs.append(torch.ones(in_ch, device='cuda'))
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.randn(32, in_ch, 3, 3, device='cuda'))
        in_ch += 32
    # in_ch = 512
    
    # Transition 1: 512 -> 256
    inputs.append(torch.ones(512, device='cuda'))
    inputs.append(torch.zeros(512, device='cuda'))
    inputs.append(torch.randn(256, 512, 1, 1, device='cuda'))
    
    # Dense block 2: 48 layers, starting channels: 256
    in_ch = 256
    for i in range(48):
        inputs.append(torch.ones(in_ch, device='cuda'))
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.randn(32, in_ch, 3, 3, device='cuda'))
        in_ch += 32
    # in_ch = 1792
    
    # Transition 2: 1792 -> 896
    inputs.append(torch.ones(1792, device='cuda'))
    inputs.append(torch.zeros(1792, device='cuda'))
    inputs.append(torch.randn(896, 1792, 1, 1, device='cuda'))
    
    # Dense block 3: 32 layers, starting channels: 896
    in_ch = 896
    for i in range(32):
        inputs.append(torch.ones(in_ch, device='cuda'))
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.randn(32, in_ch, 3, 3, device='cuda'))
        in_ch += 32
    # in_ch = 1920
    
    # Final BN
    inputs.append(torch.ones(1920, device='cuda'))
    inputs.append(torch.zeros(1920, device='cuda'))
    
    # Classifier
    inputs.append(torch.randn(10, 1920, device='cuda'))
    inputs.append(torch.zeros(10, device='cuda'))
    
    # Running stats for initial BN
    inputs.append(torch.zeros(64, device='cuda'))   # mean
    inputs.append(torch.ones(64, device='cuda'))    # var
    
    # DB0 running stats
    in_ch = 64
    for i in range(6):
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.ones(in_ch, device='cuda'))
        in_ch += 32
    
    # TR0 running stats
    inputs.append(torch.zeros(256, device='cuda'))
    inputs.append(torch.ones(256, device='cuda'))
    
    # DB1 running stats
    in_ch = 128
    for i in range(12):
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.ones(in_ch, device='cuda'))
        in_ch += 32
    
    # TR1 running stats
    inputs.append(torch.zeros(512, device='cuda'))
    inputs.append(torch.ones(512, device='cuda'))
    
    # DB2 running stats
    in_ch = 256
    for i in range(48):
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.ones(in_ch, device='cuda'))
        in_ch += 32
    
    # TR2 running stats
    inputs.append(torch.zeros(1792, device='cuda'))
    inputs.append(torch.ones(1792, device='cuda'))
    
    # DB3 running stats
    in_ch = 896
    for i in range(32):
        inputs.append(torch.zeros(in_ch, device='cuda'))
        inputs.append(torch.ones(in_ch, device='cuda'))
        in_ch += 32
    
    # Final BN running stats
    inputs.append(torch.zeros(1920, device='cuda'))
    inputs.append(torch.ones(1920, device='cuda'))
    
    # Input image
    inputs.append(torch.randn(10, 3, 224, 224, device='cuda'))
    
    return inputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args_parsed = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args_parsed.profile_only:
        # Verify correctness against PyTorch reference
        import torch.nn as nn
        
        # Rebuild the model for reference
        class DenseLayer(nn.Module):
            def __init__(self, in_features, growth_rate):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, 3, padding=1, bias=False),
                    nn.Dropout(0.0)
                )
            def forward(self, x):
                return self.seq(x)
        
        class DenseBlock(nn.Module):
            def __init__(self, num_layers, in_features, growth_rate):
                super().__init__()
                self.layers = nn.ModuleList([
                    DenseLayer(in_features + i * growth_rate, growth_rate)
                    for i in range(num_layers)
                ])
            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feat = layer(torch.cat(features, 1))
                    features.append(new_feat)
                return torch.cat(features, 1)
        
        class TransitionLayer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.transition = nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, out_features, 1, bias=False),
                    nn.AvgPool2d(2, 2)
                )
            def forward(self, x):
                return self.transition(x)
        
        class RefModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
                self.db0 = DenseBlock(6, 64, 32)
                self.tr0 = TransitionLayer(256, 128)
                self.db1 = DenseBlock(12, 128, 32)
                self.tr1 = TransitionLayer(512, 256)
                self.db2 = DenseBlock(48, 256, 32)
                self.tr2 = TransitionLayer(1792, 896)
                self.db3 = DenseBlock(32, 896, 32)
                self.final_bn = nn.BatchNorm2d(1920)
                self.classifier = nn.Linear(1920, 10)
            
            def forward(self, x):
                x = self.features(x)
                x = self.db0(x)
                x = self.tr0(x)
                x = self.db1(x)
                x = self.tr1(x)
                x = self.db2(x)
                x = self.tr2(x)
                x = self.db3(x)
                x = F.relu(self.final_bn(x), inplace=True)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = RefModel().cuda().eval()
        
        # Load weights from inputs
        inp = inputs
        idx = 0
        
        with torch.no_grad():
            model.features[0].weight.copy_(inp[idx]); idx += 1
            model.features[1].weight.copy_(inp[idx]); idx += 1
            model.features[1].bias.copy_(inp[idx]); idx += 1
            
            in_ch = 64
            for i in range(6):
                model.db0.layers[i].seq[0].weight.copy_(inp[idx]); idx += 1
                model.db0.layers[i].seq[0].bias.copy_(inp[idx]); idx += 1
                model.db0.layers[i].seq[2].weight.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.tr0.transition[0].weight.copy_(inp[idx]); idx += 1
            model.tr0.transition[0].bias.copy_(inp[idx]); idx += 1
            model.tr0.transition[2].weight.copy_(inp[idx]); idx += 1
            
            in_ch = 128
            for i in range(12):
                model.db1.layers[i].seq[0].weight.copy_(inp[idx]); idx += 1
                model.db1.layers[i].seq[0].bias.copy_(inp[idx]); idx += 1
                model.db1.layers[i].seq[2].weight.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.tr1.transition[0].weight.copy_(inp[idx]); idx += 1
            model.tr1.transition[0].bias.copy_(inp[idx]); idx += 1
            model.tr1.transition[2].weight.copy_(inp[idx]); idx += 1
            
            in_ch = 256
            for i in range(48):
                model.db2.layers[i].seq[0].weight.copy_(inp[idx]); idx += 1
                model.db2.layers[i].seq[0].bias.copy_(inp[idx]); idx += 1
                model.db2.layers[i].seq[2].weight.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.tr2.transition[0].weight.copy_(inp[idx]); idx += 1
            model.tr2.transition[0].bias.copy_(inp[idx]); idx += 1
            model.tr2.transition[2].weight.copy_(inp[idx]); idx += 1
            
            in_ch = 896
            for i in range(32):
                model.db3.layers[i].seq[0].weight.copy_(inp[idx]); idx += 1
                model.db3.layers[i].seq[0].bias.copy_(inp[idx]); idx += 1
                model.db3.layers[i].seq[2].weight.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.final_bn.weight.copy_(inp[idx]); idx += 1
            model.final_bn.bias.copy_(inp[idx]); idx += 1
            model.classifier.weight.copy_(inp[idx]); idx += 1
            model.classifier.bias.copy_(inp[idx]); idx += 1
            
            # Running stats
            model.features[1].running_mean.copy_(inp[idx]); idx += 1
            model.features[1].running_var.copy_(inp[idx]); idx += 1
            
            in_ch = 64
            for i in range(6):
                model.db0.layers[i].seq[0].running_mean.copy_(inp[idx]); idx += 1
                model.db0.layers[i].seq[0].running_var.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.tr0.transition[0].running_mean.copy_(inp[idx]); idx += 1
            model.tr0.transition[0].running_var.copy_(inp[idx]); idx += 1
            
            in_ch = 128
            for i in range(12):
                model.db1.layers[i].seq[0].running_mean.copy_(inp[idx]); idx += 1
                model.db1.layers[i].seq[0].running_var.copy_(inp[idx]); idx += 1
                in_ch += 32
            
            model.tr1.transition[0].running_mean.copy_(inp[idx]); idx += 1
            model.tr1.transition[0].running_var.copy_(inp