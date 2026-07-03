import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=3),
    ],
    key=['numel'],
)
@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    numel,
    spatial_size,  # H * W = 512 * 512 = 262144
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Determine which channel each element belongs to
    # Layout: [N, C, H, W] with strides [C*H*W, H*W, W, 1]
    # channel index = (flat_index // spatial_size) % C
    channel_idx = (offsets // spatial_size) % 64

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)

def kernel_function(p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x):
    # x: [N, C, H, W]
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial_size = H * W
    eps = 1e-5

    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    batch_norm_inference_kernel[grid](
        x,
        b_bn_running_mean,
        b_bn_running_var,
        p_bn_weight,
        p_bn_bias,
        out,
        numel,
        spatial_size,
        eps,
    )

    return out


def get_inputs():
    batch_size = 64
    features = 64
    dim1 = 512
    dim2 = 512

    p_bn_weight = torch.ones(features, device='cuda', dtype=torch.float32)
    p_bn_bias = torch.zeros(features, device='cuda', dtype=torch.float32)
    b_bn_running_mean = torch.zeros(features, device='cuda', dtype=torch.float32)
    b_bn_running_var = torch.ones(features, device='cuda', dtype=torch.float32)
    x = torch.rand(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)

    return [p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        import torch.nn as nn
        p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x = inputs

        bn = nn.BatchNorm2d(64)
        bn.weight.data = p_bn_weight.cpu()
        bn.bias.data = p_bn_bias.cpu()
        bn.running_mean.data = b_bn_running_mean.cpu()
        bn.running_var.data = b_bn_running_var.cpu()
        bn.eval()

        ref = bn(x.cpu()).cuda()
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        if max_diff < 1e-3:
            print("Correctness PASSED")
        else:
            print("Correctness FAILED")
        print("Kernel executed successfully.")