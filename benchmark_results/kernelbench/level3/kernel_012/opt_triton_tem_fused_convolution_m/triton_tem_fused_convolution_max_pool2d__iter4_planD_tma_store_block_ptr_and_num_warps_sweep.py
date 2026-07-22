import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # num_stages=3: good pipelining, moderate shared memory
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        # num_stages=4: deeper pipelining, more shared memory
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # num_warps=8 configs for small BLOCK_M (new additions)
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # X has shape (M, K), strides (stride_xm, stride_xk)
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # W has shape (N, K), strides (stride_wn, stride_wk)
    # Traverse W as (K, N) by swapping strides to get (BLOCK_K, BLOCK_N) tiles
    w_block_ptr = tl.make_block_ptr(
        base=W_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc = tl.dot(x, w, acc)
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    out_block_ptr = tl.make_block_ptr(
        base=Out_ptr, shape=(M, N), strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(out_block_ptr, acc.to(tl.float32), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit

def bias_relu_kernel(
    X_ptr, B_ptr, Out_ptr,
    M, N,
    APPLY_RELU: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    col_block = pid % ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)

    offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < M) & (offs < N)

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=offs < N, other=0.0)
    out = x + b
    if APPLY_RELU:
        out = tl.maximum(out, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))
    tl.store(Out_ptr + row * N + offs, out, mask=mask)


def run_conv_relu(x, weight, bias, padding=1):
    out = torch.ops.aten.convolution(x, weight, None, [1, 1], [padding, padding], [1, 1], False, [0, 0], 1)
    N_elem = out.numel()
    C = out.shape[1]
    out_flat = out.view(-1)
    bias_expanded = bias.view(1, C, 1, 1).expand_as(out).contiguous().view(-1)
    result = torch.maximum(out_flat + bias_expanded, torch.zeros_like(out_flat))
    return result.view_as(out)


def run_linear_bias_relu(x, weight, bias, apply_relu=True):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        APPLY_RELU=apply_relu,
    )
    return out


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_16_weight, p_features_16_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_23_weight, p_features_23_bias,
    p_features_25_weight, p_features_25_bias,
    p_features_28_weight, p_features_28_bias,
    p_features_30_weight, p_features_30_bias,
    p_features_32_weight, p_features_32_bias,
    p_features_34_weight, p_features_34_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    def conv_relu(inp, w, b):
        out = torch.ops.aten.convolution(inp, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        return torch.relu(out)

    def maxpool(inp):
        return torch.nn.functional.max_pool2d(inp, kernel_size=2, stride=2)

    # Block 1
    x = conv_relu(x, p_features_0_weight, p_features_0_bias)
    x = conv_relu(x, p_features_2_weight, p_features_2_bias)
    x = maxpool(x)

    # Block 2
    x = conv_relu(x, p_features_5_weight, p_features_5_bias)
    x = conv_relu(x, p_features_7_weight, p_features_7_bias)
    x = maxpool(x)

    # Block 3
    x = conv_relu(x, p_features_10_weight, p_features_10_bias)
    x = conv_relu(x, p_features_12_weight, p_features_12_bias)
    x = conv_relu(x, p_features_14_weight, p_features_14_bias)
    x = conv_relu(x, p_features_16_weight, p_features_16_bias)
    x = maxpool(x)

    # Block 4
    x = conv_relu(x, p_features_19_weight, p_features_19_bias)
    x = conv_relu(x, p_features_21_weight, p_features_21_bias)
    x = conv_relu(x, p_features_23_weight, p_features_23_bias)
    x = conv_relu(x, p_features_25_weight, p_features_25_bias)
    x = maxpool(x)

    # Block 5
    x = conv_relu(x, p_features_28_weight, p_features_28_bias)
    x = conv_relu(x, p_features_30_weight, p_features_30_bias)
    x = conv_relu(x, p_features_32_weight, p_features_32_bias)
    x = conv_relu(x, p_features_34_weight, p_features_34_bias)
    x = maxpool(x)

    # Flatten
    x = x.flatten(1)

    # Classifier - use Triton fused linear+bias+relu
    x = run_linear_bias_relu(x, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    x = run_linear_bias_relu(x, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    x = run_linear_bias_relu(x, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)

    return x


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'

    p_features_0_weight = torch.randn(64, 3, 3, 3, device=device)
    p_features_0_bias = torch.randn(64, device=device)
    p_features_2_weight = torch.randn(64, 64, 3, 3, device=device)
    p_features_2_bias = torch.randn(64, device=device)
    p_features_5_weight = torch.randn(128, 64, 3, 3, device=device)
    p_features_5_bias = torch.randn(128, device=device)
    p_features_7_weight = torch.randn(128, 128, 3, 3, device=device)
    p_features_7_bias = torch.randn(128, device=device)
    p_features_10_weight = torch.randn(256, 128, 3, 3, device=device)
    p_features_10_bias = torch.randn(256, device=device)
    p_features_12_weight = torch.randn(256, 256, 3, 3, device=device)
    p_features_12_bias = torch.randn(256, device=device)
    p_features_14_weight = torch.randn(256, 256, 3, 3, device=device)
    p_features_14_bias = torch.randn(256, device=device)
    p_features_16_weight = torch.randn(256, 256, 3, 3, device=device)
    p_features_16_bias = torch.randn(256, device=device)
    p_features_19_weight = torch.randn(512, 256, 3, 3, device=device)
    p_features_19_bias = torch.randn(512, device=device)
    p_features_21_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_21_bias = torch.randn(512, device=device)
    p_features_23_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_23_bias = torch.randn(512, device=device)
    p_features_25_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_25_bias = torch.randn(512, device=device)
    p_features_28_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_28_bias = torch.randn(512, device=device)
    p_features_30_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_30_bias = torch.randn(512, device=device)
    p_features_32_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_32_bias = torch.randn(512, device=device)
    p_features_34_weight = torch.randn(512, 512, 3, 3, device=device)
    p_features_34_bias = torch.randn(512, device=device)
    p_classifier_0_weight = torch.randn(4096, 25088, device=device)
    p_classifier_0_bias = torch.randn(4096, device=device)
    p_classifier_3_weight = torch.randn(4096, 4096, device=device)
    p_classifier_3_bias = torch.randn(4096, device=device)
    p_classifier_6_weight = torch.randn(1000, 4096, device=device)
    p_classifier_6_bias = torch.randn(1000, device=device)
    x = torch.randn(10, 3, 224, 224, device=device)

    return [
        p_features_0_weight, p_features_0_bias,
        p_features_2_weight, p_features_2_bias,
        p_features_5_weight, p_features_5_bias,
        p_features_7_weight, p_features_7_bias,
        p_features_10_weight, p_features_10_bias,
        p_features_12_weight, p_features_12_bias,
        p_features_14_weight, p_features_14_bias,
        p_features_16_weight, p_features_16_bias,
        p_features_19_weight, p_features_19_bias,
        p_features_21_weight, p_features_21_bias,
        p_features_23_weight, p_features_23_bias,
        p_features_25_weight, p_features_25_bias,
        p_features_28_weight, p_features_28_bias,
        p_features_30_weight, p_features_30_bias,
        p_features_32_weight, p_features_32_bias,
        p_features_34_weight, p_features_34_bias,
        p_classifier_0_weight, p_classifier_0_bias,
        p_classifier_3_weight, p_classifier_3_bias,
        p_classifier_6_weight, p_classifier_6_bias,
        x,
    ]


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

        class VGG19Ref(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 1000),
                )

        (
            p_features_0_weight, p_features_0_bias,
            p_features_2_weight, p_features_2_bias,
            p_features_5_weight, p_features_5_bias,
            p_features_7_weight, p_features_7_bias,
            p_features_10_weight, p_features_10_bias,
            p_features_12_weight, p_features_12_bias,
            p_features_14_weight, p_features_14_bias,
            p_features_16_weight, p_features_16_bias,
            p_features_19_weight, p_features_19_bias,
            p_features_21_weight, p_features_21_bias,
            p_features_23_weight, p_features_23_bias,
            p_features_25_weight, p_features_25_bias,
            p_features_28_weight, p_features_28_bias,
            p_features_30_weight, p_features_30_bias,
            p_features_32_weight, p_features_32_bias,
            p_features_34_weight, p_features_34_bias,
            p_classifier_0_weight, p_classifier_0_bias,
            p_classifier_3_weight, p_classifier_3_bias,
            p_classifier_6_weight, p_classifier_6_bias,
            x_inp,
        ) = inputs

        model = VGG19Ref().cuda().eval()
        feat_weights = [
            p_features_0_weight, p_features_2_weight,
            p_features_5_weight, p_features_7_weight,
            p_features_10_weight, p_features_12_weight, p_features_14_weight, p_features_16_weight,
            p_features_19_weight, p_features_21_weight, p_features_23_weight, p_features_25_weight,
            p_features_28_weight, p_features_30_weight, p_features_32_weight, p_features_34_weight,
        ]
        feat_biases = [
            p_features_0_bias, p_features_2_bias,
            p_features_5_bias, p_features_7_bias,
            p_features_10_bias, p_features_12_bias, p_features_14_bias, p_features_16_bias,
            p_features_19_bias, p_features_21_bias, p_features_23_bias, p_features_25_bias,
            p_features_28_bias, p_features_30_bias, p_features_32_bias, p_features_34_bias,
        ]
        conv_idx = 0
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.copy_(feat_weights[conv_idx])
                layer.bias.data.copy_(feat_biases[conv_idx])
                conv_idx += 1

        cls_weights = [p_classifier_0_weight, p_classifier_3_weight, p_classifier_6_weight]
        cls_biases = [p_classifier_0_bias, p_classifier_3_bias, p_classifier_6_bias]
        lin_idx = 0
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.copy_(cls_weights[lin_idx])
                layer.bias.data.copy_(cls_biases[lin_idx])
                lin_idx += 1

        with torch.no_grad():
            ref = model(x_inp)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")