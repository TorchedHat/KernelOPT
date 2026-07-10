import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'OC', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IC, OC,
    IN_D, IN_H, IN_W,
    OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M, K,
    # X strides: (N, IC, IN_D, IN_H, IN_W)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # W strides: (IC, OC, KD, KH, KW)
    sw_ic, sw_oc, sw_kd, sw_kh, sw_kw,
    # Y strides: (N, OC, OUT_D, OUT_H, OUT_W)
    sy_n, sy_c, sy_d, sy_h, sy_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: n, od, oh, ow
    ow_idx = offs_m % OUT_W
    oh_idx = (offs_m // OUT_W) % OUT_H
    od_idx = (offs_m // (OUT_W * OUT_H)) % OUT_D
    n_idx  = offs_m // (OUT_W * OUT_H * OUT_D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index: ic, kd, kh, kw
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        kd_idx = (offs_k // (KW * KH)) % KD
        ic_idx = offs_k // (KW * KH * KD)

        # For ConvTranspose3d:
        # od = id * STRIDE_D - PAD_D + kd  =>  id = (od + PAD_D - kd) / STRIDE_D
        # Valid when (od + PAD_D - kd) % STRIDE_D == 0 and 0 <= id < IN_D

        od_plus_pad_d = od_idx[:, None] + PAD_D - kd_idx[None, :]
        oh_plus_pad_h = oh_idx[:, None] + PAD_H - kh_idx[None, :]
        ow_plus_pad_w = ow_idx[:, None] + PAD_W - kw_idx[None, :]

        id_idx = od_plus_pad_d // STRIDE_D
        ih_idx = oh_plus_pad_h // STRIDE_H
        iw_idx = ow_plus_pad_w // STRIDE_W

        valid_d = (od_plus_pad_d % STRIDE_D == 0) & (id_idx >= 0) & (id_idx < IN_D)
        valid_h = (oh_plus_pad_h % STRIDE_H == 0) & (ih_idx >= 0) & (ih_idx < IN_H)
        valid_w = (ow_plus_pad_w % STRIDE_W == 0) & (iw_idx >= 0) & (iw_idx < IN_W)
        valid_k = offs_k[None, :] < K
        valid_m = offs_m[:, None] < M

        in_bounds = valid_d & valid_h & valid_w & valid_k & valid_m

        # X[n, ic, id, ih, iw]
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=in_bounds, other=0.0)

        # W[ic, oc, kd, kh, kw]
        valid_n = offs_n[None, :] < OC
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & valid_n, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias[None, :]

    # Store Y[n, oc, od, oh, ow]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OC)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['spatial_size'],
)
@triton.jit
def bn_mean_sub_kernel(
    Y_ptr, Out_ptr,
    BN_weight_ptr, BN_bias_ptr,
    BN_mean_ptr, BN_var_ptr,
    N, C, spatial_size,
    eps,
    sy_n, sy_c, sy_s,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n_idx = pid // C
    c_idx = pid % C

    if n_idx >= N:
        return

    # Load BN params
    bn_mean = tl.load(BN_mean_ptr + c_idx).to(tl.float32)
    bn_var  = tl.load(BN_var_ptr  + c_idx).to(tl.float32)
    bn_w    = tl.load(BN_weight_ptr + c_idx).to(tl.float32)
    bn_b    = tl.load(BN_bias_ptr   + c_idx).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(bn_var + eps)

    base_ptr = Y_ptr + n_idx * sy_n + c_idx * sy_c

    # First pass: compute BN and accumulate mean
    spatial_sum = tl.zeros((), dtype=tl.float32)
    for s_start in range(0, spatial_size, BLOCK_SIZE):
        offs = s_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size
        val = tl.load(base_ptr + offs * sy_s, mask=mask, other=0.0).to(tl.float32)
        bn_val = (val - bn_mean) * inv_std * bn_w + bn_b
        spatial_sum += tl.sum(tl.where(mask, bn_val, 0.0), axis=0)

    spatial_mean = spatial_sum / spatial_size

    # Second pass: subtract mean and store
    out_base = Out_ptr + n_idx * sy_n + c_idx * sy_c
    for s_start in range(0, spatial_size, BLOCK_SIZE):
        offs = s_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size
        val = tl.load(base_ptr + offs * sy_s, mask=mask, other=0.0).to(tl.float32)
        bn_val = (val - bn_mean) * inv_std * bn_w + bn_b
        out_val = bn_val - spatial_mean
        tl.store(out_base + offs * sy_s, out_val.to(tl.float32), mask=mask)


def kernel_function(
    p_conv_transpose_weight,
    p_conv_transpose_bias,
    p_batch_norm_weight,
    p_batch_norm_bias,
    b_batch_norm_running_mean,
    b_batch_norm_running_var,
    x,
):
    N, IC, IN_D, IN_H, IN_W = x.shape
    # weight shape: (IC, OC, KD, KH, KW)
    IC_w, OC, KD, KH, KW = p_conv_transpose_weight.shape
    assert IC == IC_w

    STRIDE_D, STRIDE_H, STRIDE_W = 2, 2, 2
    PAD_D, PAD_H, PAD_W = 1, 1, 1

    OUT_D = (IN_D - 1) * STRIDE_D - 2 * PAD_D + KD
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW

    M = N * OUT_D * OUT_H * OUT_W
    K = IC * KD * KH * KW

    x_c = x.contiguous()
    w_c = p_conv_transpose_weight.contiguous()

    Y = torch.empty(N, OC, OUT_D, OUT_H, OUT_W, device=x.device, dtype=torch.float32)

    def cdiv(a, b): return (a + b - 1) // b

    grid_conv = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(OC, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid_conv](
        x_c, w_c, p_conv_transpose_bias, Y,
        N, IC, OC,
        IN_D, IN_H, IN_W,
        OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M, K,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3), x_c.stride(4),
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3), w_c.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
    )

    spatial_size = OUT_D * OUT_H * OUT_W
    Out = torch.empty_like(Y)

    # Y is (N, OC, OUT_D, OUT_H, OUT_W), treat spatial as flat
    # strides: sy_n, sy_c, sy_s where sy_s=1 (last dim)
    # We need to flatten spatial dims; Y is contiguous so stride along spatial is 1
    # sy_n = OC * spatial_size, sy_c = spatial_size, sy_s = 1

    grid_bn = (N * OC,)
    bn_mean_sub_kernel[grid_bn](
        Y, Out,
        p_batch_norm_weight, p_batch_norm_bias,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        N, OC, spatial_size,
        1e-5,
        Y.stride(0), Y.stride(1), 1,
    )

    return Out


def get_inputs():
    p_conv_transpose_weight = torch.rand(16, 32, 3, 3, 3)
    p_conv_transpose_bias   = torch.rand(32)
    p_batch_norm_weight     = torch.rand(32)
    p_batch_norm_bias       = torch.rand(32)
    b_batch_norm_running_mean = torch.rand(32)
    b_batch_norm_running_var  = torch.rand(32).abs() + 0.1
    x = torch.rand(16, 16, 16, 32, 32)
    return [
        p_conv_transpose_weight,
        p_conv_transpose_bias,
        p_batch_norm_weight,
        p_batch_norm_bias,
        b_batch_norm_running_mean,
        b_batch_norm_running_var,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_gpu = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_gpu)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch reference
        import torch.nn as nn

        w, b, bn_w, bn_b, bn_rm, bn_rv, x_cpu = inputs
        x_gpu = inputs_gpu[-1]

        conv_t = nn.ConvTranspose3d(16, 32, 3, stride=2, padding=1, bias=True)
        conv_t.weight.data = w
        conv_t.bias.data   = b
        bn = nn.BatchNorm3d(32)
        bn.weight.data = bn_w
        bn.bias.data   = bn_b
        bn.running_mean.data = bn_rm
        bn.running_var.data  = bn_rv
        bn.eval()
        conv_t.eval()

        with torch.no_grad():
            ref = conv_t(inputs_gpu[-1].cpu())
            ref = bn(ref)
            ref = ref - ref.mean(dim=(2,3,4), keepdim=True)

        ref_gpu = ref.cuda()
        max_diff = (output.cpu() - ref).abs().max().item()
        print(f"Max diff vs reference: {max_diff:.6f}")
        if max_diff < 1e-2:
            print("PASSED")
        else:
            print("FAILED")