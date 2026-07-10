import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_sub_tanh_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, pbias_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OH * OW (output spatial positions)
    # N_out = OC (output channels)
    # K = IC * KH * KW (reduction dimension)
    # ConvTranspose2d: output[n,oc,oh,ow] = sum_{ic,kh,kw} input[n,ic,ih,iw] * weight[ic,oc,kh,kw]
    # where ih = (oh + PAD_H - kh) / STRIDE_H (only when divisible and in bounds)
    #       iw = (ow + PAD_W - kw) / STRIDE_W (only when divisible and in bounds)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OW * OH)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (ic, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ic_idx = offs_k // (KH * KW)
        
        # For ConvTranspose2d: input position ih = (oh + PAD_H - kh) / STRIDE_H
        # This is valid only when (oh + PAD_H - kh) is divisible by STRIDE_H and in bounds
        oh_kh = oh[:, None] + PAD_H - kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
        ow_kw = ow[:, None] + PAD_W - kw_idx[None, :]  # [BLOCK_M, BLOCK_K]
        
        ih = oh_kh // STRIDE_H
        iw = ow_kw // STRIDE_W
        
        valid_h = (oh_kh % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
        valid_w = (ow_kw % STRIDE_W == 0) & (iw >= 0) & (iw < IW)
        valid_k = offs_k[None, :] < K
        valid_batch = batch[:, None] < N
        valid_oc = offs_n < OC
        
        valid_input = valid_h & valid_w & valid_k & valid_batch
        
        # Load input [BLOCK_M, BLOCK_K]
        # input shape: (N, IC, IH, IW), stride: (IC*IH*IW, IH*IW, IW, 1)
        input_ptrs = (input_ptr + 
                      batch[:, None] * (IC * IH * IW) +
                      ic_idx[None, :] * (IH * IW) +
        x_tile = tl.load(input_ptrs, mask=valid_input, other=0.0)
        
        # Load weight [BLOCK_K, BLOCK_N]
        # weight shape: (IC, OC, KH, KW), stride: (OC*KH*KW, KH*KW, KW, 1)
        weight_ptrs = (weight_ptr +
                       ic_idx[:, None] * (OC * KH * KW) +
                       offs_n[None, :] * (KH * KW) +
                       kh_idx[:, None] * KW +
                       kw_idx[:, None])
        valid_weight = (offs_k[:, None] < K) & (offs_n[None, :] < OC)
        w_tile = tl.load(weight_ptrs, mask=valid_weight, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add conv bias (per output channel)
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias_vals[None, :]
    
    # Subtract pbias (per output channel, shape [OC, 1, 1])
    pbias_vals = tl.load(pbias_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc - pbias_vals[None, :]
    
    # Apply tanh
    acc = tl.extra.libdevice.tanh(acc)
    
    # Store output
    # output shape: (N, OC, OH, OW)
    out_ptrs = (output_ptr +
                batch[:, None] * (OC * OH * OW) +
                offs_n[None, :] * (OH * OW) +
                oh[:, None] * OW +
                ow[:, None])
    
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OC)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x):
    N, IC, IH, IW = x.shape
    # weight shape for ConvTranspose2d: (in_channels, out_channels, kH, kW)
    IC_w, OC, KH, KW = p_conv_transpose_weight.shape
    
    STRIDE_H = 2
    STRIDE_W = 2
    PAD_H = 1
    PAD_W = 1
    OUTPUT_PAD_H = 1
    OUTPUT_PAD_W = 1
    
    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH + OUTPUT_PAD_H
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW + OUTPUT_PAD_W
    
    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    M = N * OH * OW
    N_out = OC
    K = IC * KH * KW
    
    # pbias shape: (OC, 1, 1) -> flatten to (OC,)
    pbias_flat = p_bias.reshape(OC)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    
    conv_transpose2d_sub_tanh_kernel[grid](
        output, x, p_conv_transpose_weight, p_conv_transpose_bias, pbias_flat,
        N, IC, IH, IW, OC, OH, OW,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, N_out, K,
    )
    
    return output


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 64, 4, 4, device='cuda')
    p_conv_transpose_bias = torch.randn(64, device='cuda')
    p_bias = torch.randn(64, 1, 1, device='cuda')
    x = torch.rand(32, 64, 256, 256, device='cuda')
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        # Verify against PyTorch reference
        import torch.nn as nn
        w, b, pb, x = inputs
        # Build reference
        conv_t = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=1).cuda()
        with torch.no_grad():
            conv_t.weight.copy_(w)
            conv_t.bias.copy_(b)
        ref = torch.tanh(conv_t(x) - pb)
        print(f"Max diff: {(output - ref).abs().max().item()}")
        print(f"Match: {torch.allclose(output, ref, atol=1e-3)}")
        print("Kernel executed successfully.")