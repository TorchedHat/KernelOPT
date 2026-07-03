import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    # X strides: (N, Ci, IH, IW)
    x_stride_n, x_stride_ci, x_stride_ih, x_stride_iw,
    # W strides: (Ci, Co_per_group, KH, KW)
    w_stride_ci, w_stride_co, w_stride_kh, w_stride_kw,
    # Y strides: (N, Co, OH, OW)
    y_stride_n, y_stride_co, y_stride_oh, y_stride_ow,
    # Dimensions
    BATCH, IN_C, OUT_C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    GROUPS, IN_C_PER_GROUP, OUT_C_PER_GROUP,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W  (output spatial positions)
    # N = OUT_C
    # K = IN_C * KH * KW  (but grouped, so effectively IN_C_PER_GROUP * KH * KW per group)
    # We tile over M and N, and loop over K (the input channels * kernel)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)
    
    # Determine group from output channel
    # offs_n indexes into OUT_C
    group_idx = offs_n // OUT_C_PER_GROUP  # shape [BLOCK_N]
    local_co = offs_n % OUT_C_PER_GROUP    # local output channel within group
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K dimension: IN_C_PER_GROUP * KH * KW
    # But each output channel belongs to a specific group, and different output channels
    # may belong to different groups. We need to handle this carefully.
    # 
    # For a given output channel co in group g:
    #   output[n, co, oh, ow] += sum over (ci_local, kh, kw):
    #     X[n, g*IN_C_PER_GROUP + ci_local, ih, iw] * W[g*IN_C_PER_GROUP + ci_local, local_co, kh, kw]
    #   where ih = (oh + PAD_H - kh*DIL_H) / STRIDE_H  (must be integer and in bounds)
    #         iw = (ow + PAD_W - kw*DIL_W) / STRIDE_W
    
    # Total K per group
    K_PER_GROUP = IN_C_PER_GROUP * KH * KW
    
    for k_off in range(0, K_PER_GROUP, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ci_local, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ci_local = offs_k // (KW * KH)
        
        # For each output position (oh, ow) and kernel (kh, kw):
        # ih = (oh + PAD_H - kh * DIL_H) / STRIDE_H  must be integer in [0, IN_H)
        # iw = (ow + PAD_W - kw * DIL_W) / STRIDE_W  must be integer in [0, IN_W)
        
        # oh = out_h_idx [BLOCK_M], kh = kh_idx [BLOCK_K]
        # ih numerator: oh + PAD_H - kh * DIL_H
        ih_num = out_h_idx[:, None] + PAD_H - kh_idx[None, :] * DIL_H  # [BLOCK_M, BLOCK_K]
        iw_num = out_w_idx[:, None] + PAD_W - kw_idx[None, :] * DIL_W  # [BLOCK_M, BLOCK_K]
        
        # Check divisibility and bounds
        ih_valid = (ih_num >= 0) & (ih_num % STRIDE_H == 0)
        iw_valid = (iw_num >= 0) & (iw_num % STRIDE_W == 0)
        
        ih = ih_num // STRIDE_H
        iw = iw_num // STRIDE_W
        
        ih_in_bounds = (ih >= 0) & (ih < IN_H)
        iw_in_bounds = (iw >= 0) & (iw < IN_W)
        
        valid_mk = ih_valid & iw_valid & ih_in_bounds & iw_in_bounds & \
                   (offs_m[:, None] < M) & (offs_k[None, :] < K_PER_GROUP) & \
                   (batch_idx[:, None] < BATCH)
        
        # For loading X: we need group info, but group depends on N (output channel)
        # This is the tricky part: different output channels belong to different groups
        # We need to handle this per output channel
        # 
        # Strategy: for each (m, k) pair, the group is determined by the output channel (n)
        # But we're computing a [BLOCK_M, BLOCK_N] tile, so group varies along N dimension
        # 
        # We'll compute X contribution separately for each group present in offs_n
        # Since BLOCK_N might span multiple groups, we need to be careful
        # 
        # Alternative: load X for each possible group and select based on group_idx
        # For simplicity with small number of groups, we can handle this by
        # computing the full ci (global) based on group
        
        # For the X load: ci_global = group_idx[n] * IN_C_PER_GROUP + ci_local[k]
        # X[batch, ci_global, ih, iw]
        # This varies with both M (for ih,iw,batch) and N (for group) and K (for ci_local, kh, kw)
        
        # We need to load X as [BLOCK_M, BLOCK_K] but it depends on group (from N dimension)
        # This means we can't do a simple tl.dot without handling groups
        
        # Approach: iterate over groups within this tile
        # For each group g present in offs_n, compute the contribution
        # Since GROUPS is small (4), we can loop over groups
        
        # Actually, let's restructure: for each group, compute partial matmul
        # But that changes the kernel structure significantly
        
        # Better approach: since group_idx varies with N, we load X with group=0
        # and adjust. But this doesn't work cleanly with tl.dot.
        
        # Let's use a different approach: load X[m, k] where ci_global is computed
        # assuming group 0, then for the weight load, use the actual group.
        # This won't work for X.
        
        # The cleanest solution for grouped conv_transpose: 
        # Loop over groups explicitly in the kernel, with each program handling one group
        # OR: restructure so that N dimension is OUT_C_PER_GROUP and add group as program_id(2)
        
        # For now, let's use atomic adds and have each program handle one group
        # We'll restructure the kernel below
        pass
    
    # Store (placeholder - actual implementation below)
    pass


# Better kernel: one program per (batch*OH*OW tile, group, out_c_per_group tile)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'OUT_C_PER_GROUP', 'K_PER_GROUP'],
)
@triton.jit
def conv_transpose2d_grouped_kernel(
    X_ptr, W_ptr, Y_ptr,
    x_stride_n, x_stride_ci, x_stride_ih, x_stride_iw,
    w_stride_ci, w_stride_co, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_co, y_stride_oh, y_stride_ow,
    BATCH, IN_C, OUT_C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    GROUPS: tl.constexpr, IN_C_PER_GROUP: tl.constexpr, OUT_C_PER_GROUP: tl.constexpr,
    M, K_PER_GROUP,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0): tile over M = BATCH * OUT_H * OUT_W
    # program_id(1): tile over OUT_C_PER_GROUP
    # program_id(2): group index
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # local output channel within group
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K_PER_GROUP, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ci_local, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ci_local = offs_k // (KW * KH)
        
        # For transposed conv: output[oh, ow] gets contribution from input[ih, iw]
        # where: oh = ih * STRIDE_H - PAD_H + kh * DIL_H
        # So: ih = (oh + PAD_H - kh * DIL_H) / STRIDE_H
        
        ih_num = out_h_idx[:, None] + PAD_H - kh_idx[None, :] * DIL_H
        iw_num = out_w_idx[:, None] + PAD_W - kw_idx[None, :] * DIL_W
        
        ih_div_ok = (ih_num % STRIDE_H == 0)
        iw_div_ok = (iw_num % STRIDE_W == 0)
        
        ih = ih_num // STRIDE_H
        iw = iw_num // STRIDE_W
        
        valid_mk = (ih_div_ok & iw_div_ok &
                    (ih >= 0) & (ih < IN_H) &
                    (iw >= 0) & (iw < IN_W) &
                    (offs_m[:, None] < M) &
                    (offs_k[None, :] < K_PER_GROUP) &
                    (batch_idx[:, None] < BATCH))
        
        # Global input channel
        ci_global = pid_g * IN_C_PER_GROUP + ci_local  # [BLOCK_K]
        
        # Load X: [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr +
                  batch_idx[:, None] * x_stride_n +
                  ci_global[None, :] * x_stride_ci +
        x_tile = tl.load(x_ptrs, mask=valid_mk, other=0.0)
        
        # Load W: [BLOCK_K, BLOCK_N]
        # W shape: [IN_C, OUT_C_PER_GROUP, KH, KW]
        # W[ci_global, local_co, kh, kw]
        w_ptrs = (W_ptr +
                  ci_global[:, None] * w_stride_ci +
                  offs_n[None, :] * w_stride_co +
                  kh_idx[:, None] * w_stride_kh +
                  kw_idx[:, None] * w_stride_kw)
        w_mask = ((offs_k[:, None] < K_PER_GROUP) &
                  (offs_n[None, :] < OUT_C_PER_GROUP) &
                  (ci_global[:, None] < IN_C))
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # We can't use tl.dot directly because x_tile has irregular valid pattern
        # Use element-wise multiply and sum
        # acc[m, n] += sum_k x_tile[m, k] * w_tile[k, n]
        # This is a matmul but with masked x_tile
        # For performance, use tl.dot with masked x_tile (zeros for invalid)
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    # Global output channel: pid_g * OUT_C_PER_GROUP + offs_n
    co_global = pid_g * OUT_C_PER_GROUP + offs_n  # [BLOCK_N]
    
    out_mask = ((offs_m[:, None] < M) &
                (offs_n[None, :] < OUT_C_PER_GROUP) &
                (co_global[None, :] < OUT_C) &
                (batch_idx[:, None] < BATCH))
    
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * y_stride_n +
              co_global[None, :] * y_stride_co +
              out_h_idx[:, None] * y_stride_oh +
              out_w_idx[:, None] * y_stride_ow)
    
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(weight, x):
    # weight: [in_channels, out_channels/groups, kH, kW] = [32, 16, 3, 5]
    # x: [batch, in_channels, in_h, in_w] = [16, 32, 128, 256]
    
    batch = x.shape[0]
    in_c = x.shape[1]
    in_h = x.shape[2]
    in_w = x.shape[3]
    
    kh = weight.shape[2]
    kw = weight.shape[3]
    out_c_per_group = weight.shape[1]
    
    # Parameters matching the model
    stride_h, stride_w = 2, 3
    pad_h, pad_w = 1, 2
    dil_h, dil_w = 2, 1
    groups = 4
    
    in_c_per_group = in_c // groups
    out_c = out_c_per_group * groups
    
    # Output size for transposed conv:
    # out_h = (in_h - 1) * stride_h - 2*pad_h + dil_h*(kh-1) + 1
    out_h = (in_h - 1) * stride_h - 2 * pad_h + dil_h * (kh - 1) + 1
    out_w = (in_w - 1) * stride_w - 2 * pad_w + dil_w * (kw - 1) + 1
    
    y = torch.zeros(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)
    
    M = batch * out_h * out_w
    K_per_group = in_c_per_group * kh * kw
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(out_c_per_group, meta['BLOCK_N']),
        groups,
    )
    
    conv_transpose2d_grouped_kernel[grid](
        x, weight, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        batch, in_c, out_c, in_h, in_w, out_h, out_w,
        kh, kw,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups, in_c_per_group, out_c_per_group,
        M, K_per_group,
    )
    
    return y


def get_inputs():
    # Weight shape: [in_channels, out_channels/groups, kH, kW]
    weight = torch.randn(32, 16, 3, 5, dtype=torch.float32, device='cuda')
    x = torch.randn(16, 32, 128, 256, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        weight_ref = inputs[0].clone()
        x_ref = inputs[1].clone()
        
        # Reference using PyTorch
        conv_t = nn.ConvTranspose2d(32, 64, (3, 5), stride=(2, 3), padding=(1, 2),
                                     dilation=(2, 1), groups=4, bias=False)
        conv_t.weight.data = weight_ref
        conv_t = conv_t.cuda()
        
        with torch.no_grad():
            ref_out = conv_t(x_ref)
        
        our_out = kernel_function(*inputs)
        
        print(f"Output shape: {our_out.shape}")
        print(f"Reference shape: {ref_out.shape}")
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        print(f"Match: {torch.allclose(our_out, ref_out, atol=1e-3)}")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")