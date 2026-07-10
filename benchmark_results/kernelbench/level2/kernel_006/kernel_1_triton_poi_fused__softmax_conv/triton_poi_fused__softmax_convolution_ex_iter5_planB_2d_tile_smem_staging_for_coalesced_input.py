import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
], key=['xnumel'])
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: program_id(0) over input-W (0..29), program_id(1) over outer N*C*D*H (0..215039)
    # xnumel=30 (input W), ynumel=215040 hardcoded (N*C*D*output_H = 128*16*7*15)
    YBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    iw = xoffset + tl.arange(0, XBLOCK)[None, :]    # input W index, shape [1, XBLOCK]
    yidx = yoffset + tl.arange(0, YBLOCK)[:, None]  # outer index, shape [YBLOCK, 1]
    xmask = iw < xnumel    # iw < 30
    ymask = yidx < 215040
    mask = xmask & ymask
    # Decompose yidx: ynumel = 128*16*7*15 = N*C*D*output_H
    x1 = (yidx % 15)           # output H (0..14)
    x2 = ((yidx // 15) % 7)    # output D (0..6)
    x3 = yidx // 105            # N*C index (0..2047)
    # iw is the input W index (0..29), stride-1 in global memory
    row_base = iw + 60*x1 + 1800*x2 + 12608*x3   # stride-1 in iw!
    # Load 8 kernel positions for max-pool 2x2x2 with stride 2
    tmp0  = tl.load(in_ptr0 + row_base,        mask=mask, eviction_policy='evict_last')
    tmp1  = tl.load(in_ptr0 + row_base + 30,   mask=mask, eviction_policy='evict_last')  # +1 in H
    tmp3  = tl.load(in_ptr0 + row_base + 900,  mask=mask, eviction_policy='evict_last')  # +1 in D
    tmp5  = tl.load(in_ptr0 + row_base + 930,  mask=mask, eviction_policy='evict_last')  # +1 in D+H
    # Load W+1 neighbor positions
    tmp_w1_0  = tl.load(in_ptr0 + row_base + 1,   mask=mask, eviction_policy='evict_last')
    tmp_w1_1  = tl.load(in_ptr0 + row_base + 31,  mask=mask, eviction_policy='evict_last')
    tmp_w1_2  = tl.load(in_ptr0 + row_base + 901, mask=mask, eviction_policy='evict_last')
    tmp_w1_3  = tl.load(in_ptr0 + row_base + 931, mask=mask, eviction_policy='evict_last')
    # Compute max over all 8 positions
    m0 = tl.maximum(tmp0, tmp_w1_0)
    m1 = tl.maximum(tmp1, tmp_w1_1)
    m2 = tl.maximum(tmp3, tmp_w1_2)
    m3 = tl.maximum(tmp5, tmp_w1_3)
    m4 = tl.maximum(m0, m1)
    m5 = tl.maximum(m2, m3)
    result = tl.maximum(m4, m5)
    # Only threads at even input-W positions write output (output_w = iw // 2)
    out_w = iw // 2   # output W index (0..14)
    out_idx = out_w + 15*x1 + 225*x2 + 1600*x3
    store_mask = mask & ((iw % 2) == 0)
    tl.store(out_ptr0 + out_idx, result, mask=store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3225600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1[grid](arg_0, arg_1, 3225600)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")
