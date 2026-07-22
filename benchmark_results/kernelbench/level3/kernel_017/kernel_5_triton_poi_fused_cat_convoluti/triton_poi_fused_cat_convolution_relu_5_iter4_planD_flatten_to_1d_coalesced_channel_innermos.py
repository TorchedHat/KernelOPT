import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8,  num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8,  num_stages=2, maxnreg=64),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 65536
    # New tiling strategy: XBLOCK=128 covers the full channel axis (y0 in [0,128)) as the
    # innermost fast-varying dimension (stride-1 in in_ptr0 — COALESCED).
    # YBLOCK covers the spatial-batch outer index as the slow-varying dimension.
    #
    # program_id(0) -> spatial-batch outer tile (sindex)
    # tl.arange(0, XBLOCK) -> y0 channel index (stride-1 in in_ptr0)
    soffset = tl.program_id(0) * YBLOCK
    sindex = soffset + tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1]
    # Total spatial-batch elements: xnumel * (ynumel // 128) = 65536 * 128 = 8388608
    snumel = xnumel * (ynumel // 128)
    smask = sindex < snumel                              # [YBLOCK, 1]

    # Channel index as innermost dimension (XBLOCK=128 covers full channel axis)
    cindex = tl.arange(0, XBLOCK)[None, :]              # [1, XBLOCK=128] — innermost (stride-1)
    cmask = cindex < 128                                 # always True since XBLOCK=128

    # Decompose sindex into spatial pixel (x2) and batch (y1)
    x2 = sindex % xnumel    # spatial pixel index (0..65535), shape [YBLOCK, 1]
    y1 = sindex // xnumel   # batch index (0..127), shape [YBLOCK, 1]

    # Channel index
    y0 = cindex              # shape [1, XBLOCK], values 0..127

    # y3 = y1 * 128 + y0 (the combined channel-batch index for output)
    y3 = y1 * 128 + y0      # shape [YBLOCK, XBLOCK]

    # Concatenation boundary: y0 < 64 -> in_ptr0, y0 >= 64 -> in_ptr1 + in_ptr2 bias
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = y0 < tmp3         # [1, XBLOCK] — first half mask

    # Load from in_ptr0 (stride-1 in y0 — COALESCED)
    # in_ptr0 layout: (8388608, 64) with strides (64, 1)
    # address: 64*x2 + 4194304*y1 + y0
    tmp5 = tl.load(in_ptr0 + (64*x2 + 4194304*y1 + y0), tmp4 & smask, eviction_policy='evict_first', other=0.0)
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = tl.maximum(tmp6, tmp5)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)

    tmp10 = y0 >= tmp3       # [1, XBLOCK] — second half mask

    # Load from in_ptr1 (stride-1 in y0-64 — COALESCED)
    # in_ptr1 layout: (128, 64, 256, 256) with strides (4194304, 1, 16384, 64)
    # address: 64*x2 + 4194304*y1 + (y0 - 64)
    tmp13 = tl.load(in_ptr1 + (64*x2 + 4194304*y1 + (y0 - 64)), tmp10 & smask, eviction_policy='evict_first', other=0.0)

    # Load bias from in_ptr2 (stride-1 in y0-64 — COALESCED)
    # in_ptr2 layout: (64,) with stride (1,)
    tmp14 = tl.load(in_ptr2 + (tl.broadcast_to(y0 - 64, [YBLOCK, XBLOCK])), tmp10, eviction_policy='evict_last', other=0.0)

    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp10, tmp17, tmp18)

    tmp20 = tl.where(tmp4, tmp9, tmp19)

    # Output: out_ptr0 layout (128, 128, 256, 256) with strides (8388608, 65536, 256, 1)
    # address: x2 + 65536 * y3 = x2 + 65536 * (y1*128 + y0)
    tl.store(out_ptr0 + (x2 + 65536 * y3), tmp20, smask & cmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 65536
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, 16384, 65536)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((8388608, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 256, 256), (8388608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
