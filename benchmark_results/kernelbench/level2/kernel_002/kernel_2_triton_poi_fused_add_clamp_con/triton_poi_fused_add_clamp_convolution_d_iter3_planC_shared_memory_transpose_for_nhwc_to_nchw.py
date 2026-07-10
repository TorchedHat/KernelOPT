import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 64}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_clamp_convolution_div_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 8192 (= 128 batches * 64 channels)
    # xnumel = 65536 (= 256*256 spatial)
    # in_ptr0 strides: (4194304, 1, 16384, 64) -- NHWC channels-last
    #   flat index: c + 64*s + 4194304*n  where s = h*256+w, c in [0,64)
    # out_ptr0 strides: (4194304, 65536, 256, 1) -- NCHW
    #   flat index: s + 65536*(c + 64*n)
    #
    # New tiling strategy:
    #   XBLOCK = SBLOCK: spatial tile size (program_id(0) iterates over spatial blocks)
    #   YBLOCK = 64 (fixed): covers all 64 channels (stride-1 in in_ptr0 -> coalesced load)
    #   program_id(1) = batch index n (0..127)
    #
    # Layout: [XBLOCK, YBLOCK] = [spatial, channel]
    #   - channel (axis 1, fast warp dim) is stride-1 in in_ptr0 -> coalesced loads
    #   - spatial (axis 0) is stride-1 in out_ptr0 -> coalesced stores
    ynumel = 8192
    xnumel = 65536

    s_pid = tl.program_id(0)   # spatial block index
    n_pid = tl.program_id(1)   # batch index (0..127)

    # Spatial range: XBLOCK elements per block
    s_range = tl.arange(0, XBLOCK)          # [XBLOCK]
    s_base  = s_pid * XBLOCK
    s_idx   = s_base + s_range               # actual spatial indices [XBLOCK]
    s_mask  = s_idx < xnumel                 # [XBLOCK]

    # Channel range: always all 64 channels (YBLOCK=64 fixed)
    c_range = tl.arange(0, YBLOCK)           # [YBLOCK=64], values 0..63

    # 2D index arrays: [XBLOCK, YBLOCK] = [spatial, channel]
    s_idx2d = s_idx[:, None]                 # [XBLOCK, 1]
    c_idx2d = c_range[None, :]               # [1, YBLOCK]
    mask2d  = s_mask[:, None]                # [XBLOCK, 1] broadcasts to [XBLOCK, YBLOCK]

    # Load from in_ptr0 (NHWC): index = c + 64*s + 4194304*n
    # c is axis 1 (fast warp dim, stride-1 in in_ptr0) -> coalesced load
    in0_idx = c_idx2d + 64 * s_idx2d + 4194304 * n_pid   # [XBLOCK, YBLOCK]
    tmp0 = tl.load(in_ptr0 + in0_idx, mask=mask2d, other=0.0,
                   eviction_policy='evict_last')

    # Load bias in_ptr1: shape [64], index = c (channel)
    tmp1 = tl.load(in_ptr1 + c_idx2d, mask=None, other=0.0)   # [1, YBLOCK]

    # Load bias in_ptr2: shape [64,1,1] strides (1,1,1), index = c
    tmp3 = tl.load(in_ptr2 + c_idx2d, mask=None, other=0.0)   # [1, YBLOCK]

    # Pointwise ops (identical to original kernel)
    tmp2  = tmp0 + tmp1
    tmp4  = tmp2 + tmp3
    tmp5  = tl.full([1, 1], 0.0, tl.float32)
    tmp6  = tl.maximum(tmp4, tmp5)
    tmp7  = tl.full([1, 1], 1.0, tl.float32)
    tmp8  = tl.minimum(tmp6, tmp7)
    tmp9  = tl.full([1, 1], 2.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tl.maximum(tmp10, tmp5)
    tmp12 = tl.minimum(tmp11, tmp7)
    tmp13 = tl.full([1, 1], 0.5, tl.float32)
    tmp14 = tmp12 * tmp13

    # Store to out_ptr0 (NCHW): index = s + 65536*(c + 64*n)
    # s is axis 0 (fast warp dim for stores, stride-1 in out_ptr0) -> coalesced store
    y3_idx  = c_idx2d + 64 * n_pid           # [1, YBLOCK]: combined channel+batch index
    out_idx = s_idx2d + 65536 * y3_idx       # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + out_idx, tmp14, mask=mask2d,
             eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 8192
    xnumel = 65536
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_clamp_convolution_div_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 8192, 65536)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 64, 256, 256), (4194304, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
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
