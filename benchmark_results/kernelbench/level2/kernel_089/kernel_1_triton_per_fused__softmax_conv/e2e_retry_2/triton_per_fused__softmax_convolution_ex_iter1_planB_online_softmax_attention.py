import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_4 = r0_index
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = ((xindex // 1024) % 16)
    x3 = xindex // 16384
    x5 = (xindex % 16384)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (r0_4), None, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    # Step 1: global max over R0 (single pass, no redundant broadcast)
    tmp19 = triton_helpers.max2(tmp14, 1)[:, None].to(tl.float32)  # [XBLOCK, 1]
    # Step 2: numerically stable exp and sum in one shot
    tmp21 = tl.exp(tmp14 - tmp19)                                   # [XBLOCK, R0_BLOCK]
    tmp24 = tl.sum(tmp21, 1)[:, None].to(tl.float32)               # [XBLOCK, 1]
    # Step 3: softmax weights (shapes broadcast naturally, no extra broadcast needed)
    tmp27 = tmp21 / tmp24                                           # [XBLOCK, R0_BLOCK]
    # Step 4: subtract bias (broadcast tmp28 from [1, R0_BLOCK] to [XBLOCK, R0_BLOCK])
    tmp28_bc = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])          # [XBLOCK, R0_BLOCK]
    tmp29 = tmp27 - tmp28_bc                                        # [XBLOCK, R0_BLOCK]
    tmp30 = tl.sigmoid(tmp29)                                       # [XBLOCK, R0_BLOCK]
    tmp31 = tmp30 * tmp29                                           # [XBLOCK, R0_BLOCK]
    # Step 5: output max reduction (no redundant broadcast)
    tmp34 = triton_helpers.max2(tmp31, 1)[:, None].to(tl.float32)  # [XBLOCK, 1]
    tl.store(in_out_ptr0 + (x6), tmp34, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 2097152
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1[grid](arg_0, arg_1, arg_2, 2097152, 16, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 32, 32), (16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 32, 64, 64), (2097152, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
