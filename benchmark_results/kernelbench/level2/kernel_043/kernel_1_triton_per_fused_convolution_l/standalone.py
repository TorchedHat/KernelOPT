import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 16)
    x3 = xindex // 65536
    x5 = (xindex % 65536)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16384 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16385 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (16512 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (16513 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = triton_helpers.max2(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.full([1, 1], float("inf"), tl.float32)
    tmp20 = tmp18 == tmp19
    tmp21 = tl.full([1, 1], 0.0, tl.float32)
    tmp22 = tl.where(tmp20, tmp21, tmp17)
    tmp23 = tmp14 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tl.log(tmp27)
    tmp29 = tmp28 + tmp22
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x6), tmp31, None)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 262144
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1[grid](arg_0, arg_1, 262144, 64, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4, 1, 16, 64, 64), (65536, 65536, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
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
