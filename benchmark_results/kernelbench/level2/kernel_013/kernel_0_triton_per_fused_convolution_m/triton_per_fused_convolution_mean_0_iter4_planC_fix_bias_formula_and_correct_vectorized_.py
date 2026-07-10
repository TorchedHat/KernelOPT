import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_convolution_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    r0_numel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = (xindex % 16384)
    x4 = xindex // 16384
    x1 = ((xindex // 16384) % 64)
    x5 = xindex
    x0_vec = tl.max_contiguous(tl.multiple_of(x0, [16, 1]), [16, 1])
    # Load bias once outside loop (does not depend on r0)
    tmp1 = tl.load(in_ptr1 + (x1), mask=xmask, eviction_policy='evict_last')
    # Accumulate across r0 in registers -- no shared memory used
    acc = tl.zeros([XBLOCK, 1], dtype=tl.float32)
    tmp0_r0 = tl.load(in_ptr0 + (x0_vec + 16384*0 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r0
    tmp0_r1 = tl.load(in_ptr0 + (x0_vec + 16384*1 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r1
    tmp0_r2 = tl.load(in_ptr0 + (x0_vec + 16384*2 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r2
    tmp0_r3 = tl.load(in_ptr0 + (x0_vec + 16384*3 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r3
    tmp0_r4 = tl.load(in_ptr0 + (x0_vec + 16384*4 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r4
    tmp0_r5 = tl.load(in_ptr0 + (x0_vec + 16384*5 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r5
    tmp0_r6 = tl.load(in_ptr0 + (x0_vec + 16384*6 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r6
    tmp0_r7 = tl.load(in_ptr0 + (x0_vec + 16384*7 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r7
    tmp0_r8 = tl.load(in_ptr0 + (x0_vec + 16384*8 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r8
    tmp0_r9 = tl.load(in_ptr0 + (x0_vec + 16384*9 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r9
    tmp0_r10 = tl.load(in_ptr0 + (x0_vec + 16384*10 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r10
    tmp0_r11 = tl.load(in_ptr0 + (x0_vec + 16384*11 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r11
    tmp0_r12 = tl.load(in_ptr0 + (x0_vec + 16384*12 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r12
    tmp0_r13 = tl.load(in_ptr0 + (x0_vec + 16384*13 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r13
    tmp0_r14 = tl.load(in_ptr0 + (x0_vec + 16384*14 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r14
    tmp0_r15 = tl.load(in_ptr0 + (x0_vec + 16384*15 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r15
    tmp0_r16 = tl.load(in_ptr0 + (x0_vec + 16384*16 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r16
    tmp0_r17 = tl.load(in_ptr0 + (x0_vec + 16384*17 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r17
    tmp0_r18 = tl.load(in_ptr0 + (x0_vec + 16384*18 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r18
    tmp0_r19 = tl.load(in_ptr0 + (x0_vec + 16384*19 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r19
    tmp0_r20 = tl.load(in_ptr0 + (x0_vec + 16384*20 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r20
    tmp0_r21 = tl.load(in_ptr0 + (x0_vec + 16384*21 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r21
    tmp0_r22 = tl.load(in_ptr0 + (x0_vec + 16384*22 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r22
    tmp0_r23 = tl.load(in_ptr0 + (x0_vec + 16384*23 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r23
    tmp0_r24 = tl.load(in_ptr0 + (x0_vec + 16384*24 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r24
    tmp0_r25 = tl.load(in_ptr0 + (x0_vec + 16384*25 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r25
    tmp0_r26 = tl.load(in_ptr0 + (x0_vec + 16384*26 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r26
    tmp0_r27 = tl.load(in_ptr0 + (x0_vec + 16384*27 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r27
    tmp0_r28 = tl.load(in_ptr0 + (x0_vec + 16384*28 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r28
    tmp0_r29 = tl.load(in_ptr0 + (x0_vec + 16384*29 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r29
    tmp0_r30 = tl.load(in_ptr0 + (x0_vec + 16384*30 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r30
    tmp0_r31 = tl.load(in_ptr0 + (x0_vec + 16384*31 + 524288*x4), mask=xmask, eviction_policy='evict_first')
    acc = acc + tmp0_r31
    # Correct mean formula: sum/32 + bias
    acc = acc * (1.0 / 32.0) + tmp1
    tl.store(out_ptr0 + (x5), acc, mask=xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 16777216
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 64, 1, 128, 128), (1048576, 16384, 16777216, 128, 1), device='cuda:0', dtype=torch.float32)
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
