import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_addmm_native_layer_norm_permute_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = tl.max_contiguous(tl.multiple_of((xindex % 128), [1, 1]), [1, 128])
    x3 = xindex
    y0 = yindex
    x2 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + 256*y0), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3 + 256*y0), mask=xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 2*y0), mask=xmask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr3 + (x2 + 2*y0), mask=xmask, eviction_policy='evict_first')
    tmp14 = tl.load(in_ptr4 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1, 1], 128.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp17 = libdevice.fma(tmp13, tmp14, tmp16)  # fuses FMUL+FADD into single FFMA
    # tl.debug_barrier() REMOVED: no shared memory in this kernel (static_smem=0)
    tl.store(in_out_ptr0 + (x3 + 256*y0), tmp17, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    ynumel = 16384
    xnumel = 256
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_addmm_native_layer_norm_permute_view_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 256, YBLOCK=8, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
