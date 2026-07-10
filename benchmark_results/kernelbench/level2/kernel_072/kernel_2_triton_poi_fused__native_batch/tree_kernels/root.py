import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool3d_convolution_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3456000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 15)
    x1 = ((xindex // 15) % 15)
    x2 = ((xindex // 225) % 15)
    x3 = xindex // 3375
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (31 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (32 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (961 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (962 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (992 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (993 + 2*x0 + 62*x1 + 1922*x2 + 29792*x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tl.store(out_ptr0 + (x4), tmp16, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3456000
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool3d_convolution_2[grid](arg_0, arg_1, 3456000, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 16, 31, 31, 31), (476672, 29792, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 16, 15, 15, 15), (54000, 3375, 225, 15, 1), device='cuda:0', dtype=torch.float32)
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
