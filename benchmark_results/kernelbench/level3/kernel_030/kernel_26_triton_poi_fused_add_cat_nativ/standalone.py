import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_cat_native_layer_norm_slice_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (5472 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 752640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_cat_native_layer_norm_slice_view_26[grid](arg_0, arg_1, 752640, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 28, 28, 96), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
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
