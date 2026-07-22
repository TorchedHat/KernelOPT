import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_215(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    x1 = tl.program_id(0)          # outer row index: 0..489 (10*7*7=490 rows)
    x0 = tl.arange(0, XBLOCK)[:]   # inner col index: 0..31  (32 columns)
    x2 = x1 * XBLOCK + x0          # flat source index, same as original xindex
    tmp0 = tl.load(in_ptr0 + x2)   # no mask: 490*32=15680 exactly
    tl.store(out_ptr0 + (x0 + 1664 * x1), tmp0)
    tl.store(out_ptr1 + (x0 + 1696 * x1), tmp0)
    tl.store(out_ptr2 + (x0 + 1728 * x1), tmp0)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_215[grid](arg_0, arg_1, arg_2, arg_3, 15680, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (81536, 1, 11648, 1664), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (83104, 1, 11872, 1696), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (84672, 1, 12096, 1728), device='cuda:0', dtype=torch.float32)
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
