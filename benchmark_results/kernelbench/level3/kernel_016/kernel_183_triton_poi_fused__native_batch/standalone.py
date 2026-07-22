import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_183(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 1152*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 1184*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 1216*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 1248*x1), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + 1280*x1), tmp0, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_183[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 15680, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (58016, 1, 8288, 1184), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (59584, 1, 8512, 1216), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 7, 7), (61152, 1, 8736, 1248), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
