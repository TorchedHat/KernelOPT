import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Input:  (32,128,126,250) strides (4032000,1,32000,128)
    #   in[b,c,h,w] = in_ptr0 + b*4032000 + c*1 + h*32000 + w*128
    # Output: (32,128,126,250) strides (4032000,31500,250,1)
    #   out[b,c,h,w] = out_ptr0 + b*4032000 + c*31500 + h*250 + w
    pid_h = tl.program_id(0)   # 0..125
    pid_w = tl.program_id(1)   # 0..249
    pid_b = tl.program_id(2)   # 0..31
    c_idx = tl.arange(0, 128)  # all 128 channels
    # COALESCED load: consecutive threads differ only in c (stride=1 in input)
    in_offset = pid_b * 4032000 + c_idx + pid_h * 32000 + pid_w * 128
    tmp0 = tl.load(in_ptr0 + in_offset)
    # Store: stride=31500 between threads (unavoidable for this transpose)
    out_offset = pid_b * 4032000 + c_idx * 31500 + pid_h * 250 + pid_w
    tl.store(out_ptr0 + out_offset, tmp0)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
