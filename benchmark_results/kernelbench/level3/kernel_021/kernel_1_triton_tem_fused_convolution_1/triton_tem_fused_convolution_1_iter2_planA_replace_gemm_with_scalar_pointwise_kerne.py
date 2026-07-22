import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 256
    M = 8388608

    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M

    # Load 18 scalar B weights: B[k, n] = arg_B + k*1 + n*3
    # k=0, n=0..5
    b00 = tl.load(arg_B + 0)   # B[0,0]
    b01 = tl.load(arg_B + 3)   # B[0,1]
    b02 = tl.load(arg_B + 6)   # B[0,2]
    b03 = tl.load(arg_B + 9)   # B[0,3]
    b04 = tl.load(arg_B + 12)  # B[0,4]
    b05 = tl.load(arg_B + 15)  # B[0,5]
    # k=1, n=0..5
    b10 = tl.load(arg_B + 1)   # B[1,0]
    b11 = tl.load(arg_B + 4)   # B[1,1]
    b12 = tl.load(arg_B + 7)   # B[1,2]
    b13 = tl.load(arg_B + 10)  # B[1,3]
    b14 = tl.load(arg_B + 13)  # B[1,4]
    b15 = tl.load(arg_B + 16)  # B[1,5]
    # k=2, n=0..5
    b20 = tl.load(arg_B + 2)   # B[2,0]
    b21 = tl.load(arg_B + 5)   # B[2,1]
    b22 = tl.load(arg_B + 8)   # B[2,2]
    b23 = tl.load(arg_B + 11)  # B[2,3]
    b24 = tl.load(arg_B + 14)  # B[2,4]
    b25 = tl.load(arg_B + 17)  # B[2,5]

    # Load 6 scalar bias values: bias[n] = in_ptr0 + n
    bias0 = tl.load(in_ptr0 + 0)
    bias1 = tl.load(in_ptr0 + 1)
    bias2 = tl.load(in_ptr0 + 2)
    bias3 = tl.load(in_ptr0 + 3)
    bias4 = tl.load(in_ptr0 + 4)
    bias5 = tl.load(in_ptr0 + 5)

    # Load 3 A-column vectors: A[m, k] = arg_A + m*3 + k
    a0 = tl.load(arg_A + rows * 3 + 0, mask=mask_m, other=0.0)
    a1 = tl.load(arg_A + rows * 3 + 1, mask=mask_m, other=0.0)
    a2 = tl.load(arg_A + rows * 3 + 2, mask=mask_m, other=0.0)

    # Compute 6 output vectors: acc_n = a0*b0n + a1*b1n + a2*b2n + bias_n
    acc0 = a0 * b00 + a1 * b10 + a2 * b20 + bias0
    acc1 = a0 * b01 + a1 * b11 + a2 * b21 + bias1
    acc2 = a0 * b02 + a1 * b12 + a2 * b22 + bias2
    acc3 = a0 * b03 + a1 * b13 + a2 * b23 + bias3
    acc4 = a0 * b04 + a1 * b14 + a2 * b24 + bias4
    acc5 = a0 * b05 + a1 * b15 + a2 * b25 + bias5

    # Store 6 output vectors: out_ptr0 + m*6 + n
    tl.store(out_ptr0 + rows * 6 + 0, acc0, mask=mask_m)
    tl.store(out_ptr0 + rows * 6 + 1, acc1, mask=mask_m)
    tl.store(out_ptr0 + rows * 6 + 2, acc2, mask=mask_m)
    tl.store(out_ptr0 + rows * 6 + 3, acc3, mask=mask_m)
    tl.store(out_ptr0 + rows * 6 + 4, acc4, mask=mask_m)
    tl.store(out_ptr0 + rows * 6 + 5, acc5, mask=mask_m)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (65536, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((6, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
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
