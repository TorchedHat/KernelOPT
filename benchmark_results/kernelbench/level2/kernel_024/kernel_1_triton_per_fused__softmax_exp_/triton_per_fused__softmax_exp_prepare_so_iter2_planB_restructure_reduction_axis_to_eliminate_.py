import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 3072   # batch=128 * heads=24
    r0_numel = 900  # spatial = 30*30, stride=1 in memory
    R0_BLOCK: tl.constexpr = 1024

    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask  = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]  # 0..1023
    r0_mask  = r0_index < r0_numel              # mask out 900..1023

    # Decode (batch, head) from xindex
    # Tensor strides: (21600, 900, 30, 1)
    x_batch = xindex // 24   # [XBLOCK, 1]
    x_head  = xindex % 24    # [XBLOCK, 1]

    # Contiguous spatial access: stride=1 in r0_index
    ptrs = in_out_ptr0 + (x_batch * 21600 + x_head * 900 + r0_index)

    tmp0 = tl.load(ptrs, mask=(r0_mask & xmask), other=float('-inf'))

    # Softmax over 900 spatial positions
    tmp5 = tl.where(r0_mask & xmask, tmp0, float('-inf'))
    tmp6 = tl.max(tmp5, axis=1)[:, None]
    tmp7 = tmp0 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp11 = tl.where(r0_mask & xmask, tmp8, 0.0)
    tmp12 = tl.sum(tmp11, axis=1)[:, None]
    tmp13 = tmp0 - tmp6
    tmp14 = tl.exp(tmp13)
    tmp15 = tmp14 / tmp12

    tl.store(ptrs, tmp15, mask=(r0_mask & xmask))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 115200
    r0_numel = 24
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24, XBLOCK=128)
    return arg_0

def get_inputs():
    arg_0 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0]

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
