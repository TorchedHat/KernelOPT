import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr=128):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 112)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 28672*x2), None, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 28672*x2), None, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 28672*x2), None, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 28672*x2), None, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None, eviction_policy='evict_last')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 6422528
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 6422528, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
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
