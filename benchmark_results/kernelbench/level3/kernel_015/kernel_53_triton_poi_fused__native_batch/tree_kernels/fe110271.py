import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0  = tl.load(in_ptr0 + (x2), xmask)
    tmp1  = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')  # mean
    tmp3  = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')  # var
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')  # weight
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')  # bias
    # Precompute per-channel scale: weight / sqrt(var + eps)
    scale    = tmp12 / tl.sqrt(tmp3 + 1e-5)
    # FMA #1: bias_adj = bias - mean*scale  (per channel, broadcast)
    bias_adj = tl.math.fma(-tmp1, scale, tmp14)
    # FMA #2: per-element normalized+scaled+shifted value
    tmp_norm = tl.math.fma(tmp0, scale, bias_adj)
    # ReLU with float zero to avoid int32->float32 conversion overhead
    tmp17    = tl.maximum(tl.zeros([XBLOCK], tl.float32), tmp_norm)
    tl.store(out_ptr0 + (x2), tmp17, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 1003520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_53[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1003520)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
