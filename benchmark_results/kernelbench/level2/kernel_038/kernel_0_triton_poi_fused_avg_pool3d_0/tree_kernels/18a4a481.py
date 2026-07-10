import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    # tl.multiple_of tells compiler 2*x0 is always 2-float (8-byte) aligned
    # enabling 64-bit vectorized loads for each +0/+1 contiguous pair
    base = tl.multiple_of(2 * x0, 2) + 128 * x1 + 8192 * x2
    # Pair 1: D=0, H=0 input-W neighbors (contiguous floats at base+0, base+1)
    tmp01 = tl.load(in_ptr0 + base,        None, eviction_policy='evict_last') \
          + tl.load(in_ptr0 + base + 1,    None, eviction_policy='evict_last')
    # Pair 2: D=0, H=1 input-W neighbors (contiguous floats at base+64, base+65)
    tmp23 = tl.load(in_ptr0 + base + 64,   None, eviction_policy='evict_last') \
          + tl.load(in_ptr0 + base + 65,   None, eviction_policy='evict_last')
    # Pair 3: D=1, H=0 input-W neighbors (contiguous floats at base+4096, base+4097)
    tmp45 = tl.load(in_ptr0 + base + 4096, None, eviction_policy='evict_last') \
          + tl.load(in_ptr0 + base + 4097, None, eviction_policy='evict_last')
    # Pair 4: D=1, H=1 input-W neighbors (contiguous floats at base+4160, base+4161)
    tmp67 = tl.load(in_ptr0 + base + 4160, None, eviction_policy='evict_last') \
          + tl.load(in_ptr0 + base + 4161, None, eviction_policy='evict_last')
    tmp_sum = tmp01 + tmp23 + tmp45 + tmp67
    tl.store(out_ptr0 + x3, tmp_sum * 0.125, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16777216
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
