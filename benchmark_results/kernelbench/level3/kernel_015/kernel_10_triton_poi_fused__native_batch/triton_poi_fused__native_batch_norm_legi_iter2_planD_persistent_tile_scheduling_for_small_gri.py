import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Persistent 1D kernel: each CTA loops over its share of the 46080 total elements
    # YBLOCK is used as BLOCK_SIZE; XBLOCK is ignored (set to 1 in configs)
    total_elems = 46080  # ynumel * xnumel = 5120 * 9
    BLOCK_SIZE: tl.constexpr = YBLOCK
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(total_elems, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, num_ctas):
        elem_start = tile_id * BLOCK_SIZE
        offsets = elem_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elems
        # Decompose flat index -> (y3, x2)
        y3 = offsets // 9   # row in [0, ynumel)
        x2 = offsets % 9    # col in [0, xnumel)
        y0 = y3 % 160
        y1 = y3 // 160
        # Load: matches input stride (1440, 9, 3, 1)
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask, eviction_policy='evict_last')
        # Store: matches output stride (1440, 1, 480, 160)
        tl.store(out_ptr0 + (y0 + 160 * x2 + 1440 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 5120
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10[grid](arg_0, arg_1, 5120, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
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
