import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 25
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    # Step 1: Coalesced load into SRAM staging buffer (evict_first = streaming, no reuse)
    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), xmask & ymask, eviction_policy='evict_first')
    # Step 2: Stage in shared memory (YBLOCK x XBLOCK tile)
    smem = tl.zeros([YBLOCK, XBLOCK], dtype=tl.float32)
    smem = tl.where(xmask & ymask, tmp0, smem)
    # Step 3: Store from SRAM to global (evict_first = write-once, no reuse)
    tl.store(out_ptr0 + (y0 + 16 * x2 + 400 * y1), smem, xmask & ymask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 768
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((48, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((48, 16, 5, 5), (400, 1, 80, 16), device='cuda:0', dtype=torch.float32)
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
