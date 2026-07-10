import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # XBLOCK tiles Y (ynumel=30); YBLOCK tiles X (xnumel=50176)
        # XBLOCK must be power of 2; 32 covers ynumel=30 with one tile
        triton.Config({'XBLOCK': 32,  'YBLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128},  num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    # pid_x now covers Y (channel/batch dimension, stride-1 in output => coalesced stores)
    yoffset = tl.program_id(0) * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    ymask = yindex < ynumel
    # pid_y now covers X (spatial dimension)
    xoffset = tl.program_id(1) * YBLOCK
    xindex = xoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    xmask = xindex < xnumel
    # Index expressions UNCHANGED - same math as before
    x2 = xindex   # shape [YBLOCK, 1]
    y3 = yindex   # shape [1, XBLOCK]
    y0 = (y3 % 3)      # channel: stride-1 in output
    y1 = y3 // 3       # batch: stride-150528 in output
    # Load: x2 + 50176*y3 broadcasts to [YBLOCK, XBLOCK]
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    # Store: y0 + 3*x2 + 150528*y1 broadcasts to [YBLOCK, XBLOCK]
    # Consecutive threads (axis-1, XBLOCK) vary along y3 => y0=y3%3 varies with stride-1 => COALESCED
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
