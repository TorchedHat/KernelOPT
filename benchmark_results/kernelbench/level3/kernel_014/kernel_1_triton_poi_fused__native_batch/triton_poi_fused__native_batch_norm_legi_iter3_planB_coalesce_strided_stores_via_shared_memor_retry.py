import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 512},  num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)
    ymask = yindex < ynumel
    y0 = yindex % 32
    y1 = yindex // 32
    xoffset = tl.program_id(0) * XBLOCK
    # Outer loop over x (slow dim in output, stride 32):
    for xi in tl.static_range(XBLOCK):
        x2 = xoffset + xi
        xvalid = x2 < xnumel
        # Load: in_ptr0 has stride (9,1) so x is fast — coalesced across y
        val = tl.load(in_ptr0 + (x2 + 9 * yindex),
                      mask=ymask & xvalid, other=0.0,
                      eviction_policy='evict_last')
        # Store: output index y0 + 32*x2 + 288*y1
        # Consecutive yindex values have consecutive y0 (stride 1 for y0<32)
        # => coalesced writes
        tl.store(out_ptr0 + (y0 + 32 * x2 + 288 * y1),
                 val, mask=ymask & xvalid)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1024
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
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
