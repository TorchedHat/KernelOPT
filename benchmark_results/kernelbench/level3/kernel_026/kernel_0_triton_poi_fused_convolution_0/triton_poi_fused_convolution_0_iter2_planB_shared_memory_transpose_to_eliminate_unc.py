import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # numel = ynumel * xnumel = 30 * 50176 = 1,505,280
    # We use a 1D flat-output-index approach with YBLOCK=1 always
    # program_id(0) iterates over flat output indices in chunks of XBLOCK
    numel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    # Decode flat OUTPUT index -> (y1=batch, x2=HW, y0=channel)
    # out_ptr0 layout: NHWC strides (150528, 1, 672, 3) over (N=10, C=3, H=224, W=224)
    # flat_out = y1*150528 + x2*3 + y0  =>  xindex IS flat_out
    y0 = xindex % 3                  # channel in [0,3)
    tmp = xindex // 3
    x2 = tmp % 50176                 # spatial position in [0,50176)
    y1 = tmp // 50176                # batch index in [0,10)
    # Encode flat INPUT index in NCHW layout: strides (150528, 50176, 224, 1)
    # flat_in = y1*150528 + y0*50176 + x2
    flat_in = y1 * 150528 + y0 * 50176 + x2
    tmp0 = tl.load(in_ptr0 + flat_in, xmask)
    # Store to flat output index — stride-1, fully coalesced
    tl.store(out_ptr0 + xindex, tmp0, xmask)

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
