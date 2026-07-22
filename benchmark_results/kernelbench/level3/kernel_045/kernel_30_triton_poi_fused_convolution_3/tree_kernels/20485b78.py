import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    tx = tl.arange(0, XBLOCK)   # [0..XBLOCK-1]
    ty = tl.arange(0, YBLOCK)   # [0..YBLOCK-1]
    x_idx = pid_x * XBLOCK + tx[None, :]   # shape [1, XBLOCK] -> broadcast to [YBLOCK, XBLOCK]
    y_idx = pid_y * YBLOCK + ty[:, None]   # shape [YBLOCK, 1] -> broadcast to [YBLOCK, XBLOCK]
    x_mask = x_idx < 2048
    y_mask = y_idx < 4096
    # LOAD: contiguous in x (stride-1 in input), strided in y (stride 2048)
    tile = tl.load(in_ptr0 + (x_idx + 2048 * y_idx), mask=x_mask & y_mask, other=0.0)
    # STORE: output strides (1048576, 1, 65536, 512) -> index = y0 + 512*x_idx + 1048576*y1
    y0 = y_idx % 512
    y1 = y_idx // 512
    tl.store(out_ptr0 + (y0 + 512 * x_idx + 1048576 * y1), tile, mask=x_mask & y_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 2048
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_30[grid](arg_0, arg_1, 4096, 2048, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 512, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 512, 16, 128), (1048576, 1, 65536, 512), device='cuda:0', dtype=torch.float32)
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
