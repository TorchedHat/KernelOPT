import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 4}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 128,  'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 16}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 13312
    xnumel = 9
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_x_tiles = tl.cdiv(9, XBLOCK)
    num_y_tiles = tl.cdiv(13312, YBLOCK)
    total_tiles = num_x_tiles * num_y_tiles
    for tile_id in tl.range(sm_id, total_tiles, num_sms):
        x_tile = tile_id % num_x_tiles
        y_tile = tile_id // num_x_tiles
        xoffset = x_tile * XBLOCK
        yoffset = y_tile * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 416)
        y1 = yindex // 416
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 416*x2 + 3744*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 13312
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34[grid](arg_0, arg_1, 13312, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 416, 3, 3), (3744, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 416, 3, 3), (3744, 1, 1248, 416), device='cuda:0', dtype=torch.float32)
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
