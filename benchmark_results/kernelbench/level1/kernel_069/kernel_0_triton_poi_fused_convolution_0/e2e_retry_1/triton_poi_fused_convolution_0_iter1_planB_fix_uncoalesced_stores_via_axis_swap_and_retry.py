import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 8, 'YBLOCK': 256}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 8, 'YBLOCK': 128}, num_warps=2, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # After axis swap:
    # XBLOCK covers N*C = 30 (fast thread axis -> channel c is stride-1 in output)
    # YBLOCK covers HW = 50176 (slow axis)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < 30
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < 50176
    # Decompose xindex (N*C) into channel c and batch n
    x0 = xindex % 3       # channel c, stride-1 in NHWC output
    x1 = xindex // 3      # batch n
    y2 = yindex            # HW pixel index
    # Load from NCHW input: in_ptr0[hw + 50176*(n*3+c)] = in_ptr0[y2 + 50176*xindex]
    tmp0 = tl.load(in_ptr0 + (y2 + 50176 * xindex), xmask & ymask, eviction_policy='evict_last')
    # Store to NHWC output: out_ptr0[c + 3*hw + 150528*n] = out_ptr0[x0 + 3*y2 + 150528*x1]
    tl.store(out_ptr0 + (x0 + 3 * y2 + 150528 * x1), tmp0, xmask & ymask)

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
