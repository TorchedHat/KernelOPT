import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci,
    stride_on, stride_oc, stride_oh, stride_ow,
    HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # M = N_batch * H * W (output spatial positions)
    # N = out_channels (Co)
    # K = in_channels (Ci)
    # x layout: (N_batch, H, W, Ci) — NHWC, coalesced on K axis
    # w layout: (Ci, Co) — transposed, coalesced on N axis
    # out layout: (N_batch, Co, H, W) with strides stride_on, stride_oc, stride_oh, stride_ow

    pid = tl.program_id(0).to(tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly reordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # Decode offs_m -> (batch_idx, hw_idx)
    # offs_m = batch_idx * HW + hw_idx
    batch_idx = offs_m // HW   # [BLOCK_M]
    hw_idx = offs_m % HW       # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k  # [BLOCK_K]

        # Load x tile: shape [BLOCK_M, BLOCK_K]
        # x in NHWC layout: x[batch, hw, ci] = x_ptr + (batch*HW + hw)*K + ci
        # k_offs is unit-stride (ci dimension) -> fully coalesced
        x_row = (batch_idx[:, None].to(tl.int64) * HW + hw_idx[:, None].to(tl.int64)) * K  # [BLOCK_M, 1]
        x_ptrs = x_ptr + x_row + k_offs[None, :].to(tl.int64)  # [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile: shape [BLOCK_K, BLOCK_N]
        # w in (Ci, Co) layout: w[ci, co] = w_ptr + ci*N + co
        # offs_n is unit-stride (co dimension) -> fully coalesced
        w_ptrs = w_ptr + k_offs[:, None].to(tl.int64) * N + offs_n[None, :].to(tl.int64)  # [BLOCK_K, BLOCK_N]
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Store output: out[batch_idx[m], offs_n[n], hw_idx[m]]
    # out_ptr + batch_idx[m]*stride_on + offs_n[n]*stride_oc + hw_idx[m]
    out_batch = batch_idx[:, None]   # [BLOCK_M, 1]
    out_co = offs_n[None, :]         # [1, BLOCK_N]
    out_hw = hw_idx[:, None]         # [BLOCK_M, 1]
    out_ptrs = out_ptr + out_batch.to(tl.int64) * stride_on + out_co.to(tl.int64) * stride_oc + out_hw.to(tl.int64)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)

def kernel_function(w, x):
    # w: (Co, Ci, 1, 1), x: (N, Ci, H, W)
    N_batch, Ci, H, W = x.shape
    Co = w.shape[0]
    HW = H * W
    M = N_batch * HW
    K = Ci
    Nout = Co

    out = torch.empty((N_batch, Co, H, W), device=x.device, dtype=x.dtype)

    # Strides for x (NCHW)
    stride_xn = x.stride(0)
    stride_xc = x.stride(1)
    stride_xh = x.stride(2)
    stride_xw = x.stride(3)

    # Strides for w (Co, Ci, 1, 1) -> treat as (Co, Ci)
    w_2d = w.view(Co, Ci)
    stride_wco = w_2d.stride(0)
    stride_wci = w_2d.stride(1)

    # Strides for out (NCHW)
    stride_on = out.stride(0)
    stride_oc = out.stride(1)
    stride_oh = out.stride(2)
    stride_ow = out.stride(3)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Nout, meta['BLOCK_N']),)

    pointwise_conv2d_kernel[grid](
        x, w_2d, out,
        M, Nout, K,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wco, stride_wci,
        stride_on, stride_oc, stride_oh, stride_ow,
        HW,
    )

    return out


def get_inputs():
    w = torch.rand(128, 64, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(16, 64, 1024, 1024, device='cuda', dtype=torch.float32)
    return [w, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    w, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn.functional as F
        ref = F.conv2d(x, w, bias=None, stride=1, padding=0)
        out = kernel_function(w, x)
        torch.cuda.synchronize()
        max_diff = (out - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(w, x)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")