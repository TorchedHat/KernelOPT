import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def hinge_loss_row_kernel(
    pred_ptr, target_ptr, partial_ptr,
    M, N,
    pred_stride_m, pred_stride_n,
    BLOCK_N: tl.constexpr,
):
    # Each CTA processes 2 consecutive rows to amortize target_ptr loads
    block_id = tl.program_id(0)
    row0 = block_id * 2
    row1 = block_id * 2 + 1

    valid0 = row0 < M
    valid1 = row1 < M

    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)

    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        # Load target once, reuse for both rows
        tgt = tl.load(target_ptr + cols, mask=col_mask, other=0.0)

        if valid0:
            pred0 = tl.load(pred_ptr + row0 * pred_stride_m + cols * pred_stride_n, mask=col_mask, other=0.0)
            v0 = 1.0 - pred0 * tgt
            acc0 += tl.sum(tl.where(v0 > 0.0, v0, 0.0), axis=0)

        if valid1:
            pred1 = tl.load(pred_ptr + row1 * pred_stride_m + cols * pred_stride_n, mask=col_mask, other=0.0)
            v1 = 1.0 - pred1 * tgt
            acc1 += tl.sum(tl.where(v1 > 0.0, v1, 0.0), axis=0)

    if valid0:
        tl.store(partial_ptr + row0, acc0)
    if valid1:
        tl.store(partial_ptr + row1, acc1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 256}, num_warps=4, num_stages=2),
    ],
    key=['M'],
)
@triton.jit

def hinge_loss_reduce_kernel(
    partial_ptr, out_ptr,
    M, total_elements,
    BLOCK_R: tl.constexpr,
):
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, M, BLOCK_R):
        idxs = off + tl.arange(0, BLOCK_R)
        mask = idxs < M
        vals = tl.load(partial_ptr + idxs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    result = acc / total_elements
    tl.store(out_ptr, result)


def kernel_function(predictions, targets):
    M, N = predictions.shape
    assert targets.shape[0] == N, f"targets shape {targets.shape} must match predictions columns {N}"

    partial = torch.empty(M, device=predictions.device, dtype=torch.float32)

    grid_row = (M,)
    hinge_loss_row_kernel[grid_row](
        predictions, targets, partial,
        M, N,
        predictions.stride(0), predictions.stride(1),
    )

    out = torch.empty((), device=predictions.device, dtype=torch.float32)
    total_elements = float(M * N)

    hinge_loss_reduce_kernel[(1,)](
        partial, out,
        M, total_elements,
    )

    return out


def get_inputs():
    torch.manual_seed(0)
    predictions = torch.rand(32768, 32768, device='cuda', dtype=torch.float32)
    targets = (torch.randint(0, 2, (32768,), device='cuda').float() * 2 - 1)
    return [predictions, targets]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()

    # Verify correctness
    if not args.profile_only:
        predictions, targets = inputs
        output = kernel_function(predictions, targets)
        torch.cuda.synchronize()

        # Reference computation
        ref = torch.mean(torch.clamp(1 - predictions * targets, min=0))
        print(f"Triton output: {output.item()}")
        print(f"Reference output: {ref.item()}")
        diff = abs(output.item() - ref.item())
        print(f"Max diff: {diff}")
        if diff < 1e-3:
            print("PASSED")
        else:
            print("FAILED")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")