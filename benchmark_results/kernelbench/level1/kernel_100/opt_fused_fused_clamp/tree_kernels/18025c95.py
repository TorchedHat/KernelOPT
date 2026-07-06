import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256},  num_warps=8, num_stages=2),
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
    BLOCK_ROWS: tl.constexpr = 32
    block_start = tl.program_id(0) * BLOCK_ROWS
    row_ids = block_start + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < M

    acc = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        pred_ptrs = pred_ptr + row_ids[:, None] * pred_stride_m + cols[None, :] * pred_stride_n
        pred = tl.load(pred_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        tgt = tl.load(target_ptr + cols, mask=col_mask, other=0.0)
        val = 1.0 - pred * tgt[None, :]
        val = tl.where(val > 0.0, val, 0.0)
        acc += tl.sum(val, axis=1)

    tl.store(partial_ptr + row_ids, acc, mask=row_mask)


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