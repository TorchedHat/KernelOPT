import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def triplet_margin_loss_row_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    dist_pos_ptr, dist_neg_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    """
      dist_pos[i] = sqrt(sum((anchor[i] - positive[i] + 1e-6)^2))
      dist_neg[i] = sqrt(sum((anchor[i] - negative[i] + 1e-6)^2))
    """
    row = tl.program_id(0)
    base = row * N

    acc_pos = tl.zeros((), dtype=tl.float32)
    acc_neg = tl.zeros((), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        # Annotate contiguity and alignment so compiler emits 128-bit LDG transactions
        cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_N), BLOCK_N)
        mask = cols < N

        a = tl.load(anchor_ptr   + base + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(positive_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        n = tl.load(negative_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)

        diff_pos = a - p + 1e-6
        diff_neg = a - n + 1e-6

        sq_pos = diff_pos * diff_pos
        sq_neg = diff_neg * diff_neg

        # Zero out masked elements
        sq_pos = tl.where(mask, sq_pos, 0.0)
        sq_neg = tl.where(mask, sq_neg, 0.0)

        acc_pos += tl.sum(sq_pos, axis=0)
        acc_neg += tl.sum(sq_neg, axis=0)

    dist_pos = libdevice.sqrt(acc_pos)
    dist_neg = libdevice.sqrt(acc_neg)

    tl.store(dist_pos_ptr + row, dist_pos)
    tl.store(dist_neg_ptr + row, dist_neg)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 32768}, num_warps=16, num_stages=2),
    ],
    key=['M'],
)
@triton.jit

def triplet_margin_loss_reduce_kernel(
    dist_pos_ptr, dist_neg_ptr,
    output_ptr,
    M,
    margin,
    BLOCK_R: tl.constexpr,
):
    """
    Compute mean(clamp(dist_pos + margin - dist_neg, min=0)) over M rows.
    """
    acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, M, BLOCK_R):
        rows = off + tl.arange(0, BLOCK_R)
        mask = rows < M

        dp = tl.load(dist_pos_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        dn = tl.load(dist_neg_ptr + rows, mask=mask, other=0.0).to(tl.float32)

        loss = dp + margin - dn
        loss = tl.where(loss > 0.0, loss, 0.0)
        loss = tl.where(mask, loss, 0.0)

        acc += tl.sum(loss, axis=0)

    result = acc / M
    tl.store(output_ptr, result)


def kernel_function(anchor, positive, negative, margin=1.0):
    M, N = anchor.shape

    dist_pos = torch.empty(M, device=anchor.device, dtype=torch.float32)
    dist_neg = torch.empty(M, device=anchor.device, dtype=torch.float32)

    # Launch row-wise kernel: one program per row
    grid_rows = (M,)
    triplet_margin_loss_row_kernel[grid_rows](
        anchor, positive, negative,
        dist_pos, dist_neg,
        M, N,
    )

    # Launch reduction kernel: single program
    output = torch.empty((), device=anchor.device, dtype=torch.float32)
    grid_reduce = (1,)
    triplet_margin_loss_reduce_kernel[grid_reduce](
        dist_pos, dist_neg,
        output,
        M,
        margin,
    )

    return output


def get_inputs():
    batch_size = 32768
    input_shape = (8192,)
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape, device='cuda') * scale,
        torch.rand(batch_size, *input_shape, device='cuda'),
        torch.rand(batch_size, *input_shape, device='cuda'),
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch reference
        anchor, positive, negative = inputs
        import torch.nn as nn
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        ref = loss_fn(anchor, positive, negative)
        print(f"Triton output: {output.item():.6f}")
        print(f"PyTorch ref:   {ref.item():.6f}")
        print(f"Max abs diff:  {abs(output.item() - ref.item()):.2e}")
        assert abs(output.item() - ref.item()) < 1e-3, f"Mismatch: {output.item()} vs {ref.item()}"
        print("Kernel executed successfully.")