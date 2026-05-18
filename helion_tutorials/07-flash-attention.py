"""Helion Tutorial Problem 7: Simple FlashAttention with online softmax.

Source: https://helionlang.com/helion_tutorials.html#problem-7-simple-flashattention
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def flashatt_spec(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def flashatt_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    seq_len = q.size(0)
    out = torch.empty_like(q)

    for tile_q in hl.tile(seq_len):
        q_tile = q[tile_q]

        max_val = hl.full([tile_q], float('-inf'), dtype=torch.float32)
        sum_exp = hl.zeros([tile_q], dtype=torch.float32)
        weighted_sum = hl.zeros([tile_q], dtype=torch.float32)

        for tile_kv in hl.tile(seq_len):
            k_tile = k[tile_kv]
            v_tile = v[tile_kv]

            scores = q_tile[:, None] * k_tile[None, :]

            batch_max = torch.amax(scores, dim=1)
            new_max = torch.maximum(max_val, batch_max)

            scale_factor = torch.exp(max_val - new_max)
            sum_exp = sum_exp * scale_factor
            weighted_sum = weighted_sum * scale_factor

            exp_scores = torch.exp(scores - new_max[:, None])
            sum_exp = sum_exp + torch.sum(exp_scores, dim=1)
            weighted_sum = weighted_sum + torch.sum(exp_scores * v_tile[None, :], dim=1)

            max_val = new_max

        out[tile_q] = weighted_sum / sum_exp

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return flashatt_spec(*inputs)

def kernel_function(*inputs):
    return flashatt_kernel(*inputs)

def get_inputs():
    return [
        torch.randn(200, device="cuda"),
        torch.randn(200, device="cuda"),
        torch.randn(200, device="cuda"),
    ]


if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    if "--profile-only" in sys.argv:
        kernel_function(*inputs)
        torch.cuda.synchronize()
    else:
        kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
