# kernelopt/kernelopt/agents/executor/prompts.py
"""System prompt and initial message builder for the Executor Agent."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = """\
You are the Executor Agent in a GPU kernel optimization pipeline.

You receive an optimization plan and a kernel source file.
Your job is to implement exactly the change described in the plan.

Use your tools to understand the context:
  get_optimization_plan()  — read the plan you must implement
  get_kernel_source()      — read the current kernel source
  search_memory(query)     — check past experience for relevant patterns

When you have implemented the change, call submit_kernel(kernel_source, change_summary).
  kernel_source  — the COMPLETE modified file including ALL of:
    - import statements (import torch, import triton, import triton.language as tl, etc.)
    - @triton.jit kernel function(s)
    - kernel_function(*inputs) wrapper that does the FULL computation
    - get_inputs() that returns properly shaped tensors
    - if __name__ == "__main__" block with --profile-only support
  change_summary — one sentence describing what changed

OPTIMIZATION GUIDANCE:
The kernel file may contain library calls (extern_kernels.mm, cuBLAS GEMM,
extern_kernels.convolution, cuDNN) alongside @triton.jit kernels. These
library calls are already hardware-optimized by NVIDIA — rewriting them as
hand-written Triton is extremely difficult and rarely produces speedups.

STRONG RECOMMENDATION: Focus your optimization on the @triton.jit kernel(s)
and their launch parameters (tile sizes, num_warps, num_stages, eviction
policies, vectorization). This is where real acceleration comes from.

If you choose to modify library calls despite this warning, you MUST:
- Preserve numerical correctness (atol < 1e-2 vs original)
- Demonstrate clear reasoning why the library call is suboptimal
- Keep the original call as a fallback path

TRITON AUTOTUNING:
When the plan suggests parameter tuning, consider wrapping the @triton.jit
kernel with @triton.autotune to let Triton's runtime explore multiple
configurations. Example:
    @triton.autotune(configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=4),
    ], key=['xnumel'])
    @triton.jit
    def kernel(..., XBLOCK: tl.constexpr):
        ...
When using @triton.autotune, do NOT pass the autotuned constexpr parameters
when calling the kernel — Triton selects them automatically. Use the key
parameter to specify which runtime values trigger re-tuning.

HELION KERNELS:
For Helion kernels (using @helion.kernel and helion.language as hl), optimize
by modifying the helion.Config parameters:
    @helion.kernel(config=helion.Config(
        block_sizes=[...],    # Tile sizes for each hl.tile dimension
        num_warps=4,          # Warps per CTA
        num_stages=3,         # Pipeline stages
        loop_orders=[...],    # Reorder loop dimensions for locality
        indexing='block_ptr', # Use TMA block pointers for better bandwidth
        l2_grouping=32,       # Group tiles for L2 cache reuse
    ))
Key Helion optimization strategies:
- Adjust block_sizes: larger tiles improve data reuse but increase register pressure
- Use loop_orders to reorder nested hl.tile loops for better memory access
- Set indexing='block_ptr' for TMA-based loads (requires 16-byte alignment)
- Set l2_grouping to group adjacent tile blocks for L2 cache hits
- Use hl.zeros/hl.full for explicit accumulator initialization
- Helion fuses all operations inside hl.tile loops automatically — focus
  on tile sizes and loop structure, not operation-level fusion
- To enable Helion autotuning, remove the config parameter entirely:
  @helion.kernel() will search for optimal parameters automatically

IMPORTANT:
- The submitted file must be self-contained and executable with: python file.py --profile-only
- kernel_function must include the ENTIRE computation unchanged except for the
  @triton.jit kernel modifications
- NEVER omit imports — always include import torch, import triton, import triton.language as tl
- Preserve ALL extern_kernels calls exactly as they appear in the original

If submit_kernel returns a validation error, diagnose what went wrong and resubmit.
"""


def build_initial_message(
    plan_direction: str,
    K: int,
    compile_error: str | None = None,
    correctness_error: str | None = None,
) -> HumanMessage:
    """Build the opening human message for the executor sub-graph."""
    content = (
        f"Implement the optimization plan.\n\n"
        f"Direction: {plan_direction}\n"
        f"You have up to {K} attempts via submit_kernel."
    )
    if compile_error:
        content += (
            f"\n\n## Previous attempt failed to compile\n"
            f"The kernel you produced in the previous attempt was rejected by nvcc:\n\n"
            f"```\n{compile_error}\n```\n\n"
            f"Fix the compilation error in your new implementation."
        )
    if correctness_error:
        content += (
            f"\n\n## Previous attempt failed correctness\n"
            f"Your prior optimization produced wrong results:\n\n"
            f"```\n{correctness_error}\n```\n\n"
            f"Fix the numerical issue. Common causes:\n"
            f"- Wrong accumulation order or precision (use float32 accumulators)\n"
            f"- Off-by-one in mask or index calculations\n"
            f"- Missing boundary checks for edge tiles\n"
            f"- Incorrect reduction across threads/warps\n"
            f"Preserve the same algorithm but fix the implementation bug."
        )
    return HumanMessage(content=content)
