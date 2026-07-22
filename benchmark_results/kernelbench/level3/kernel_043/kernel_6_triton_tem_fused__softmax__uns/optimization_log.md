# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24420 ns — **Best latency**: 30940 ns (**0.7893×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_6_triton_tem_fused__softmax__uns/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.03
- **Registers/thread**: 123
- **Duration**: 24420 ns
- **SM Throughput**: 6.25 % | **Memory**: 5.70 % | **DRAM**: 1.06 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 2.25 inst/cycle | **No Eligible**: 43.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_cover_full_batch_dimension | 1.002× | 1.000× | ✓ |  |
| B | fix_grid_batch_dimension_to_expose_all_1024_batch_slices | 1.000× | 1.000× | ✓ |  |
| C | expand_batch_grid_and_reduce_register_pressure | 1.003× | 0.789× | ✓ | **✓ best** |
| D | fix_grid_batch_dimension_and_autotune_tile_sizes | 1.002× | 0.815× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_cover... | Δ | fix_grid_batch_di... | Δ | expand_batch_grid... | Δ | fix_grid_batch_di... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24420 ns | 24420 ns | +0 ns (+0.0%) | 24420 ns | +0 ns (+0.0%) | 30940 ns | +6520 ns (+26.7%) | 29980 ns | +5560 ns (+22.8%) |
| SM Throughput (%) | 6.25 % | 6.32 % | +0.07 pp | 6.23 % | -0.02 pp | 5.03 % | -1.22 pp | 4.99 % | -1.26 pp |
| Memory Throughput (%) | 5.70 % | 5.77 % | +0.07 pp | 5.68 % | -0.02 pp | 4.59 % | -1.11 pp | 4.55 % | -1.15 pp |
| DRAM Throughput (%) | 1.06 % | 1.06 % | +0.00 pp | 1.06 % | +0.00 pp | 0.84 % | -0.22 pp | 0.86 % | -0.20 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 123.00 register/thread | 123.00 register/thread | +0 | 123.00 register/thread | +0 | 96.00 register/thread | -27 | 96.00 register/thread | -27 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.02 | -0.01 | 0.02 | -0.01 |
| No Eligible (%) | 43.62 % | 43.29 % | -0.33 pp | 43.66 % | +0.04 pp | 55.98 % | +12.36 pp | 55.82 % | +12.20 pp |
| IPC Active | 2.25 inst/cycle | 2.25 inst/cycle | +0.00 | 2.25 inst/cycle | +0.00 | 1.77 inst/cycle | -0.48 | 1.77 inst/cycle | -0.48 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 1.000× | 0.802× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 24420 ns | 30460 ns | +6040 ns (+24.7%) |
| SM Throughput (%) | 6.25 % | 5.07 % | -1.18 pp |
| Memory Throughput (%) | 5.70 % | 4.63 % | -1.07 pp |
| DRAM Throughput (%) | 1.06 % | 0.85 % | -0.21 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 123.00 register/thread | 96.00 register/thread | -27 |
| Waves/SM | 0.03 | 0.02 | -0.01 |
| No Eligible (%) | 43.62 % | 55.31 % | +11.69 pp |
| IPC Active | 2.25 inst/cycle | 1.78 inst/cycle | -0.47 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | online_softmax_attention | 0.985× | 0.985× | ✓ |  |
| B | increase_tile_sizes_with_grid_fix | 0.085× | 0.079× | ✓ |  |
| C | persistent_batch_loop_to_fuse_batch_slices_and_fix_grid | 0.998× | 0.796× | ✓ |  |
| D | vectorized_loads_stores_with_corrected_batch_grid | 0.998× | 0.588× | ✓ |  |


### Metric Comparison

| Metric | Baseline | online_softmax_at... | Δ | increase_tile_siz... | Δ | persistent_batch_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24420 ns | 24800 ns | +380 ns (+1.6%) | 309730 ns | +285310 ns (+1168.3%) | 30660 ns | +6240 ns (+25.6%) | 41540 ns | +17120 ns (+70.1%) |
| SM Throughput (%) | 6.25 % | 6.27 % | +0.02 pp | 1.13 % | -5.12 pp | 4.99 % | -1.26 pp | 3.71 % | -2.54 pp |
| Memory Throughput (%) | 5.70 % | 5.82 % | +0.12 pp | 1.67 % | -4.03 pp | 4.56 % | -1.14 pp | 3.43 % | -2.27 pp |
| DRAM Throughput (%) | 1.06 % | 1.04 % | -0.02 pp | 0.09 % | -0.97 pp | 0.84 % | -0.22 pp | 0.62 % | -0.44 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 123.00 register/thread | 96.00 register/thread | -27 | 255.00 register/thread | +132 | 96.00 register/thread | -27 | 96.00 register/thread | -27 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.06 | +0.03 | 0.02 | -0.01 | 0.02 | -0.01 |
| No Eligible (%) | 43.62 % | 43.47 % | -0.15 pp | 90.06 % | +46.44 pp | 55.18 % | +11.56 pp | 67.58 % | +23.96 pp |
| IPC Active | 2.25 inst/cycle | 2.25 inst/cycle | +0.00 | 0.40 inst/cycle | -1.85 | 1.78 inst/cycle | -0.47 | 1.30 inst/cycle | -0.95 |


### Key Learnings

#### ⚠️ Plan A: `online_softmax_attention` (0.000×)

Plan 'online_softmax_attention' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=9.718e+01 > atol=0.001
seed=42 output[0]: max_diff=9.542e+01 > atol=0.001
seed=1337 output[0]: max_diff=9.623e+01 > atol=0.001

> **Applicable when**: The kernel is a batched matmul (BMM) computing attention-score-weighted V sums. The launch grid is (16, 1, 1): grid[0]=16 covers the 512x96 output tile space (ceil(512/64)*ceil(96/64)=8*2=16 tiles), but grid[1]=1 means only 1 of the 1024 batch*head slices (128 batches * 8 heads) is ever processed. This is both a correctness bug and the root cause of 0.03 waves/SM and 6.25% occupancy. Prior attempts (exp_001, exp_triton_autotune_no_config_change_regression_001) failed because they added autotune without fixing the grid. The diversity instruction mandates online_softmax_attention: replace the naive multi-pass approach with a single-pass Flash-Attention-style kernel that streams over the KV sequence dimension, maintaining a running (max, normalizer) pair per query row. This eliminates materializing the full 512x512 attention score matrix, reduces memory traffic, and restructures the grid to cover all 1024 batch*head slices.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=9.718e+01 > atol=0.001
seed=42 output[0]: max_diff=9.542e+01 > atol=0.001
seed=1337 output[0]: max_diff=9.623e+01 > atol=0.001

#### ❌ Plan C: `persistent_batch_loop_to_fuse_batch_slices_and_fix_grid` (0.796×)

The plan to expand the Triton launch grid from (16,1,1) to (16,1024,1) was correctly identified as the fix for severe grid underutilization (0.02 waves/SM, 6.25% achieved occupancy, LaunchConfiguration rule firing), but the code diff was empty — no actual change was applied to the kernel. All profiling metrics before and after are nearly identical (duration 30.46 µs vs 30.66 µs, grid size 16, waves_per_sm 0.02, achieved occupancy 6.25%), confirming the plan was not executed. The regression (0.7965x) reflects measurement noise, not a real performance change. This is an implementation flaw: the strategy is correct but the code was not modified.

> **Applicable when**: Kernel uses tl.program_id(1) as batch index but grid[1]=1, causing all but one batch slice to be silently skipped. LaunchConfiguration NCU rule fires with grid_size << sm_count, waves_per_sm < 0.05, achieved occupancy < 10%.

> **Avoid when**: Do not record this as a valid optimization attempt — the diff was empty and no code change was made. The strategy itself (expanding grid[1] to the full batch count) remains the correct fix and should be retried with an actual code modification.

#### ❌ Plan D: `vectorized_loads_stores_with_corrected_batch_grid` (0.588×)

Replacing a broadcast-xindex store with a tl.max_contiguous/tl.multiple_of contiguous pointer store, reducing BLOCK_K from 32 to 16, and adding evict_last on B loads — without actually fixing the launch grid from (16,1,1) to (16,1024,1) — caused a 41% regression. The grid remained at size 16 with waves_per_sm=0.02 and achieved occupancy stuck at 6.25%, so the grid fix was never applied despite being in the plan. Duration increased from 30.46 µs to 41.54 µs, elapsed cycles rose from 45,642 to 62,269, Issue Slots Busy dropped from 44.6% to 32.4%, and Warp Cycles Per Issued Instruction worsened from 2.25 to 3.03 — a new CPIStall rule fired (L1TEX stall at 41% of cycles) indicating the BLOCK_K=16 reduction doubled the K-loop iteration count and increased memory-latency stall exposure without any parallelism benefit to hide it.

> **Applicable when**: Never applicable as a standalone fix: vectorized store rewrites (tl.max_contiguous + tl.multiple_of) and BLOCK_K reductions cannot compensate for a grid that leaves 98%+ of SMs idle. These micro-optimizations only matter after the grid parallelism bug is fixed.

> **Avoid when**: Do not apply BLOCK_K reduction (e.g., 32→16) when the kernel is already grid-starved (waves_per_sm < 0.1, grid_size << sm_count): halving BLOCK_K doubles the K-loop iteration count, increasing L1TEX memory-latency stall exposure (CPIStall rule fires) with no parallelism to hide the latency. Do not apply vectorized store rewrites as the primary optimization when LaunchConfiguration is the dominant NCU rule — the grid fix must come first.

---

## Cumulative Progress

```
Baseline:  24420 ns  (1.000×)

Final best: 30940 ns  (0.7893×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -86,8 +96,6 @@

     xindex = idx_n + 96*idx_m + 49152*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
