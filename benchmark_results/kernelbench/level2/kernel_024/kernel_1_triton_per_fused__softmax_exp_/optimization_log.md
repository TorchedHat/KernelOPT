# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_exp_prepare_softmax_online_sub_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11420 ns — **Best latency**: 9500 ns (**1.2021×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_register_pressure_via_num_warps_2_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_exp_prepare_softmax_online_sub_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_024/kernel_1_triton_per_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (900,1,1), Waves/SM: 0.76
- **Registers/thread**: 53
- **Duration**: 11420 ns
- **SM Throughput**: 20.76 % | **Memory**: 56.27 % | **DRAM**: 19.82 %
- **Achieved Occupancy**: 35.49 %
- **IPC Active**: 1.06 inst/cycle | **No Eligible**: 73.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 1.062× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 11420 ns | 11330 ns | -90 ns (-0.8%) |
| SM Throughput (%) | 20.76 % | 20.42 % | -0.34 pp |
| Memory Throughput (%) | 56.27 % | 55.35 % | -0.92 pp |
| DRAM Throughput (%) | 19.82 % | 19.94 % | +0.12 pp |
| Achieved Occupancy (%) | 35.49 % | 35.40 % | -0.09 pp |
| Registers/Thread | 53.00 register/thread | 53.00 register/thread | +0 |
| Waves/SM | 0.76 | 0.76 | +0.00 |
| No Eligible (%) | 73.04 % | 73.45 % | +0.41 pp |
| IPC Active | 1.06 inst/cycle | 1.05 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0079× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_load_store | 1.069× | 0.997× | ✓ | **✓ best** |
| D | increase_xblock_tile_size_for_better_sm_utilization | 1.061× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11420 ns | 11460 ns | +40 ns (+0.4%) | 11170 ns | -250 ns (-2.2%) |
| SM Throughput (%) | 20.76 % | 21.56 % | +0.80 pp | 20.81 % | +0.05 pp |
| Memory Throughput (%) | 56.27 % | 56.15 % | -0.12 pp | 56.36 % | +0.09 pp |
| DRAM Throughput (%) | 19.82 % | 19.74 % | -0.08 pp | 20.22 % | +0.40 pp |
| Achieved Occupancy (%) | 35.49 % | 35.69 % | +0.20 pp | 35.50 % | +0.01 pp |
| Registers/Thread | 53.00 register/thread | 53.00 register/thread | +0 | 53.00 register/thread | +0 |
| Waves/SM | 0.76 | 0.76 | +0.00 | 0.76 | +0.00 |
| No Eligible (%) | 73.04 % | 71.85 % | -1.19 pp | 72.82 % | -0.22 pp |
| IPC Active | 1.06 inst/cycle | 1.11 inst/cycle | +0.05 | 1.07 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0079× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | reduce_register_pressure_via_num_warps_2_autotune | 1.182× | 1.202× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 11420 ns | 9500 ns | -1920 ns (-16.8%) |
| SM Throughput (%) | 20.76 % | 22.41 % | +1.65 pp |
| Memory Throughput (%) | 56.27 % | 30.56 % | -25.71 pp |
| DRAM Throughput (%) | 19.82 % | 23.81 % | +3.99 pp |
| Achieved Occupancy (%) | 35.49 % | 17.90 % | -17.59 pp |
| Registers/Thread | 53.00 register/thread | 92.00 register/thread | +39 |
| Waves/SM | 0.76 | 0.68 | -0.08 |
| No Eligible (%) | 73.04 % | 66.91 % | -6.13 pp |
| IPC Active | 1.06 inst/cycle | 1.27 inst/cycle | +0.21 |

---

## Iteration 5

**Best so far**: 1.2021× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_swap_tile_axes_r0_contiguous | 1.154× | 1.214× | ✓ |  |
| D | expand_grid_via_smaller_xblock_num_warps_1_configs | 1.157× | 1.182× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | expand_grid_via_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11420 ns | 9410 ns | -2010 ns (-17.6%) | 9660 ns | -1760 ns (-15.4%) |
| SM Throughput (%) | 20.76 % | 22.78 % | +2.02 pp | 18.76 % | -2.00 pp |
| Memory Throughput (%) | 56.27 % | 30.99 % | -25.28 pp | 37.08 % | -19.19 pp |
| DRAM Throughput (%) | 19.82 % | 24.09 % | +4.27 pp | 23.66 % | +3.84 pp |
| Achieved Occupancy (%) | 35.49 % | 17.91 % | -17.58 pp | 9.53 % | -25.96 pp |
| Registers/Thread | 53.00 register/thread | 92.00 register/thread | +39 | 168.00 register/thread | +115 |
| Waves/SM | 0.76 | 0.68 | -0.08 | 0.57 | -0.19 |
| No Eligible (%) | 73.04 % | 67.87 % | -5.17 pp | 71.50 % | -1.54 pp |
| IPC Active | 1.06 inst/cycle | 1.27 inst/cycle | +0.21 | 1.06 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan B: `memory_coalescing_swap_tile_axes_r0_contiguous` (1.214×)

Swapping the broadcasting axes of a 2D Triton reduction tile — making the reduction dimension the row axis ([:, None]) and the output dimension the column axis ([None, :]), then changing the reduction axis from 1 to 0 — is intended to eliminate shared memory bank conflicts by placing consecutive reduction elements contiguously in shared memory. However, in this case the profiling metrics barely changed: SharedMemoryConflicts remained at 4.5-way bank conflicts with 77.78% excessive wavefronts, Warp Cycles Per Issued Instruction stayed at ~8.9, and Duration dropped only marginally from 9.5us to 9.41us (1.21x speedup). The axis swap alone does not resolve the bank conflict pattern when the underlying memory access stride and block geometry remain unchanged — the conflict is structural to the tile shape, not just the axis annotation order.

> **Applicable when**: Triton 2D reduction kernel where SharedMemoryConflicts is triggered (4+ way bank conflicts, >70% excessive wavefronts) and the reduction dimension is the column axis ([None, :]) while the output dimension is the row axis ([:, None]). The axis swap is a minimal, correctness-safe change that preserves all address formulas and masks. Yields a small but real speedup (~1.2x) even when bank conflicts are not fully eliminated.

> **Avoid when**: Do not expect this axis swap alone to eliminate bank conflicts when the tile geometry (XBLOCK, R0_BLOCK) and global memory stride pattern are the root cause of the conflict — the SharedMemoryConflicts metric may remain nearly unchanged. Do not apply when the kernel has already been restructured to use 1D reductions or register-only softmax. Do not apply when correctness is fragile (this kernel had 6 prior correctness failures on more aggressive restructurings).

#### ✅ Plan D: `expand_grid_via_smaller_xblock_num_warps_1_configs` (1.182×)

Adding num_warps=1 configs (XBLOCK=128) to a Triton @triton.autotune decorator caused the autotuner to select a 32-thread block, which eliminated shared memory bank conflicts (4.5-way → 0) and reduced Warp Cycles Per Issued Instruction from 8.91 to 5.76. However, the smaller block size also doubled register usage per thread (92 → 168 registers/thread), cutting theoretical occupancy from 31.25% to 18.75% and achieved occupancy from 17.9% to 9.53%, and the grid did not expand (waves/SM dropped from 0.68 to 0.57). The net result was a modest 1.18x speedup, substantially below the predicted 30-50%, because the occupancy loss partially offset the bank conflict reduction and the grid did not expand as expected since XBLOCK remained at 128 (not 32).

> **Applicable when**: Triton reduction kernel with SharedMemoryConflicts rule triggered (4+ way bank conflicts, >50% excessive wavefronts), where the bank conflicts are the dominant stall source (CPIStall on MIO >40% of cycles). The kernel already has num_warps=2 as the smallest config. Adding num_warps=1 can eliminate bank conflicts at the cost of lower occupancy.

> **Avoid when**: Do not apply when the kernel is already occupancy-limited (achieved occupancy < 15%) and waves_per_sm < 1.0, as further reducing num_warps will worsen occupancy without sufficient grid expansion to compensate. Also avoid when XBLOCK cannot be reduced below 128 (e.g., due to hardcoded grid constraints), since the grid will not expand and the occupancy loss dominates. The strategy yields diminishing returns after the first num_warps reduction step (from 4→2 gave 1.20x; from 2→1 gave only 1.18x).

---

## Cumulative Progress

```
Baseline:  11420 ns  (1.000×)
Iter 2     11330 ns  (1.008×)  — add_triton_autotune
Iter 4     9500 ns  (1.202×)  — reduce_register_pressure_via_num_warps_2_autotune

Final best: 9500 ns  (1.2021×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    # NEW: num_warps=2 gives larger register budget per warp (XBLOCK>=128 required for correctness with fixed grid=900)
+    triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),
+    triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=3),
+    # EXISTING: keep all original configs
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+    # NEW: num_stages=3 variants for software pipelining
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 115200
@@ -24,7 +40,7 @@

     x0 = (xindex % 900)
     x1 = xindex // 900
     x3 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), mask=r0_mask & xmask, other=0.0, eviction_policy='evict_first')
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
     tmp5 = tl.where(r0_mask & xmask, tmp3, float("-inf"))
@@ -37,16 +53,14 @@

     tmp13 = tmp0 - tmp6
     tmp14 = tl.exp(tmp13)
     tmp15 = (tmp14 / tmp12)
-    tl.store(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), tmp15, r0_mask & xmask)
-
-
+    tl.store(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), tmp15, mask=r0_mask & xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     xnumel = 115200
     r0_numel = 24
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24, XBLOCK=128)
+    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24)
     return arg_0
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
