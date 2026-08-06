# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_log_mul_sub_sum_xlogy_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 807300 ns — **Best latency**: 807300 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_log_mul_sub_sum_xlogy_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_098/kernel_0_triton_red_fused_log_mul_sub_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.72
- **Registers/thread**: 55
- **Duration**: 807300 ns
- **SM Throughput**: 82.14 % | **Memory**: 54.19 % | **DRAM**: 54.19 %
- **Achieved Occupancy**: 44.47 %
- **IPC Active**: 3.40 inst/cycle | **No Eligible**: 14.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_tile_sizes | 0.958× | 0.998× | ✓ |  |
| B | autotune_tile_sizes_to_reduce_register_pressure_and_partial_wave | 0.948× | 1.000× | ✓ |  |
| C | add_autotune_with_vectorized_coalesced_stores | 0.932× | 0.005× | ✓ |  |
| D | add_autotune_increase_xblock_for_coalesced_stores_and_occupancy | 0.954× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_tile_siz... | Δ | add_autotune_with... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 807300 ns | 808930 ns | +1630 ns (+0.2%) | 807620 ns | +320 ns (+0.0%) | 154810000 ns | +154002700 ns (+19076.3%) | 811170 ns | +3870 ns (+0.5%) |
| SM Throughput (%) | 82.14 % | 82.04 % | -0.10 pp | 82.19 % | +0.05 pp | 17.18 % | -64.96 pp | 81.98 % | -0.16 pp |
| Memory Throughput (%) | 54.19 % | 54.08 % | -0.11 pp | 54.17 % | -0.02 pp | 66.57 % | +12.38 pp | 53.93 % | -0.26 pp |
| DRAM Throughput (%) | 54.19 % | 54.08 % | -0.11 pp | 54.17 % | -0.02 pp | 66.57 % | +12.38 pp | 53.93 % | -0.26 pp |
| Achieved Occupancy (%) | 44.47 % | 44.36 % | -0.11 pp | 44.38 % | -0.09 pp | 93.88 % | +49.41 pp | 44.40 % | -0.07 pp |
| Registers/Thread | 55.00 register/thread | 55.00 register/thread | +0 | 55.00 register/thread | +0 | 32.00 register/thread | -23 | 55.00 register/thread | +0 |
| Waves/SM | 1.72 | 1.72 | +0.00 | 1.72 | +0.00 | 0.97 | -0.75 | 1.72 | +0.00 |
| No Eligible (%) | 14.99 % | 15.12 % | +0.13 pp | 15.10 % | +0.11 pp | 82.11 % | +67.12 pp | 15.09 % | +0.10 pp |
| IPC Active | 3.40 inst/cycle | 3.40 inst/cycle | +0.00 | 3.40 inst/cycle | +0.00 | 0.72 inst/cycle | -2.68 | 3.40 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan C: `add_autotune_with_vectorized_coalesced_stores` (0.005×)

When @triton.autotune is added but the grid lambda is NOT updated to use META['XBLOCK'] (grid remains hardcoded to cdiv(xnumel, 1)), the kernel is launched with 2048 blocks regardless of the autotuned XBLOCK value. Each block then redundantly processes the same output rows, causing executed instructions to balloon from ~520M to ~21B (40x increase) and duration to increase from 807 µs to 155 ms (192x regression). The fix requires the grid to be a lambda: `lambda META: (triton.cdiv(xnumel, META['XBLOCK']),)` so block count scales inversely with XBLOCK. Additionally, when autotune injects XBLOCK automatically, passing XBLOCK as an explicit kwarg in the kernel call causes a conflict; the explicit kwarg must be removed, but the grid must also be updated simultaneously.

> **Applicable when**: Any Triton kernel where @triton.autotune is added to vary XBLOCK (or any tiling parameter that affects grid size). The grid must always be a lambda consuming META to reflect the autotuned tile size.

> **Avoid when**: Never use a static/hardcoded grid when @triton.autotune varies a parameter that determines the number of blocks. This pattern always causes catastrophic regression: blocks multiply redundantly, instruction count explodes proportionally to (autotuned_XBLOCK / original_XBLOCK).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_tune_num_warps_num_stages | 0.950× | 0.998× | ✓ |  |
| B | fix_autotune_grid_lambda_to_eliminate_redundant_blocks | 0.946× | 0.993× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_larger_xblock_configs | 0.958× | 0.998× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_to_eliminate_redundant_block_execution | 0.941× | 0.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 807300 ns | 808860 ns | +1560 ns (+0.2%) | 812960 ns | +5660 ns (+0.7%) | 809340 ns | +2040 ns (+0.3%) | 155470000 ns | +154662700 ns (+19158.0%) |
| SM Throughput (%) | 82.14 % | 82.05 % | -0.09 pp | 81.99 % | -0.15 pp | 82.09 % | -0.05 pp | 17.14 % | -65.00 pp |
| Memory Throughput (%) | 54.19 % | 54.08 % | -0.11 pp | 53.81 % | -0.38 pp | 54.05 % | -0.14 pp | 66.26 % | +12.07 pp |
| DRAM Throughput (%) | 54.19 % | 54.08 % | -0.11 pp | 53.81 % | -0.38 pp | 54.05 % | -0.14 pp | 66.26 % | +12.07 pp |
| Achieved Occupancy (%) | 44.47 % | 44.41 % | -0.06 pp | 44.40 % | -0.07 pp | 44.42 % | -0.05 pp | 93.91 % | +49.44 pp |
| Registers/Thread | 55.00 register/thread | 55.00 register/thread | +0 | 55.00 register/thread | +0 | 55.00 register/thread | +0 | 32.00 register/thread | -23 |
| Waves/SM | 1.72 | 1.72 | +0.00 | 1.72 | +0.00 | 1.72 | +0.00 | 0.97 | -0.75 |
| No Eligible (%) | 14.99 % | 15.13 % | +0.14 pp | 15.16 % | +0.17 pp | 15.10 % | +0.11 pp | 82.10 % | +67.11 pp |
| IPC Active | 3.40 inst/cycle | 3.40 inst/cycle | +0.00 | 3.39 inst/cycle | -0.01 | 3.40 inst/cycle | +0.00 | 0.72 inst/cycle | -2.68 |

---

## Cumulative Progress

```
Baseline:  807300 ns  (1.000×)

Final best: 807300 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK':  512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK':  256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK':  512}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK':  256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK':  128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_log_mul_sub_sum_xlogy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -43,8 +56,6 @@

     tmp14 = tl.sum(_tmp14, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp14, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -52,7 +63,7 @@

     xnumel = 2048
     r0_numel = 131072
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_log_mul_sub_sum_xlogy_0[grid](arg_0, arg_1, arg_2, 2048, 131072, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_log_mul_sub_sum_xlogy_0[grid](arg_0, arg_1, arg_2, 2048, 131072)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
