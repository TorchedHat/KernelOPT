# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_layer_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4190 ns — **Best latency**: 2660 ns (**1.5752×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_autotune_to_explore_xblock_and_num_warps_for_grid_spread`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_layer_norm_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_040/kernel_1_triton_per_fused_native_layer_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 44
- **Duration**: 4190 ns
- **SM Throughput**: 0.14 % | **Memory**: 0.48 % | **DRAM**: 0.13 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 1.05 inst/cycle | **No Eligible**: 71.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_xblock_to_match_xnumel_and_increase_grid_parallelism | 0.959× | 0.903× | ✓ |  |
| D | fix_xblock_grid_underutilization | 1.001× | 0.957× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | fix_xblock_grid_u... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4640 ns | +450 ns (+10.7%) | 4380 ns | +190 ns (+4.5%) |
| SM Throughput (%) | 0.14 % | 0.13 % | -0.01 pp | 0.13 % | -0.01 pp |
| Memory Throughput (%) | 0.48 % | 0.48 % | +0.00 pp | 0.53 % | +0.05 pp |
| DRAM Throughput (%) | 0.13 % | 0.12 % | -0.01 pp | 0.13 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 44.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 71.17 % | 71.41 % | +0.24 pp | 73.30 % | +2.13 pp |
| IPC Active | 1.05 inst/cycle | 1.14 inst/cycle | +0.09 | 1.14 inst/cycle | +0.09 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_to_explore_xblock_and_num_warps_for_grid_spread | 1.377× | 1.575× | ✓ | **✓ best** |
| C | autotune_xblock_and_num_warps_to_eliminate_wasted_threads_and_increase_grid_parallelism | 1.302× | 1.791× | ✓ |  |
| D | fix_grid_underutilization_via_xblock_1_with_correct_indexing | 1.003× | 0.903× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_to_e... | Δ | autotune_xblock_a... | Δ | fix_grid_underuti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 2660 ns | -1530 ns (-36.5%) | 2340 ns | -1850 ns (-44.2%) | 4640 ns | +450 ns (+10.7%) |
| SM Throughput (%) | 0.14 % | 0.01 % | -0.13 pp | 0.01 % | -0.13 pp | 0.12 % | -0.02 pp |
| Memory Throughput (%) | 0.48 % | 0.39 % | -0.09 pp | 0.42 % | -0.06 pp | 0.44 % | -0.04 pp |
| DRAM Throughput (%) | 0.13 % | 0.04 % | -0.09 pp | 0.05 % | -0.08 pp | 0.12 % | -0.01 pp |
| Achieved Occupancy (%) | 6.19 % | 1.56 % | -4.63 pp | 1.56 % | -4.63 pp | 6.20 % | +0.01 pp |
| Registers/Thread | 44.00 register/thread | 18.00 register/thread | -26 | 18.00 register/thread | -26 | 44.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 71.17 % | 89.92 % | +18.75 pp | 88.28 % | +17.11 pp | 71.14 % | -0.03 pp |
| IPC Active | 1.05 inst/cycle | 0.11 inst/cycle | -0.94 | 0.12 inst/cycle | -0.93 | 0.98 inst/cycle | -0.07 |

---

## Iteration 3

**Best so far**: 1.5752× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_for_correct_autotune_dispatch | 1.285× | 1.637× | ✓ |  |
| B | fix_grid_lambda_to_propagate_autotuned_xblock_into_grid_size | 1.372× | 1.791× | ✓ |  |
| C | autotune_xblock_and_num_warps_to_maximize_warp_scheduler_utilization | 1.376× | 1.724× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 2560 ns | -1630 ns (-38.9%) | 2340 ns | -1850 ns (-44.2%) | 2430 ns | -1760 ns (-42.0%) |
| SM Throughput (%) | 0.14 % | 0.01 % | -0.13 pp | 0.01 % | -0.13 pp | 0.01 % | -0.13 pp |
| Memory Throughput (%) | 0.48 % | 0.39 % | -0.09 pp | 0.43 % | -0.05 pp | 0.41 % | -0.07 pp |
| DRAM Throughput (%) | 0.13 % | 0.04 % | -0.09 pp | 0.05 % | -0.08 pp | 0.04 % | -0.09 pp |
| Achieved Occupancy (%) | 6.19 % | 1.56 % | -4.63 pp | 1.56 % | -4.63 pp | 1.56 % | -4.63 pp |
| Registers/Thread | 44.00 register/thread | 18.00 register/thread | -26 | 18.00 register/thread | -26 | 18.00 register/thread | -26 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 71.17 % | 91.93 % | +20.76 pp | 90.09 % | +18.92 pp | 88.09 % | +16.92 pp |
| IPC Active | 1.05 inst/cycle | 0.09 inst/cycle | -0.96 | 0.10 inst/cycle | -0.95 | 0.10 inst/cycle | -0.95 |

---

## Iteration 4

**Best so far**: 1.5752× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_to_full_xnumel_with_optimal_num_warps | 0.964× | 0.948× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotuned_xblock_via_lambda | 1.286× | 1.791× | ✓ |  |
| C | fix_call_site_grid_hardcoded_divisor_to_use_autotuned_xblock | 0.429× | 1.703× | ✓ |  |
| D | add_triton_autotune | 0.549× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | fix_hardcoded_gri... | Δ | fix_call_site_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4420 ns | +230 ns (+5.5%) | 2340 ns | -1850 ns (-44.2%) | 2460 ns | -1730 ns (-41.3%) | 4190 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.14 % | 0.13 % | -0.01 pp | 0.01 % | -0.13 pp | 0.01 % | -0.13 pp | 0.14 % | +0.00 pp |
| Memory Throughput (%) | 0.48 % | 0.46 % | -0.02 pp | 0.42 % | -0.06 pp | 0.40 % | -0.08 pp | 0.48 % | +0.00 pp |
| DRAM Throughput (%) | 0.13 % | 0.13 % | +0.00 pp | 0.04 % | -0.09 pp | 0.04 % | -0.09 pp | 0.13 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 1.56 % | -4.63 pp | 1.56 % | -4.63 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 18.00 register/thread | -26 | 18.00 register/thread | -26 | 44.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 71.17 % | 74.36 % | +3.19 pp | 91.03 % | +19.86 pp | 88.05 % | +16.88 pp | 72.98 % | +1.81 pp |
| IPC Active | 1.05 inst/cycle | 1.14 inst/cycle | +0.09 | 0.12 inst/cycle | -0.93 | 0.10 inst/cycle | -0.95 | 1.14 inst/cycle | +0.09 |

---

## Cumulative Progress

```
Baseline:  4190 ns  (1.000×)
Iter 2     2660 ns  (1.575×)  — add_autotune_to_explore_xblock_and_num_warps_for_grid_spread

Final best: 2660 ns  (1.5752×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},  num_warps=1),
+        triton.Config({'XBLOCK': 1},  num_warps=2),
+        triton.Config({'XBLOCK': 2},  num_warps=1),
+        triton.Config({'XBLOCK': 2},  num_warps=2),
+        triton.Config({'XBLOCK': 4},  num_warps=1),
+        triton.Config({'XBLOCK': 4},  num_warps=2),
+        triton.Config({'XBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 8},  num_warps=2),
+        triton.Config({'XBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32}, num_warps=4),
+    ],
+    key=[],  # xnumel and r0_numel are compile-time constants
+)
 @triton.jit
 def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 16
@@ -38,8 +56,6 @@

     tl.store(out_ptr0 + (x0), tmp13, xmask)
     tl.store(out_ptr1 + (x0), tmp14, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -49,7 +65,7 @@

     xnumel = 16
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_native_layer_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 16, 32, XBLOCK=128)
+    triton_per_fused_native_layer_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 16, 32)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
