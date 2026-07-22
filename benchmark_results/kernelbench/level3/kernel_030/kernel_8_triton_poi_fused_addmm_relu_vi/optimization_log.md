# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2560 ns (**1.0117×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_register_pressure_via_smaller_warps_to_restore_block_limit_registers`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_view_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_8_triton_poi_fused_addmm_relu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (676,1,1), Waves/SM: 0.32
- **Registers/thread**: 16
- **Duration**: 2590 ns
- **SM Throughput**: 4.44 % | **Memory**: 4.16 % | **DRAM**: 2.78 %
- **Achieved Occupancy**: 25.62 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 84.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.024× | 0.963× | ✓ |  |
| B | autotune_xblock_to_increase_occupancy_and_reduce_launch_overhead | 1.028× | 0.899× | ✓ | **✓ best** |
| C | add_autotune_with_larger_xblock_and_vectorized_loads | 1.008× | 0.918× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_t... | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2690 ns | +100 ns (+3.9%) | 2880 ns | +290 ns (+11.2%) | 2820 ns | +230 ns (+8.9%) |
| SM Throughput (%) | 4.44 % | 5.37 % | +0.93 pp | 14.77 % | +10.33 pp | 13.42 % | +8.98 pp |
| Memory Throughput (%) | 4.16 % | 3.79 % | -0.37 pp | 7.06 % | +2.90 pp | 6.44 % | +2.28 pp |
| DRAM Throughput (%) | 2.78 % | 2.68 % | -0.10 pp | 2.51 % | -0.27 pp | 2.57 % | -0.21 pp |
| Achieved Occupancy (%) | 25.62 % | 17.59 % | -8.03 pp | 45.78 % | +20.16 pp | 43.93 % | +18.31 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 24.00 register/thread | +8 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.64 | +0.32 | 0.64 | +0.32 |
| No Eligible (%) | 84.80 % | 79.55 % | -5.25 pp | 48.06 % | -36.74 pp | 55.65 % | -29.15 pp |
| IPC Active | 0.54 inst/cycle | 0.71 inst/cycle | +0.17 | 1.53 inst/cycle | +0.99 | 1.75 inst/cycle | +1.21 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_and_add_persistent_kernel_with_correct_xblock_grid | 1.025× | 0.963× | ✓ |  |
| B | fix_grid_launch_to_use_autotuned_xblock_and_tune_warps_stages | 1.034× | 0.836× | ✓ | **✓ best** |
| C | fix_hardcoded_grid_and_add_num_stages_to_autotune | 1.005× | 0.881× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ | fix_grid_launch_t... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2690 ns | +100 ns (+3.9%) | 3100 ns | +510 ns (+19.7%) | 2940 ns | +350 ns (+13.5%) |
| SM Throughput (%) | 4.44 % | 5.41 % | +0.97 pp | 14.86 % | +10.42 pp | 15.31 % | +10.87 pp |
| Memory Throughput (%) | 4.16 % | 3.80 % | -0.36 pp | 7.12 % | +2.96 pp | 7.31 % | +3.15 pp |
| DRAM Throughput (%) | 2.78 % | 2.69 % | -0.09 pp | 2.34 % | -0.44 pp | 2.46 % | -0.32 pp |
| Achieved Occupancy (%) | 25.62 % | 17.32 % | -8.30 pp | 44.20 % | +18.58 pp | 43.68 % | +18.06 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 24.00 register/thread | +8 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.64 | +0.32 | 0.64 | +0.32 |
| No Eligible (%) | 84.80 % | 84.94 % | +0.14 pp | 47.90 % | -36.90 pp | 48.98 % | -35.82 pp |
| IPC Active | 0.54 inst/cycle | 0.53 inst/cycle | -0.01 | 1.87 inst/cycle | +1.33 | 1.83 inst/cycle | +1.29 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_formula_and_add_streaming_eviction_for_main_tensor | 0.994× | 0.989× | ✓ |  |
| B | fix_grid_lambda_and_add_small_xblock_low_register_configs | 1.038× | 0.918× | ✓ |  |
| C | reduce_register_pressure_via_smaller_warps_to_restore_block_limit_registers | 1.067× | 1.012× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_formula_... | Δ | fix_grid_lambda_a... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2620 ns | +30 ns (+1.2%) | 2820 ns | +230 ns (+8.9%) | 2560 ns | -30 ns (-1.2%) |
| SM Throughput (%) | 4.44 % | 4.85 % | +0.41 pp | 15.34 % | +10.90 pp | 2.39 % | -2.05 pp |
| Memory Throughput (%) | 4.16 % | 4.03 % | -0.13 pp | 7.34 % | +3.18 pp | 3.98 % | -0.18 pp |
| DRAM Throughput (%) | 2.78 % | 2.77 % | -0.01 pp | 2.56 % | -0.22 pp | 2.81 % | +0.03 pp |
| Achieved Occupancy (%) | 25.62 % | 27.28 % | +1.66 pp | 47.05 % | +21.43 pp | 12.77 % | -12.85 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.64 | +0.32 | 0.16 | -0.16 |
| No Eligible (%) | 84.80 % | 81.90 % | -2.90 pp | 48.24 % | -36.56 pp | 92.90 % | +8.10 pp |
| IPC Active | 0.54 inst/cycle | 0.49 inst/cycle | -0.05 | 1.56 inst/cycle | +1.02 | 0.30 inst/cycle | -0.24 |

---

## Iteration 4

**Best so far**: 1.0117× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_dynamic_grid_and_expand_autotune_configs_for_full_coverage | 1.025× | 0.963× | ✓ |  |
| B | fix_hardcoded_grid_and_increase_warps_for_occupancy | 1.007× | 0.989× | ✓ |  |
| C | fix_grid_lambda_use_meta_xblock_and_add_streaming_eviction_on_main_load | 1.043× | 0.794× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_dynamic_grid_... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_u... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2690 ns | +100 ns (+3.9%) | 2620 ns | +30 ns (+1.2%) | 3260 ns | +670 ns (+25.9%) |
| SM Throughput (%) | 4.44 % | 8.94 % | +4.50 pp | 5.02 % | +0.58 pp | 29.53 % | +25.09 pp |
| Memory Throughput (%) | 4.16 % | 3.88 % | -0.28 pp | 3.91 % | -0.25 pp | 10.92 % | +6.76 pp |
| DRAM Throughput (%) | 2.78 % | 2.69 % | -0.09 pp | 2.75 % | -0.03 pp | 2.23 % | -0.55 pp |
| Achieved Occupancy (%) | 25.62 % | 42.28 % | +16.66 pp | 27.03 % | +1.41 pp | 74.02 % | +48.40 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 26.00 register/thread | +10 |
| Waves/SM | 0.32 | 0.64 | +0.32 | 0.32 | +0.00 | 1.28 | +0.96 |
| No Eligible (%) | 84.80 % | 68.79 % | -16.01 pp | 81.76 % | -3.04 pp | 32.34 % | -52.46 pp |
| IPC Active | 0.54 inst/cycle | 1.16 inst/cycle | +0.62 | 0.50 inst/cycle | -0.04 | 2.59 inst/cycle | +2.05 |

---

## Iteration 5

**Best so far**: 1.0117× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_tune_warps_for_sm_coverage | 1.021× | 1.000× | ✓ |  |
| B | fix_grid_lambda_and_add_evict_first_for_streaming_load | 1.023× | 0.881× | ✓ |  |
| C | fix_grid_lambda_to_eliminate_partial_wave_tail | 0.993× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2590 ns | +0 ns (+0.0%) | 2940 ns | +350 ns (+13.5%) | 2660 ns | +70 ns (+2.7%) |
| SM Throughput (%) | 4.44 % | 2.62 % | -1.82 pp | 9.56 % | +5.12 pp | 11.41 % | +6.97 pp |
| Memory Throughput (%) | 4.16 % | 2.26 % | -1.90 pp | 3.69 % | -0.47 pp | 4.06 % | -0.10 pp |
| DRAM Throughput (%) | 2.78 % | 1.49 % | -1.29 pp | 2.44 % | -0.34 pp | 2.74 % | -0.04 pp |
| Achieved Occupancy (%) | 25.62 % | 13.30 % | -12.32 pp | 43.05 % | +17.43 pp | 45.14 % | +19.52 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.16 | -0.16 | 0.64 | +0.32 | 0.64 | +0.32 |
| No Eligible (%) | 84.80 % | 89.09 % | +4.29 pp | 73.82 % | -10.98 pp | 58.24 % | -26.56 pp |
| IPC Active | 0.54 inst/cycle | 0.34 inst/cycle | -0.20 | 1.30 inst/cycle | +0.76 | 1.26 inst/cycle | +0.72 |

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)
Iter 3     2560 ns  (1.012×)  — reduce_register_pressure_via_smaller_warps_to_restore_block_limit_registers

Final best: 2560 ns  (1.0117×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_relu_view_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 86528
@@ -14,20 +25,18 @@

     x0 = (xindex % 512)
     x2 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 86528
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_relu_view_8[grid](arg_0, arg_1, 86528, XBLOCK=128)
+    triton_poi_fused_addmm_relu_view_8[grid](arg_0, arg_1, 86528)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
