# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4000 ns — **Best latency**: 3840 ns (**1.0417×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_tl_maximum_with_tl_where_for_fp32_instruction_fusion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_7_triton_poi_fused_addmm_relu_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2688,1,1), Waves/SM: 1.27
- **Registers/thread**: 16
- **Duration**: 4000 ns
- **SM Throughput**: 11.58 % | **Memory**: 9.59 % | **DRAM**: 7.10 %
- **Achieved Occupancy**: 54.05 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 73.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 1.016× | 1.000× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave_and_improve_cache_reuse | 1.056× | 1.058× | ✓ | **✓ best** |
| C | vectorized_loads_stores_with_larger_xblock | 1.026× | 1.050× | ✓ |  |
| D | increase_xblock_to_eliminate_partial_wave_tail | 0.993× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_e... | Δ | vectorized_loads_... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 4000 ns | +0 ns (+0.0%) | 3780 ns | -220 ns (-5.5%) | 3810 ns | -190 ns (-4.7%) | 3970 ns | -30 ns (-0.7%) |
| SM Throughput (%) | 11.58 % | 11.62 % | +0.04 pp | 14.65 % | +3.07 pp | 16.15 % | +4.57 pp | 11.45 % | -0.13 pp |
| Memory Throughput (%) | 9.59 % | 9.78 % | +0.19 pp | 9.73 % | +0.14 pp | 9.64 % | +0.05 pp | 9.59 % | +0.00 pp |
| DRAM Throughput (%) | 7.10 % | 7.14 % | +0.04 pp | 7.50 % | +0.40 pp | 7.43 % | +0.33 pp | 7.11 % | +0.01 pp |
| Achieved Occupancy (%) | 54.05 % | 54.21 % | +0.16 pp | 57.89 % | +3.84 pp | 63.54 % | +9.49 pp | 60.11 % | +6.06 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.27 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 |
| No Eligible (%) | 73.98 % | 70.20 % | -3.78 pp | 47.38 % | -26.60 pp | 44.22 % | -29.76 pp | 71.95 % | -2.03 pp |
| IPC Active | 1.10 inst/cycle | 1.10 inst/cycle | +0.00 | 1.93 inst/cycle | +0.83 | 1.86 inst/cycle | +0.76 | 0.94 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.0582× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_override_to_enable_larger_xblock | 1.046× | 1.042× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_xblock_selection | 1.054× | 1.050× | ✓ |  |
| C | autotune_larger_xblock_with_evict_first_on_streaming_tensor | 1.054× | 0.985× | ✓ |  |
| D | autotune_xblock_with_evict_first_on_streaming_load | 1.024× | 1.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | autotune_larger_x... | Δ | autotune_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3840 ns | -160 ns (-4.0%) | 3810 ns | -190 ns (-4.7%) | 4060 ns | +60 ns (+1.5%) | 3810 ns | -190 ns (-4.7%) |
| SM Throughput (%) | 11.58 % | 14.75 % | +3.17 pp | 16.14 % | +4.56 pp | 15.50 % | +3.92 pp | 15.91 % | +4.33 pp |
| Memory Throughput (%) | 9.59 % | 9.44 % | -0.15 pp | 9.71 % | +0.12 pp | 9.22 % | -0.37 pp | 9.77 % | +0.18 pp |
| DRAM Throughput (%) | 7.10 % | 7.36 % | +0.26 pp | 7.42 % | +0.32 pp | 7.00 % | -0.10 pp | 7.45 % | +0.35 pp |
| Achieved Occupancy (%) | 54.05 % | 58.05 % | +4.00 pp | 59.33 % | +5.28 pp | 57.33 % | +3.28 pp | 57.82 % | +3.77 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.27 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 |
| No Eligible (%) | 73.98 % | 48.00 % | -25.98 pp | 44.71 % | -29.27 pp | 44.18 % | -29.80 pp | 44.41 % | -29.57 pp |
| IPC Active | 1.10 inst/cycle | 1.95 inst/cycle | +0.85 | 2.09 inst/cycle | +0.99 | 2.09 inst/cycle | +0.99 | 2.07 inst/cycle | +0.97 |

---

## Iteration 3

**Best so far**: 1.0582× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_to_autotune_configs_and_fix_grid | 1.053× | 1.034× | ✓ |  |
| B | fix_grid_override_and_add_num_stages_for_latency_hiding | 1.024× | 1.042× | ✓ |  |
| C | repair_callsite_to_honor_existing_autotune | 1.027× | 1.042× | ✓ |  |
| D | fix_callsite_to_activate_existing_autotune_configs | 1.059× | 1.050× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | fix_grid_override... | Δ | repair_callsite_t... | Δ | fix_callsite_to_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3870 ns | -130 ns (-3.2%) | 3840 ns | -160 ns (-4.0%) | 3840 ns | -160 ns (-4.0%) | 3810 ns | -190 ns (-4.7%) |
| SM Throughput (%) | 11.58 % | 16.10 % | +4.52 pp | 16.10 % | +4.52 pp | 14.70 % | +3.12 pp | 16.17 % | +4.59 pp |
| Memory Throughput (%) | 9.59 % | 9.65 % | +0.06 pp | 9.71 % | +0.12 pp | 9.47 % | -0.12 pp | 9.68 % | +0.09 pp |
| DRAM Throughput (%) | 7.10 % | 7.32 % | +0.22 pp | 7.43 % | +0.33 pp | 7.41 % | +0.31 pp | 7.43 % | +0.33 pp |
| Achieved Occupancy (%) | 54.05 % | 58.41 % | +4.36 pp | 62.27 % | +8.22 pp | 57.77 % | +3.72 pp | 58.43 % | +4.38 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.27 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 |
| No Eligible (%) | 73.98 % | 44.39 % | -29.59 pp | 49.59 % | -24.39 pp | 55.75 % | -18.23 pp | 54.16 % | -19.82 pp |
| IPC Active | 1.10 inst/cycle | 2.08 inst/cycle | +0.98 | 1.92 inst/cycle | +0.82 | 1.94 inst/cycle | +0.84 | 2.08 inst/cycle | +0.98 |

---

## Iteration 4

**Best so far**: 1.0582× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hardcode_optimal_xblock_and_remove_broken_autotune | 0.988× | 1.000× | ✓ |  |
| B | expand_autotune_configs_with_higher_num_warps_and_fused_relu_add | 1.034× | 1.026× | ✓ |  |
| C | fuse_fp32_instructions_and_fix_callsite_grid_for_autotune | 1.054× | 1.042× | ✓ |  |
| D | persistent_cta_loop_to_eliminate_partial_wave | 0.993× | 0.976× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hardcode_optimal_... | Δ | expand_autotune_c... | Δ | fuse_fp32_instruc... | Δ | persistent_cta_lo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 4000 ns | +0 ns (+0.0%) | 3900 ns | -100 ns (-2.5%) | 3840 ns | -160 ns (-4.0%) | 4100 ns | +100 ns (+2.5%) |
| SM Throughput (%) | 11.58 % | 11.98 % | +0.40 pp | 15.84 % | +4.26 pp | 17.44 % | +5.86 pp | 16.51 % | +4.93 pp |
| Memory Throughput (%) | 9.59 % | 9.76 % | +0.17 pp | 9.45 % | -0.14 pp | 9.81 % | +0.22 pp | 9.44 % | -0.15 pp |
| DRAM Throughput (%) | 7.10 % | 7.10 % | +0.00 pp | 7.27 % | +0.17 pp | 7.42 % | +0.32 pp | 6.91 % | -0.19 pp |
| Achieved Occupancy (%) | 54.05 % | 56.68 % | +2.63 pp | 58.80 % | +4.75 pp | 60.09 % | +6.04 pp | 57.44 % | +3.39 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.27 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 |
| No Eligible (%) | 73.98 % | 66.38 % | -7.60 pp | 44.43 % | -29.55 pp | 38.53 % | -35.45 pp | 60.45 % | -13.53 pp |
| IPC Active | 1.10 inst/cycle | 1.22 inst/cycle | +0.12 | 2.08 inst/cycle | +0.98 | 2.24 inst/cycle | +1.14 | 1.56 inst/cycle | +0.46 |

---

## Iteration 5

**Best so far**: 1.0582× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_smaller_xblock_with_higher_num_warps_for_occupancy_recovery | 1.059× | 0.905× | ✓ |  |
| B | replace_tl_maximum_with_tl_where_for_fp32_instruction_fusion | 1.063× | 1.042× | ✓ | **✓ best** |
| C | autotune_num_warps_and_num_stages_for_latency_hiding | 1.052× | 1.008× | ✓ |  |
| D | increase_num_warps_via_xblock512_hardcode_to_reduce_partial_wave_and_hide_l1tex_latency | 0.985× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_smaller_... | Δ | replace_tl_maximu... | Δ | autotune_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 4420 ns | +420 ns (+10.5%) | 3840 ns | -160 ns (-4.0%) | 3970 ns | -30 ns (-0.7%) | 4060 ns | +60 ns (+1.5%) |
| SM Throughput (%) | 11.58 % | 23.89 % | +12.31 pp | 17.48 % | +5.90 pp | 14.11 % | +2.53 pp | 16.34 % | +4.76 pp |
| Memory Throughput (%) | 9.59 % | 10.76 % | +1.17 pp | 9.72 % | +0.13 pp | 9.15 % | -0.44 pp | 9.44 % | -0.15 pp |
| DRAM Throughput (%) | 7.10 % | 6.44 % | -0.66 pp | 7.37 % | +0.27 pp | 7.10 % | +0.00 pp | 6.98 % | -0.12 pp |
| Achieved Occupancy (%) | 54.05 % | 69.33 % | +15.28 pp | 60.55 % | +6.50 pp | 60.59 % | +6.54 pp | 57.15 % | +3.10 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.27 | 2.55 | +1.28 | 1.27 | +0.00 | 1.27 | +0.00 | 1.27 | +0.00 |
| No Eligible (%) | 73.98 % | 54.90 % | -19.08 pp | 39.95 % | -34.03 pp | 51.91 % | -22.07 pp | 60.05 % | -13.93 pp |
| IPC Active | 1.10 inst/cycle | 1.66 inst/cycle | +0.56 | 2.25 inst/cycle | +1.15 | 1.78 inst/cycle | +0.68 | 1.56 inst/cycle | +0.46 |

---

## Cumulative Progress

```
Baseline:  4000 ns  (1.000×)
Iter 1     3780 ns  (1.058×)  — increase_xblock_eliminate_partial_wave_and_improve_cache_reuse

Final best: 3840 ns  (1.0417×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
@@ -13,19 +33,17 @@

     x0 = (xindex % 84)
     x2 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp2 = tmp0 + tmp1
-    tmp3 = tl.full([1], 0, tl.int32)
-    tmp4 = tl.maximum(tmp3, tmp2)
+    tmp4 = tl.where(tmp2 > 0.0, tmp2, 0.0)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 344064
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_relu_7[grid](arg_0, arg_1, 344064, XBLOCK=128)
+    triton_poi_fused_addmm_relu_7[grid](arg_0, arg_1, 344064)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
