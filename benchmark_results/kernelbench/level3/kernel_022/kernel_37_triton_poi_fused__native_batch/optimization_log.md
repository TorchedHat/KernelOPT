# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 3070 ns (**0.9902×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_37_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (735,1,1), Waves/SM: 0.35
- **Registers/thread**: 18
- **Duration**: 3040 ns
- **SM Throughput**: 10.09 % | **Memory**: 4.02 % | **DRAM**: 2.62 %
- **Achieved Occupancy**: 28.16 %
- **IPC Active**: 1.06 inst/cycle | **No Eligible**: 71.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_larger_xblock | 0.981× | 0.709× | ✓ |  |
| B | autotune_xblock_num_warps_and_complete_hardtanh_fusion | 0.989× | 0.813× | ✓ |  |
| C | autotune_xblock_for_underutilized_pointwise_kernel | 1.032× | 0.905× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 1.036× | 0.913× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_n... | Δ | autotune_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 4290 ns | +1250 ns (+41.1%) | 3740 ns | +700 ns (+23.0%) | 3360 ns | +320 ns (+10.5%) | 3330 ns | +290 ns (+9.5%) |
| SM Throughput (%) | 10.09 % | 25.24 % | +15.15 pp | 31.04 % | +20.95 pp | 16.86 % | +6.77 pp | 17.44 % | +7.35 pp |
| Memory Throughput (%) | 4.02 % | 6.00 % | +1.98 pp | 6.89 % | +2.87 pp | 4.62 % | +0.60 pp | 4.78 % | +0.76 pp |
| DRAM Throughput (%) | 2.62 % | 1.86 % | -0.76 pp | 2.12 % | -0.50 pp | 2.35 % | -0.27 pp | 2.38 % | -0.24 pp |
| Achieved Occupancy (%) | 28.16 % | 24.03 % | -4.13 pp | 51.16 % | +23.00 pp | 21.85 % | -6.31 pp | 23.71 % | -4.45 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 0.35 | 0.56 | +0.21 | 0.70 | +0.35 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 71.69 % | 48.54 % | -23.15 pp | 32.26 % | -39.43 pp | 59.03 % | -12.66 pp | 59.18 % | -12.51 pp |
| IPC Active | 1.06 inst/cycle | 2.02 inst/cycle | +0.96 | 2.61 inst/cycle | +1.55 | 1.58 inst/cycle | +0.52 | 1.46 inst/cycle | +0.40 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_grid_lambda_to_enable_correct_autotune_tile_selection | 1.005× | 0.968× | ✓ |  |
| B | fix_static_grid_to_lambda_with_register_safe_configs | 0.991× | 0.959× | ✓ |  |
| C | persistent_loop_within_fixed_cta_grid | 0.995× | 0.968× | ✓ |  |
| D | remove_autotune_to_restore_low_register_pressure | 1.050× | 0.990× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_grid_l... | Δ | fix_static_grid_t... | Δ | persistent_loop_w... | Δ | remove_autotune_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 3140 ns | +100 ns (+3.3%) | 3170 ns | +130 ns (+4.3%) | 3140 ns | +100 ns (+3.3%) | 3070 ns | +30 ns (+1.0%) | 3040 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 10.09 % | 7.31 % | -2.78 pp | 7.27 % | -2.82 pp | 11.41 % | +1.32 pp | 10.18 % | +0.09 pp | 10.14 % | +0.05 pp |
| Memory Throughput (%) | 4.02 % | 4.03 % | +0.01 pp | 4.03 % | +0.01 pp | 3.91 % | -0.11 pp | 3.94 % | -0.08 pp | 4.01 % | -0.01 pp |
| DRAM Throughput (%) | 2.62 % | 2.53 % | -0.09 pp | 2.51 % | -0.11 pp | 2.55 % | -0.07 pp | 2.58 % | -0.04 pp | 2.61 % | -0.01 pp |
| Achieved Occupancy (%) | 28.16 % | 15.44 % | -12.72 pp | 15.07 % | -13.09 pp | 30.75 % | +2.59 pp | 30.80 % | +2.64 pp | 29.37 % | +1.21 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 24.00 register/thread | +6 | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.35 | 0.17 | -0.18 | 0.17 | -0.18 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 71.69 % | 79.21 % | +7.52 pp | 81.18 % | +9.49 pp | 69.76 % | -1.93 pp | 72.62 % | +0.93 pp | 70.91 % | -0.78 pp |
| IPC Active | 1.06 inst/cycle | 0.78 inst/cycle | -0.28 | 0.74 inst/cycle | -0.32 | 1.09 inst/cycle | +0.03 | 0.98 inst/cycle | -0.08 | 1.12 inst/cycle | +0.06 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_complete_hardtanh_fusion | 0.987× | 0.968× | ✓ |  |
| B | increase_xblock_to_reduce_grid_and_improve_warp_utilization | 1.043× | 0.990× | ✓ |  |
| C | fix_persistent_grid_launch_to_num_sms_with_larger_xblock | 0.993× | 0.981× | ✓ |  |
| D | persistent_sm_loop_for_underutilized_pointwise_kernel | 0.998× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | increase_xblock_t... | Δ | fix_persistent_gr... | Δ | persistent_sm_loo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 3140 ns | +100 ns (+3.3%) | 3070 ns | +30 ns (+1.0%) | 3100 ns | +60 ns (+2.0%) | 3100 ns | +60 ns (+2.0%) |
| SM Throughput (%) | 10.09 % | 10.54 % | +0.45 pp | 10.37 % | +0.28 pp | 11.63 % | +1.54 pp | 11.43 % | +1.34 pp |
| Memory Throughput (%) | 4.02 % | 3.91 % | -0.11 pp | 4.13 % | +0.11 pp | 3.87 % | -0.15 pp | 4.07 % | +0.05 pp |
| DRAM Throughput (%) | 2.62 % | 2.52 % | -0.10 pp | 2.58 % | -0.04 pp | 2.53 % | -0.09 pp | 2.53 % | -0.09 pp |
| Achieved Occupancy (%) | 28.16 % | 28.23 % | +0.07 pp | 27.95 % | -0.21 pp | 30.66 % | +2.50 pp | 28.76 % | +0.60 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 71.69 % | 74.24 % | +2.55 pp | 69.06 % | -2.63 pp | 70.56 % | -1.13 pp | 69.82 % | -1.87 pp |
| IPC Active | 1.06 inst/cycle | 1.09 inst/cycle | +0.03 | 1.07 inst/cycle | +0.01 | 1.09 inst/cycle | +0.03 | 1.14 inst/cycle | +0.08 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_float4_memory_access_to_hide_l1tex_latency | 0.954× | 1.010× | ✓ |  |
| B | fma_fusion_via_rsqrt_with_corrected_persistent_grid | 1.016× | 1.045× | ✓ |  |
| C | persistent_sm_grid_with_hardtanh_completion_and_vectorized_xblock | 1.034× | 0.959× | ✓ |  |
| D | preload_channel_broadcasts_into_shared_memory_to_eliminate_l1tex_stalls | 0.987× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_float4... | Δ | fma_fusion_via_rs... | Δ | persistent_sm_gri... | Δ | preload_channel_b... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 3010 ns | -30 ns (-1.0%) | 2910 ns | -130 ns (-4.3%) | 3170 ns | +130 ns (+4.3%) | 3390 ns | +350 ns (+11.5%) |
| SM Throughput (%) | 10.09 % | 10.54 % | +0.45 pp | 9.41 % | -0.68 pp | 11.10 % | +1.01 pp | 15.75 % | +5.66 pp |
| Memory Throughput (%) | 4.02 % | 4.01 % | -0.01 pp | 4.36 % | +0.34 pp | 3.99 % | -0.03 pp | 11.54 % | +7.52 pp |
| DRAM Throughput (%) | 2.62 % | 2.63 % | +0.01 pp | 2.71 % | +0.09 pp | 2.50 % | -0.12 pp | 2.31 % | -0.31 pp |
| Achieved Occupancy (%) | 28.16 % | 28.16 % | +0.00 pp | 30.03 % | +1.87 pp | 29.81 % | +1.65 pp | 32.71 % | +4.55 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 22.00 register/thread | +4 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 71.69 % | 68.59 % | -3.10 pp | 75.44 % | +3.75 pp | 67.51 % | -4.18 pp | 73.68 % | +1.99 pp |
| IPC Active | 1.06 inst/cycle | 1.09 inst/cycle | +0.03 | 0.88 inst/cycle | -0.18 | 1.17 inst/cycle | +0.11 | 1.03 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)

Final best: 3070 ns  (0.9902×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -29,9 +30,9 @@

     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
+    tmp16 = triton_helpers.maximum(tl.full([XBLOCK], 0.0, tl.float32), tmp15)
+    tmp17 = triton_helpers.minimum(tl.full([XBLOCK], 6.0, tl.float32), tmp16)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
