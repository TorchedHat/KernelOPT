# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3490 ns — **Best latency**: 3200 ns (**1.0906×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_xblock_to_reduce_grid_and_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_43_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 0.58
- **Registers/thread**: 18
- **Duration**: 3490 ns
- **SM Throughput**: 15.40 % | **Memory**: 5.64 % | **DRAM**: 3.75 %
- **Achieved Occupancy**: 43.39 %
- **IPC Active**: 1.56 inst/cycle | **No Eligible**: 59.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.949× | 0.550× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_batch_norm | 0.931× | 1.020× | ✓ |  |
| C | increase_xblock_tile_size_for_better_utilization | 0.994× | 0.545× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.918× | 0.808× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_a... | Δ | increase_xblock_t... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 6340 ns | +2850 ns (+81.7%) | 3420 ns | -70 ns (-2.0%) | 6400 ns | +2910 ns (+83.4%) | 4320 ns | +830 ns (+23.8%) |
| SM Throughput (%) | 15.40 % | 55.83 % | +40.43 pp | 15.65 % | +0.25 pp | 52.35 % | +36.95 pp | 42.18 % | +26.78 pp |
| Memory Throughput (%) | 5.64 % | 11.05 % | +5.41 pp | 5.69 % | +0.05 pp | 11.06 % | +5.42 pp | 9.58 % | +3.94 pp |
| DRAM Throughput (%) | 3.75 % | 2.06 % | -1.69 pp | 3.80 % | +0.05 pp | 2.04 % | -1.71 pp | 3.04 % | -0.71 pp |
| Achieved Occupancy (%) | 43.39 % | 79.15 % | +35.76 pp | 43.21 % | -0.18 pp | 49.07 % | +5.68 pp | 72.10 % | +28.71 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 32.00 register/thread | +14 |
| Waves/SM | 0.58 | 2.32 | +1.74 | 0.58 | +0.00 | 1.86 | +1.28 | 1.16 | +0.58 |
| No Eligible (%) | 59.72 % | 17.76 % | -41.96 pp | 58.17 % | -1.55 pp | 23.58 % | -36.14 pp | 23.87 % | -35.85 pp |
| IPC Active | 1.56 inst/cycle | 3.21 inst/cycle | +1.65 | 1.57 inst/cycle | +0.01 | 3.08 inst/cycle | +1.52 | 2.79 inst/cycle | +1.23 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_tile_scaling | 1.001× | 1.009× | ✓ | **✓ best** |
| B | fix_grid_xblock_mismatch_remove_broken_autotune_use_static_tile | 1.000× | 1.020× | ✓ |  |
| C | remove_broken_autotune_restore_value_and_fix_grid_to_static_config | 0.955× | 1.048× | ✓ |  |
| D | remove_autotune_restore_value_to_fix_register_pressure_regression | 0.994× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_xblock_m... | Δ | remove_broken_aut... | Δ | remove_autotune_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3460 ns | -30 ns (-0.9%) | 3420 ns | -70 ns (-2.0%) | 3330 ns | -160 ns (-4.6%) | 3360 ns | -130 ns (-3.7%) |
| SM Throughput (%) | 15.40 % | 15.14 % | -0.26 pp | 15.36 % | -0.04 pp | 14.97 % | -0.43 pp | 15.40 % | +0.00 pp |
| Memory Throughput (%) | 5.64 % | 5.75 % | +0.11 pp | 5.71 % | +0.07 pp | 5.88 % | +0.24 pp | 5.87 % | +0.23 pp |
| DRAM Throughput (%) | 3.75 % | 3.82 % | +0.07 pp | 3.82 % | +0.07 pp | 4.02 % | +0.27 pp | 3.91 % | +0.16 pp |
| Achieved Occupancy (%) | 43.39 % | 46.95 % | +3.56 pp | 44.49 % | +1.10 pp | 48.29 % | +4.90 pp | 46.24 % | +2.85 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 59.72 % | 62.58 % | +2.86 pp | 62.17 % | +2.45 pp | 59.55 % | -0.17 pp | 55.44 % | -4.28 pp |
| IPC Active | 1.56 inst/cycle | 1.56 inst/cycle | +0.00 | 1.61 inst/cycle | +0.05 | 1.44 inst/cycle | -0.12 | 1.60 inst/cycle | +0.04 |

---

## Iteration 3

**Best so far**: 1.0087× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_no_autotune | 0.952× | 1.048× | ✓ |  |
| B | persistent_tile_scheduling_fixed_xblock_multi_tile_loop | 0.975× | 0.948× | ✓ |  |
| C | persistent_tile_scheduling_inner_loop | 0.986× | 1.030× | ✓ |  |
| D | persistent_tile_scheduling_remove_broken_autotune | 0.976× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3330 ns | -160 ns (-4.6%) | 3680 ns | +190 ns (+5.4%) | 3390 ns | -100 ns (-2.9%) | 3620 ns | +130 ns (+3.7%) |
| SM Throughput (%) | 15.40 % | 15.70 % | +0.30 pp | 15.05 % | -0.35 pp | 17.37 % | +1.97 pp | 15.10 % | -0.30 pp |
| Memory Throughput (%) | 5.64 % | 5.88 % | +0.24 pp | 5.39 % | -0.25 pp | 5.80 % | +0.16 pp | 5.41 % | -0.23 pp |
| DRAM Throughput (%) | 3.75 % | 3.92 % | +0.17 pp | 3.54 % | -0.21 pp | 3.86 % | +0.11 pp | 3.62 % | -0.13 pp |
| Achieved Occupancy (%) | 43.39 % | 46.41 % | +3.02 pp | 49.87 % | +6.48 pp | 44.32 % | +0.93 pp | 50.05 % | +6.66 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 59.72 % | 57.48 % | -2.24 pp | 62.45 % | +2.73 pp | 58.06 % | -1.66 pp | 59.08 % | -0.64 pp |
| IPC Active | 1.56 inst/cycle | 1.65 inst/cycle | +0.09 | 1.63 inst/cycle | +0.07 | 1.66 inst/cycle | +0.10 | 1.55 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0087× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_and_enable_fma_fusion | 1.014× | 1.039× | ✓ | **✓ best** |
| B | fix_persistent_kernel_grid_to_cap_at_sm_count | 0.943× | 1.030× | ✓ |  |
| C | fix_persistent_kernel_grid_to_enable_multi_tile_iteration_per_sm | 0.976× | 0.975× | ✓ |  |
| D | fix_persistent_kernel_grid_size_to_num_sms | 0.978× | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | fix_persistent_ke... | Δ | fix_persistent_ke... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3360 ns | -130 ns (-3.7%) | 3390 ns | -100 ns (-2.9%) | 3580 ns | +90 ns (+2.6%) | 3740 ns | +250 ns (+7.2%) |
| SM Throughput (%) | 15.40 % | 10.86 % | -4.54 pp | 17.16 % | +1.76 pp | 15.01 % | -0.39 pp | 15.02 % | -0.38 pp |
| Memory Throughput (%) | 5.64 % | 6.01 % | +0.37 pp | 5.81 % | +0.17 pp | 5.47 % | -0.17 pp | 5.26 % | -0.38 pp |
| DRAM Throughput (%) | 3.75 % | 3.87 % | +0.12 pp | 3.86 % | +0.11 pp | 3.65 % | -0.10 pp | 3.51 % | -0.24 pp |
| Achieved Occupancy (%) | 43.39 % | 42.11 % | -1.28 pp | 43.48 % | +0.09 pp | 50.30 % | +6.91 pp | 50.14 % | +6.75 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 59.72 % | 71.68 % | +11.96 pp | 54.04 % | -5.68 pp | 58.55 % | -1.17 pp | 58.63 % | -1.09 pp |
| IPC Active | 1.56 inst/cycle | 1.05 inst/cycle | -0.51 | 1.62 inst/cycle | +0.06 | 1.50 inst/cycle | -0.06 | 1.51 inst/cycle | -0.05 |

---

## Iteration 5

**Best so far**: 1.0387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_reduce_grid_and_hide_l1tex_latency | 1.013× | 1.091× | ✓ |  |
| B | algebraic_simplification_rsqrt_fma_fusion_to_reduce_non_fused_fp32_instructions | 0.971× | 1.091× | ✓ |  |
| C | fuse_sqrt_reciprocal_to_rsqrt_eliminate_nonfused_fp32_instructions | 1.015× | 1.091× | ✓ | **✓ best** |
| D | fix_dead_persistent_loop_grid_and_enable_fma_via_rsqrt | 0.978× | 1.101× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | algebraic_simplif... | Δ | fuse_sqrt_recipro... | Δ | fix_dead_persiste... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3200 ns | -290 ns (-8.3%) | 3200 ns | -290 ns (-8.3%) | 3200 ns | -290 ns (-8.3%) | 3170 ns | -320 ns (-9.2%) |
| SM Throughput (%) | 15.40 % | 11.11 % | -4.29 pp | 13.26 % | -2.14 pp | 12.26 % | -3.14 pp | 12.24 % | -3.16 pp |
| Memory Throughput (%) | 5.64 % | 6.32 % | +0.68 pp | 6.32 % | +0.68 pp | 6.23 % | +0.59 pp | 6.28 % | +0.64 pp |
| DRAM Throughput (%) | 3.75 % | 4.11 % | +0.36 pp | 4.07 % | +0.32 pp | 4.09 % | +0.34 pp | 4.10 % | +0.35 pp |
| Achieved Occupancy (%) | 43.39 % | 43.53 % | +0.14 pp | 45.74 % | +2.35 pp | 44.10 % | +0.71 pp | 44.31 % | +0.92 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 59.72 % | 73.85 % | +14.13 pp | 66.79 % | +7.07 pp | 68.16 % | +8.44 pp | 68.61 % | +8.89 pp |
| IPC Active | 1.56 inst/cycle | 0.97 inst/cycle | -0.59 | 1.22 inst/cycle | -0.34 | 1.20 inst/cycle | -0.36 | 1.20 inst/cycle | -0.36 |

---

## Cumulative Progress

```
Baseline:  3490 ns  (1.000×)
Iter 2     3460 ns  (1.009×)  — fix_autotune_grid_lambda_to_enable_correct_tile_scaling
Iter 4     3360 ns  (1.039×)  — replace_sqrt_div_with_rsqrt_and_enable_fma_fusion
Iter 5     3200 ns  (1.091×)  — fuse_sqrt_reciprocal_to_rsqrt_eliminate_nonfused_fp32_instructions

Final best: 3200 ns  (1.0906×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 156800
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 320)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    sm_id = tl.program_id(0)
+    tile_id = sm_id
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x2 = xindex
+        x0 = (xindex % 320)
+        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+        tmp2 = tmp0 - tmp1
+        tmp4 = tl.full([1], 1e-05, tl.float32)
+        tmp5 = tmp3 + tmp4
+        tmp10 = libdevice.rsqrt(tmp5)
+        tmp11 = tmp2 * tmp10
+        tmp13 = tmp11 * tmp12
+        tmp15 = tmp13 + tmp14
+        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
+        tile_id += tl.num_programs(0)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
