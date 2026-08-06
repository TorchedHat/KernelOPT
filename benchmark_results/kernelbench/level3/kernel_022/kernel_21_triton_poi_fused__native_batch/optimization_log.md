# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4190 ns — **Best latency**: 4100 ns (**1.0220×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`persistent_sm_loop_to_eliminate_partial_wave_and_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2450,1,1), Waves/SM: 1.16
- **Registers/thread**: 22
- **Duration**: 4190 ns
- **SM Throughput**: 24.49 % | **Memory**: 13.02 % | **DRAM**: 12.28 %
- **Achieved Occupancy**: 63.80 %
- **IPC Active**: 2.01 inst/cycle | **No Eligible**: 49.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_larger_xblock_and_autotune | 1.002× | 0.381× | ✓ | **✓ best** |
| B | increase_xblock_tile_size_with_autotune | 0.947× | 0.909× | ✓ |  |
| C | eliminate_partial_wave_via_larger_xblock_autotune | 0.916× | 0.915× | ✓ |  |
| D | vectorized_loads_larger_xblock_autotune | 0.917× | 0.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_t... | Δ | eliminate_partial... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 11010 ns | +6820 ns (+162.8%) | 4610 ns | +420 ns (+10.0%) | 4580 ns | +390 ns (+9.3%) | 4580 ns | +390 ns (+9.3%) |
| SM Throughput (%) | 24.49 % | 68.98 % | +44.49 pp | 43.08 % | +18.59 pp | 43.24 % | +18.75 pp | 41.39 % | +16.90 pp |
| Memory Throughput (%) | 13.02 % | 16.18 % | +3.16 pp | 15.81 % | +2.79 pp | 15.86 % | +2.84 pp | 15.19 % | +2.17 pp |
| DRAM Throughput (%) | 12.28 % | 4.66 % | -7.62 pp | 11.17 % | -1.11 pp | 11.29 % | -0.99 pp | 11.27 % | -1.01 pp |
| Achieved Occupancy (%) | 63.80 % | 83.23 % | +19.43 pp | 69.86 % | +6.06 pp | 69.98 % | +6.18 pp | 69.86 % | +6.06 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 4.64 | +3.48 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 49.43 % | 15.63 % | -33.80 pp | 23.43 % | -26.00 pp | 23.77 % | -25.66 pp | 29.43 % | -20.00 pp |
| IPC Active | 2.01 inst/cycle | 3.35 inst/cycle | +1.34 | 3.00 inst/cycle | +0.99 | 2.99 inst/cycle | +0.98 | 2.99 inst/cycle | +0.98 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_and_static_grid_to_eliminate_partial_wave | 1.003× | 1.014× | ✓ | **✓ best** |
| B | persistent_tile_scheduling_fixed_grid_sm_loop | 0.984× | 0.957× | ✓ |  |
| C | fix_broken_autotune_grid_and_reduce_warp_count_for_partial_wave | 0.970× | 1.000× | ✓ |  |
| D | remove_autotune_restore_value_overhead_and_use_fixed_xblock_with_rsqrt_fusion | 0.978× | 0.948× | ✓ |  |
| E | add_triton_autotune | 0.978× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | persistent_tile_s... | Δ | fix_broken_autotu... | Δ | remove_autotune_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4130 ns | -60 ns (-1.4%) | 4380 ns | +190 ns (+4.5%) | 4190 ns | +0 ns (+0.0%) | 4420 ns | +230 ns (+5.5%) | 4290 ns | +100 ns (+2.4%) |
| SM Throughput (%) | 24.49 % | 25.15 % | +0.66 pp | 26.26 % | +1.77 pp | 24.19 % | -0.30 pp | 20.06 % | -4.43 pp | 24.73 % | +0.24 pp |
| Memory Throughput (%) | 13.02 % | 13.42 % | +0.40 pp | 13.10 % | +0.08 pp | 12.83 % | -0.19 pp | 13.86 % | +0.84 pp | 13.17 % | +0.15 pp |
| DRAM Throughput (%) | 12.28 % | 12.43 % | +0.15 pp | 11.81 % | -0.47 pp | 12.27 % | -0.01 pp | 11.74 % | -0.54 pp | 11.97 % | -0.31 pp |
| Achieved Occupancy (%) | 63.80 % | 63.58 % | -0.22 pp | 64.93 % | +1.13 pp | 62.30 % | -1.50 pp | 67.54 % | +3.74 pp | 63.06 % | -0.74 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 21.00 register/thread | -1 | 22.00 register/thread | +0 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 49.43 % | 48.02 % | -1.41 pp | 50.78 % | +1.35 pp | 48.55 % | -0.88 pp | 67.46 % | +18.03 pp | 48.08 % | -1.35 pp |
| IPC Active | 2.01 inst/cycle | 2.04 inst/cycle | +0.03 | 2.05 inst/cycle | +0.04 | 1.98 inst/cycle | -0.03 | 1.30 inst/cycle | -0.71 | 1.95 inst/cycle | -0.06 |

---

## Iteration 3

**Best so far**: 1.0145× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_via_rsqrt_and_fma_rewrite | 1.014× | 0.957× | ✓ | **✓ best** |
| B | fuse_rsqrt_and_eliminate_nonfused_fp_instructions | 1.013× | 0.993× | ✓ |  |
| C | replace_reciprocal_sqrt_sequence_with_rsqrt_to_enable_fma_fusion | 0.986× | 1.022× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 0.974× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fuse_rsqrt_and_el... | Δ | replace_reciproca... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4380 ns | +190 ns (+4.5%) | 4220 ns | +30 ns (+0.7%) | 4100 ns | -90 ns (-2.1%) | 4320 ns | +130 ns (+3.1%) |
| SM Throughput (%) | 24.49 % | 18.64 % | -5.85 pp | 18.90 % | -5.59 pp | 22.24 % | -2.25 pp | 19.43 % | -5.06 pp |
| Memory Throughput (%) | 13.02 % | 12.92 % | -0.10 pp | 12.82 % | -0.20 pp | 13.69 % | +0.67 pp | 13.47 % | +0.45 pp |
| DRAM Throughput (%) | 12.28 % | 11.85 % | -0.43 pp | 12.18 % | -0.10 pp | 12.57 % | +0.29 pp | 11.87 % | -0.41 pp |
| Achieved Occupancy (%) | 63.80 % | 65.71 % | +1.91 pp | 65.08 % | +1.28 pp | 65.56 % | +1.76 pp | 64.24 % | +0.44 pp |
| Registers/Thread | 22.00 register/thread | 21.00 register/thread | -1 | 21.00 register/thread | -1 | 32.00 register/thread | +10 | 21.00 register/thread | -1 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 49.43 % | 63.98 % | +14.55 pp | 63.86 % | +14.43 pp | 59.82 % | +10.39 pp | 66.75 % | +17.32 pp |
| IPC Active | 2.01 inst/cycle | 1.42 inst/cycle | -0.59 | 1.35 inst/cycle | -0.66 | 1.58 inst/cycle | -0.43 | 1.41 inst/cycle | -0.60 |

---

## Iteration 4

**Best so far**: 1.0145× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_sm_grid_with_increased_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.986× | 1.014× | ✓ |  |
| B | persistent_sm_loop_kernel_fixed_grid | 1.000× | 1.000× | ✓ |  |
| C | persistent_sm_loop_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.016× | 1.022× | ✓ | **✓ best** |
| D | persistent_sm_loop_grid_reduction_to_eliminate_partial_wave | 0.967× | 0.957× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_sm_gri... | Δ | persistent_sm_loo... | Δ | persistent_sm_loo... | Δ | persistent_sm_loo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4130 ns | -60 ns (-1.4%) | 4190 ns | +0 ns (+0.0%) | 4100 ns | -90 ns (-2.1%) | 4380 ns | +190 ns (+4.5%) |
| SM Throughput (%) | 24.49 % | 21.71 % | -2.78 pp | 23.72 % | -0.77 pp | 22.50 % | -1.99 pp | 24.42 % | -0.07 pp |
| Memory Throughput (%) | 13.02 % | 13.40 % | +0.38 pp | 13.03 % | +0.01 pp | 13.73 % | +0.71 pp | 13.28 % | +0.26 pp |
| DRAM Throughput (%) | 12.28 % | 12.44 % | +0.16 pp | 12.29 % | +0.01 pp | 12.57 % | +0.29 pp | 11.75 % | -0.53 pp |
| Achieved Occupancy (%) | 63.80 % | 65.11 % | +1.31 pp | 66.88 % | +3.08 pp | 64.97 % | +1.17 pp | 67.77 % | +3.97 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 49.43 % | 58.26 % | +8.83 pp | 62.24 % | +12.81 pp | 60.06 % | +10.63 pp | 58.65 % | +9.22 pp |
| IPC Active | 2.01 inst/cycle | 1.60 inst/cycle | -0.41 | 1.56 inst/cycle | -0.45 | 1.56 inst/cycle | -0.45 | 1.64 inst/cycle | -0.37 |

---

## Iteration 5

**Best so far**: 1.022× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_persistent_grid_to_true_sm_count_persistence | 0.984× | 1.014× | ✓ |  |
| B | pin_grid_to_sm_count_to_activate_persistent_while_loop | 1.014× | 0.984× | ✓ |  |
| C | pin_grid_to_sm_count_exploit_existing_while_loop | 0.967× | 0.948× | ✓ |  |
| D | activate_dormant_persistent_loop_with_sm_count_grid | 0.995× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_persis... | Δ | pin_grid_to_sm_co... | Δ | pin_grid_to_sm_co... | Δ | activate_dormant_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 4130 ns | -60 ns (-1.4%) | 4260 ns | +70 ns (+1.7%) | 4420 ns | +230 ns (+5.5%) | 4220 ns | +30 ns (+0.7%) |
| SM Throughput (%) | 24.49 % | 21.99 % | -2.50 pp | 22.27 % | -2.22 pp | 24.34 % | -0.15 pp | 23.60 % | -0.89 pp |
| Memory Throughput (%) | 13.02 % | 13.65 % | +0.63 pp | 13.63 % | +0.61 pp | 13.29 % | +0.27 pp | 12.83 % | -0.19 pp |
| DRAM Throughput (%) | 12.28 % | 12.53 % | +0.25 pp | 12.06 % | -0.22 pp | 11.71 % | -0.57 pp | 12.15 % | -0.13 pp |
| Achieved Occupancy (%) | 63.80 % | 71.40 % | +7.60 pp | 66.82 % | +3.02 pp | 67.27 % | +3.47 pp | 68.47 % | +4.67 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 49.43 % | 60.25 % | +10.82 pp | 58.94 % | +9.51 pp | 59.53 % | +10.10 pp | 61.24 % | +11.81 pp |
| IPC Active | 2.01 inst/cycle | 1.36 inst/cycle | -0.65 | 1.55 inst/cycle | -0.46 | 1.49 inst/cycle | -0.52 | 1.43 inst/cycle | -0.58 |

---

## Cumulative Progress

```
Baseline:  4190 ns  (1.000×)
Iter 2     4130 ns  (1.014×)  — fix_broken_autotune_and_static_grid_to_eliminate_partial_wave
Iter 4     4100 ns  (1.022×)  — persistent_sm_loop_to_eliminate_partial_wave_and_hide_l1tex_latency

Final best: 4100 ns  (1.0220×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
+def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK: tl.constexpr):
     xnumel = 313600
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 40)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
-    tmp17 = tmp15 + tmp16
-    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
+    pid = tl.program_id(0)
+    num_sms = tl.num_programs(0)
+    xoffset = pid * XBLOCK
+    while xoffset < xnumel:
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x2 = xindex
+        x0 = (xindex % 40)
+        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+        tmp16 = tl.load(in_ptr4 + (x2), xmask)
+        tmp2 = tmp0 - tmp1
+        tmp4 = tl.full([1], 1e-05, tl.float32)
+        tmp5 = tmp3 + tmp4
+        tmp10 = tl.rsqrt(tmp5)
+        tmp11 = tmp2 * tmp10
+        tmp13 = tmp11 * tmp12
+        tmp15 = tmp13 + tmp14
+        tmp17 = tmp15 + tmp16
+        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
+        xoffset += num_sms * XBLOCK
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
