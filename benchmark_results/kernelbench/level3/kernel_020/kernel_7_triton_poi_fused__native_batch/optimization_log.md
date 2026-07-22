# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17090 ns — **Best latency**: 16770 ns (**1.0191×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_2d_grid_launch_for_channel_iteration`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 17
- **Duration**: 17090 ns
- **SM Throughput**: 51.26 % | **Memory**: 22.72 % | **DRAM**: 14.37 %
- **Achieved Occupancy**: 56.41 %
- **IPC Active**: 2.37 inst/cycle | **No Eligible**: 40.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 0.998× | 1.000× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.997× | 0.714× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock | 0.748× | 0.713× | ✓ |  |
| D | autotune_xblock_tile_size_for_latency_hiding | 0.914× | 0.898× | ✓ |  |
| E | add_triton_autotune | 1.010× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 17090 ns | +0 ns (+0.0%) | 23940 ns | +6850 ns (+40.1%) | 23970 ns | +6880 ns (+40.3%) | 19040 ns | +1950 ns (+11.4%) | 17180 ns | +90 ns (+0.5%) |
| SM Throughput (%) | 51.26 % | 60.49 % | +9.23 pp | 78.75 % | +27.49 pp | 78.77 % | +27.51 pp | 70.72 % | +19.46 pp | 60.31 % | +9.05 pp |
| Memory Throughput (%) | 22.72 % | 22.94 % | +0.22 pp | 23.52 % | +0.80 pp | 23.44 % | +0.72 pp | 27.24 % | +4.52 pp | 22.89 % | +0.17 pp |
| DRAM Throughput (%) | 14.37 % | 14.36 % | -0.01 pp | 10.26 % | -4.11 pp | 10.25 % | -4.12 pp | 12.88 % | -1.49 pp | 14.30 % | -0.07 pp |
| Achieved Occupancy (%) | 56.41 % | 61.81 % | +5.40 pp | 86.13 % | +29.72 pp | 86.41 % | +30.00 pp | 68.80 % | +12.39 pp | 60.81 % | +4.40 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.34 % | 30.96 % | -9.38 pp | 14.05 % | -26.29 pp | 13.74 % | -26.60 pp | 17.16 % | -23.18 pp | 29.48 % | -10.86 pp |
| IPC Active | 2.37 inst/cycle | 2.77 inst/cycle | +0.40 | 3.43 inst/cycle | +1.06 | 3.42 inst/cycle | +1.05 | 3.31 inst/cycle | +0.94 | 2.78 inst/cycle | +0.41 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.714×)

Increasing XBLOCK via autotune to expose more independent memory requests per CTA improved eligible warps per scheduler (1.28→7.12) and achieved occupancy (56%→86%), but the selected config doubled register usage per thread (17→32), which halved the block limit from registers (21→16 blocks/SM) and more than doubled total executed instructions (6.9M→14.8M). The net effect was a 40% longer kernel duration (17→24 µs) despite better scheduler utilization. When a pointwise kernel already has low register pressure and high instruction count per element, enlarging XBLOCK inflates instruction count faster than latency hiding benefits materialize, causing a regression.

> **Applicable when**: Kernel has CPIStall/L1TEX stalls >40% of CPI, low eligible warps per scheduler (<2), and achieved occupancy well below theoretical — AND register pressure is already high (>24 registers/thread) so larger tiles won't increase register usage further.

> **Avoid when**: Kernel has very low register usage per thread (≤20 registers/thread) where larger XBLOCK configs cause the compiler to allocate significantly more registers, reducing block-limit-registers and inflating total instruction count beyond the latency-hiding benefit. Also avoid when the kernel body has a long arithmetic chain per element that scales linearly with XBLOCK, as instruction count growth dominates.

#### ❌ Plan C: `vectorized_loads_stores_via_larger_xblock` (0.713×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) caused registers per thread to jump from 17 to 32, which more than doubled total executed instructions (6.87M → 14.79M) and increased kernel duration from 17.09 µs to 23.97 µs despite improving occupancy (56% → 86%) and scheduler eligibility. The root cause is that larger XBLOCK increases per-thread work (more loop iterations or wider unrolled code), and the autotune overhead plus the instruction count explosion outweighed any vectorization benefit. When the original kernel already has low register pressure and the bottleneck is L1TEX stalls rather than pure memory bandwidth, simply enlarging the tile size does not produce vectorized loads — it instead multiplies arithmetic and address-computation instructions per thread, worsening total work.

> **Applicable when**: Never: this specific pattern (autotune with larger XBLOCK on a kernel that already has low register count and non-coalesced NHWC access) caused regression. Avoid when the access pattern is non-contiguous (stride != 1 in the innermost dimension) because larger XBLOCK does not fix non-coalescing — it only increases instruction count.

> **Avoid when**: Kernel has non-unit strides in the innermost access dimension (NHWC / channel-last layout); original register count is already low (≤20 registers/thread) and increasing XBLOCK doubles register usage; grid is not updated to match the new XBLOCK (causing over-subscription or under-subscription); the bottleneck is L1TEX scoreboard stalls due to non-coalesced access rather than insufficient ILP — larger tiles do not fix non-coalescing.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_and_num_warps_increase_to_reduce_cpi_stall | 1.009× | 1.011× | ✓ |  |
| B | precompute_per_channel_scale_to_eliminate_redundant_sqrt_div | 1.003× | 1.008× | ✓ |  |
| C | replace_int_reciprocal_with_rsqrt_to_enable_fma_fusion | 1.015× | 1.002× | ✓ | **✓ best** |
| D | fma_fusion_via_arithmetic_simplification | 1.009× | 1.011× | ✓ |  |
| E | add_triton_autotune | 1.009× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_nu... | Δ | precompute_per_ch... | Δ | replace_int_recip... | Δ | fma_fusion_via_ar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16900 ns | -190 ns (-1.1%) | 16960 ns | -130 ns (-0.8%) | 17060 ns | -30 ns (-0.2%) | 16900 ns | -190 ns (-1.1%) | 17120 ns | +30 ns (+0.2%) |
| SM Throughput (%) | 51.26 % | 36.67 % | -14.59 pp | 36.00 % | -15.26 pp | 41.56 % | -9.70 pp | 36.71 % | -14.55 pp | 60.53 % | +9.27 pp |
| Memory Throughput (%) | 22.72 % | 22.67 % | -0.05 pp | 22.61 % | -0.11 pp | 30.80 % | +8.08 pp | 22.69 % | -0.03 pp | 22.94 % | +0.22 pp |
| DRAM Throughput (%) | 14.37 % | 14.55 % | +0.18 pp | 14.50 % | +0.13 pp | 14.42 % | +0.05 pp | 14.54 % | +0.17 pp | 14.35 % | -0.02 pp |
| Achieved Occupancy (%) | 56.41 % | 43.10 % | -13.31 pp | 43.52 % | -12.89 pp | 33.83 % | -22.58 pp | 42.77 % | -13.64 pp | 61.63 % | +5.22 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.34 % | 56.91 % | +16.57 pp | 57.68 % | +17.34 pp | 48.71 % | +8.37 pp | 56.79 % | +16.45 pp | 29.84 % | -10.50 pp |
| IPC Active | 2.37 inst/cycle | 1.72 inst/cycle | -0.65 | 1.68 inst/cycle | -0.69 | 2.03 inst/cycle | -0.34 | 1.70 inst/cycle | -0.67 | 2.80 inst/cycle | +0.43 |

---

## Iteration 3

**Best so far**: 1.0018× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_modulo_via_2d_grid_decomposition | 1.041× | 1.027× | ✓ | **✓ best** |
| B | precompute_rsqrt_as_separate_input_to_eliminate_transcendental_stall | 1.021× | 1.011× | ✓ |  |
| C | fix_broken_autotune_grid_and_restructure_2d_nhwc_tiling | 0.497× | 0.463× | ✓ |  |
| D | shared_memory_precompute_bn_channel_params | 0.913× | 0.885× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_modulo_... | Δ | precompute_rsqrt_... | Δ | fix_broken_autotu... | Δ | shared_memory_pre... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16640 ns | -450 ns (-2.6%) | 16900 ns | -190 ns (-1.1%) | 36930 ns | +19840 ns (+116.1%) | 19300 ns | +2210 ns (+12.9%) |
| SM Throughput (%) | 51.26 % | 39.96 % | -11.30 pp | 34.83 % | -16.43 pp | 68.50 % | +17.24 pp | 70.08 % | +18.82 pp |
| Memory Throughput (%) | 22.72 % | 19.33 % | -3.39 pp | 22.37 % | -0.35 pp | 43.62 % | +20.90 pp | 79.86 % | +57.14 pp |
| DRAM Throughput (%) | 14.37 % | 2.46 % | -11.91 pp | 14.54 % | +0.17 pp | 6.64 % | -7.73 pp | 12.74 % | -1.63 pp |
| Achieved Occupancy (%) | 56.41 % | 15.47 % | -40.94 pp | 41.94 % | -14.47 pp | 44.14 % | -12.27 pp | 85.48 % | +29.07 pp |
| Registers/Thread | 17.00 register/thread | 20.00 register/thread | +3 | 18.00 register/thread | +1 | 64.00 register/thread | +47 | 21.00 register/thread | +4 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 22.27 | +11.13 | 11.14 | +0.00 |
| No Eligible (%) | 40.34 % | 57.94 % | +17.60 pp | 58.75 % | +18.41 pp | 31.57 % | -8.77 pp | 47.72 % | +7.38 pp |
| IPC Active | 2.37 inst/cycle | 1.65 inst/cycle | -0.72 | 1.65 inst/cycle | -0.72 | 2.73 inst/cycle | +0.36 | 2.08 inst/cycle | -0.29 |

---

## Iteration 4

**Best so far**: 1.027× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_2d_grid_launch_for_channel_iteration | 1.042× | 1.019× | ✓ | **✓ best** |
| B | evict_first_store_to_free_l1_for_channel_stat_broadcasts | 1.021× | 1.014× | ✓ |  |
| C | eliminate_tl_gather_to_remove_shared_memory_bank_conflicts | 1.014× | 1.000× | ✓ |  |
| D | revert_2d_tiling_to_1d_pointwise_with_correct_autotune_grid | 1.005× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_2d_grid_launc... | Δ | evict_first_store... | Δ | eliminate_tl_gath... | Δ | revert_2d_tiling_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16770 ns | -320 ns (-1.9%) | 16860 ns | -230 ns (-1.3%) | 17090 ns | +0 ns (+0.0%) | 16860 ns | -230 ns (-1.3%) |
| SM Throughput (%) | 51.26 % | 39.42 % | -11.84 pp | 37.39 % | -13.87 pp | 41.50 % | -9.76 pp | 33.83 % | -17.43 pp |
| Memory Throughput (%) | 22.72 % | 19.04 % | -3.68 pp | 22.67 % | -0.05 pp | 30.88 % | +8.16 pp | 22.28 % | -0.44 pp |
| DRAM Throughput (%) | 14.37 % | 2.44 % | -11.93 pp | 14.54 % | +0.17 pp | 14.38 % | +0.01 pp | 14.55 % | +0.18 pp |
| Achieved Occupancy (%) | 56.41 % | 15.46 % | -40.95 pp | 42.44 % | -13.97 pp | 33.72 % | -22.69 pp | 41.75 % | -14.66 pp |
| Registers/Thread | 17.00 register/thread | 20.00 register/thread | +3 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.34 % | 57.71 % | +17.37 pp | 55.89 % | +15.55 pp | 49.51 % | +9.17 pp | 65.58 % | +25.24 pp |
| IPC Active | 2.37 inst/cycle | 1.65 inst/cycle | -0.72 | 1.76 inst/cycle | -0.61 | 2.03 inst/cycle | -0.34 | 1.38 inst/cycle | -0.99 |

---

## Iteration 5

**Best so far**: 1.027× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_channel_innermost_flat_1d_reindex | 1.019× | 1.011× | ✓ |  |
| B | reduce_xblock_with_maxnreg_cap_to_improve_l1tex_hit_rate | 1.016× | 1.006× | ✓ |  |
| C | pin_channel_stats_in_l1_via_evict_last_and_stream_main_tensor_via_evict_first | 1.000× | 1.011× | ✓ |  |
| D | precompute_invstd_in_caller_to_eliminate_transcendental_from_hot_path | 1.003× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_channel... | Δ | reduce_xblock_wit... | Δ | pin_channel_stats... | Δ | precompute_invstd... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16900 ns | -190 ns (-1.1%) | 16990 ns | -100 ns (-0.6%) | 16900 ns | -190 ns (-1.1%) | 16960 ns | -130 ns (-0.8%) |
| SM Throughput (%) | 51.26 % | 37.33 % | -13.93 pp | 37.94 % | -13.32 pp | 40.10 % | -11.16 pp | 26.65 % | -24.61 pp |
| Memory Throughput (%) | 22.72 % | 22.78 % | +0.06 pp | 18.99 % | -3.73 pp | 22.60 % | -0.12 pp | 21.24 % | -1.48 pp |
| DRAM Throughput (%) | 14.37 % | 14.53 % | +0.16 pp | 3.62 % | -10.75 pp | 14.54 % | +0.17 pp | 14.49 % | +0.12 pp |
| Achieved Occupancy (%) | 56.41 % | 43.48 % | -12.93 pp | 40.57 % | -15.84 pp | 44.46 % | -11.95 pp | 31.18 % | -25.23 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 21.00 register/thread | +4 | 18.00 register/thread | +1 | 16.00 register/thread | -1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.34 % | 56.59 % | +16.25 pp | 55.36 % | +15.02 pp | 52.68 % | +12.34 pp | 65.29 % | +24.95 pp |
| IPC Active | 2.37 inst/cycle | 1.75 inst/cycle | -0.62 | 1.78 inst/cycle | -0.59 | 1.89 inst/cycle | -0.48 | 1.35 inst/cycle | -1.02 |

---

## Cumulative Progress

```
Baseline:  17090 ns  (1.000×)
Iter 2     17060 ns  (1.002×)  — replace_int_reciprocal_with_rsqrt_to_enable_fma_fusion
Iter 3     16640 ns  (1.027×)  — eliminate_modulo_via_2d_grid_decomposition

Final best: 16770 ns  (1.0191×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 3010560
+    SPATIAL: tl.constexpr = 31360  # 3010560 // 96
+    x0 = tl.program_id(1).to(tl.int32)           # channel [0..95], zero-latency scalar
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x2 = xindex
-    x0 = (xindex % 96)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    xindex = xoffset + tl.arange(0, XBLOCK)[:]    # spatial index within tile
+    xmask = xindex < SPATIAL
+    x2 = xindex * 96 + x0                         # flat NHWC index
+    # Broadcast x0 to block shape so loads can accept xmask
+    x0_block = tl.full([XBLOCK], x0, tl.int32)
+    # Issue per-channel loads — x0 always in [0,95], mask guards spatial bounds
+    tmp1  = tl.load(in_ptr0 + x0_block, mask=xmask, eviction_policy='evict_last')
+    tmp3  = tl.load(in_ptr1 + x0_block, mask=xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + x0_block, mask=xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + x0_block, mask=xmask, eviction_policy='evict_last')
+    tmp0  = tl.load(in_out_ptr0 + x2, mask=xmask)  # main vector load
+    tmp_eps = tl.full([1], 1e-05, tl.float32)
+    tmp_var_eps = tmp3 + tmp_eps
+    tmp_inv_std = tl.math.rsqrt(tmp_var_eps)
+    tmp_xnorm = (tmp0 - tmp1) * tmp_inv_std
+    tmp_out = tl.math.fma(tmp_xnorm, tmp12, tmp14)
+    tmp_zero = tl.full([1], 0.0, tl.float32)
+    tmp_six  = tl.full([1], 6.0, tl.float32)
+    tmp_out  = tl.minimum(tl.maximum(tmp_out, tmp_zero), tmp_six)
+    tl.store(in_out_ptr0 + x2, tmp_out, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
