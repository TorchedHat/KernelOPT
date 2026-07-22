# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 19260 ns — **Best latency**: 19140 ns (**1.0063×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27000,1,1), Waves/SM: 12.78
- **Registers/thread**: 18
- **Duration**: 19260 ns
- **SM Throughput**: 61.19 % | **Memory**: 23.21 % | **DRAM**: 14.65 %
- **Achieved Occupancy**: 60.80 %
- **IPC Active**: 2.80 inst/cycle | **No Eligible**: 29.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.000× | 0.212× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.989× | 1.000× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads_and_better_occupancy | 0.906× | 0.213× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy_and_latency_hiding | 0.999× | 0.709× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 90690 ns | +71430 ns (+370.9%) | 19260 ns | +0 ns (+0.0%) | 90430 ns | +71170 ns (+369.5%) | 27170 ns | +7910 ns (+41.1%) |
| SM Throughput (%) | 61.19 % | 86.91 % | +25.72 pp | 61.56 % | +0.37 pp | 85.58 % | +24.39 pp | 79.29 % | +18.10 pp |
| Memory Throughput (%) | 23.21 % | 17.91 % | -5.30 pp | 23.33 % | +0.12 pp | 17.79 % | -5.42 pp | 23.83 % | +0.62 pp |
| DRAM Throughput (%) | 14.65 % | 3.11 % | -11.54 pp | 14.64 % | -0.01 pp | 3.11 % | -11.54 pp | 10.38 % | -4.27 pp |
| Achieved Occupancy (%) | 60.80 % | 85.68 % | +24.88 pp | 60.99 % | +0.19 pp | 55.81 % | -4.99 pp | 86.49 % | +25.69 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 32.00 register/thread | +14 |
| Waves/SM | 12.78 | 51.14 | +38.36 | 12.78 | +0.00 | 40.91 | +28.13 | 12.78 | +0.00 |
| No Eligible (%) | 29.54 % | 11.05 % | -18.49 pp | 30.28 % | +0.74 pp | 12.21 % | -17.33 pp | 13.46 % | -16.08 pp |
| IPC Active | 2.80 inst/cycle | 3.57 inst/cycle | +0.77 | 2.82 inst/cycle | +0.02 | 3.50 inst/cycle | +0.70 | 3.44 inst/cycle | +0.64 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_increase_tile_size` (0.212×)

Adding @triton.autotune with larger XBLOCK configs to a pointwise Triton kernel caused a 4.7x slowdown (19 µs → 91 µs). Although achieved occupancy improved from 60.8% to 85.7% and compute SM throughput rose from 61% to 87%, the autotuner selected a block size (512) that increased registers per thread from 18 to 32, which reduced blocks-per-SM from 21 to 4 (register-limited), and the total executed instruction count exploded from 9.3M to 62.3M — indicating the larger tile size caused massive redundant work or boundary overhead. The grid was not updated to use meta['XBLOCK'] dynamically (it remained hardcoded to cdiv(xnumel, 128)), so the autotuner ran all 9 configs against a fixed grid of 27,000 blocks, causing each config with XBLOCK>128 to process far more elements than exist, multiplying instruction count by the ratio XBLOCK/128.

> **Applicable when**: Never applicable in this form. The pattern of adding @triton.autotune while keeping a hardcoded grid is always wrong.

> **Avoid when**: Always: when adding @triton.autotune, the grid lambda MUST use meta['XBLOCK'] dynamically. If the grid remains hardcoded to cdiv(xnumel, FIXED_BLOCK), the autotuner will launch too many blocks for larger XBLOCK configs, causing each block to redundantly process elements already handled by other blocks, exploding instruction count and duration.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_tune_num_warps_at_fixed_xblock | 1.000× | 0.990× | ✓ |  |
| B | eliminate_redundant_fp32_ops_and_expose_fma_fusion | 1.007× | 0.951× | ✓ | **✓ best** |
| C | fix_broken_autotune_grid_and_revert_to_static_baseline_config | 0.988× | 0.995× | ✓ |  |
| D | fix_hardcoded_grid_and_fuse_fp32_instructions_into_fma | 1.000× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | eliminate_redunda... | Δ | fix_broken_autotu... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19460 ns | +200 ns (+1.0%) | 20260 ns | +1000 ns (+5.2%) | 19360 ns | +100 ns (+0.5%) | 19460 ns | +200 ns (+1.0%) |
| SM Throughput (%) | 61.19 % | 45.12 % | -16.07 pp | 49.55 % | -11.64 pp | 61.74 % | +0.55 pp | 61.88 % | +0.69 pp |
| Memory Throughput (%) | 23.21 % | 20.67 % | -2.54 pp | 34.09 % | +10.88 pp | 23.42 % | +0.21 pp | 23.47 % | +0.26 pp |
| DRAM Throughput (%) | 14.65 % | 14.48 % | -0.17 pp | 13.93 % | -0.72 pp | 14.57 % | -0.08 pp | 14.50 % | -0.15 pp |
| Achieved Occupancy (%) | 60.80 % | 39.50 % | -21.30 pp | 39.76 % | -21.04 pp | 61.01 % | +0.21 pp | 61.20 % | +0.40 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 12.78 | 6.39 | -6.39 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.54 % | 45.06 % | +15.52 pp | 43.35 % | +13.81 pp | 29.59 % | +0.05 pp | 29.72 % | +0.18 pp |
| IPC Active | 2.80 inst/cycle | 2.19 inst/cycle | -0.61 | 2.31 inst/cycle | -0.49 | 2.79 inst/cycle | -0.01 | 2.80 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_chains_into_explicit_fma_instructions | 0.999× | 0.998× | ✓ |  |
| B | replace_integer_division_with_rsqrt_to_enable_fma_fusion | 0.997× | 1.010× | ✓ |  |
| C | replace_sqrt_reciprocal_with_rsqrt_to_fuse_fp32_instructions | 0.997× | 1.000× | ✓ |  |
| D | fix_autotune_grid_lambda_to_enable_correct_xblock_exploration | 1.009× | 0.954× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | replace_integer_d... | Δ | replace_sqrt_reci... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19300 ns | +40 ns (+0.2%) | 19070 ns | -190 ns (-1.0%) | 19260 ns | +0 ns (+0.0%) | 20190 ns | +930 ns (+4.8%) |
| SM Throughput (%) | 61.19 % | 36.73 % | -24.46 pp | 21.93 % | -39.26 pp | 37.38 % | -23.81 pp | 49.55 % | -11.64 pp |
| Memory Throughput (%) | 23.21 % | 22.66 % | -0.55 pp | 21.14 % | -2.07 pp | 23.08 % | -0.13 pp | 34.05 % | +10.84 pp |
| DRAM Throughput (%) | 14.65 % | 14.62 % | -0.03 pp | 14.78 % | +0.13 pp | 14.62 % | -0.03 pp | 13.97 % | -0.68 pp |
| Achieved Occupancy (%) | 60.80 % | 42.85 % | -17.95 pp | 28.72 % | -32.08 pp | 42.64 % | -18.16 pp | 40.15 % | -20.65 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 6.39 | -6.39 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.54 % | 57.34 % | +27.80 pp | 72.77 % | +43.23 pp | 56.80 % | +27.26 pp | 43.30 % | +13.76 pp |
| IPC Active | 2.80 inst/cycle | 1.71 inst/cycle | -1.09 | 1.06 inst/cycle | -1.74 | 1.72 inst/cycle | -1.08 | 2.29 inst/cycle | -0.51 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_fix_grid_lambda_for_latency_hiding | 0.998× | 1.002× | ✓ |  |
| B | add_streaming_eviction_policy_on_large_tensor_load_to_reduce_l1tex_stalls | 1.009× | 1.013× | ✓ |  |
| C | add_evict_first_on_streaming_main_tensor_load | 0.998× | 1.012× | ✓ |  |
| D | fix_hardcoded_grid_lambda_and_add_streaming_eviction_for_main_tensor | 1.008× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | add_streaming_evi... | Δ | add_evict_first_o... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19230 ns | -30 ns (-0.2%) | 19010 ns | -250 ns (-1.3%) | 19040 ns | -220 ns (-1.1%) | 20160 ns | +900 ns (+4.7%) |
| SM Throughput (%) | 61.19 % | 37.44 % | -23.75 pp | 40.77 % | -20.42 pp | 41.02 % | -20.17 pp | 52.08 % | -9.11 pp |
| Memory Throughput (%) | 23.21 % | 23.12 % | -0.09 pp | 22.99 % | -0.22 pp | 23.14 % | -0.07 pp | 33.88 % | +10.67 pp |
| DRAM Throughput (%) | 14.65 % | 14.66 % | +0.01 pp | 14.82 % | +0.17 pp | 14.81 % | +0.16 pp | 13.97 % | -0.68 pp |
| Achieved Occupancy (%) | 60.80 % | 42.34 % | -18.46 pp | 43.18 % | -17.62 pp | 43.44 % | -17.36 pp | 40.81 % | -19.99 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.54 % | 57.44 % | +27.90 pp | 52.64 % | +23.10 pp | 52.85 % | +23.31 pp | 38.80 % | +9.26 pp |
| IPC Active | 2.80 inst/cycle | 1.71 inst/cycle | -1.09 | 1.88 inst/cycle | -0.92 | 1.89 inst/cycle | -0.91 | 2.43 inst/cycle | -0.37 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding | 1.009× | 1.006× | ✓ | **✓ best** |
| B | add_vectorized_load_hints_to_reduce_l1tex_transaction_count | 1.008× | 1.000× | ✓ |  |
| C | fix_launch_grid_and_tune_num_warps_at_safe_xblock128 | 0.998× | 1.002× | ✓ |  |
| D | fix_grid_lambda_and_trim_autotune_to_low_register_pressure_configs | 1.002× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | add_vectorized_lo... | Δ | fix_launch_grid_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19140 ns | -120 ns (-0.6%) | 19260 ns | +0 ns (+0.0%) | 19230 ns | -30 ns (-0.2%) | 19100 ns | -160 ns (-0.8%) |
| SM Throughput (%) | 61.19 % | 40.90 % | -20.29 pp | 40.54 % | -20.65 pp | 36.83 % | -24.36 pp | 23.67 % | -37.52 pp |
| Memory Throughput (%) | 23.21 % | 23.13 % | -0.08 pp | 22.92 % | -0.29 pp | 22.74 % | -0.47 pp | 21.23 % | -1.98 pp |
| DRAM Throughput (%) | 14.65 % | 14.73 % | +0.08 pp | 14.66 % | +0.01 pp | 14.66 % | +0.01 pp | 14.77 % | +0.12 pp |
| Achieved Occupancy (%) | 60.80 % | 43.97 % | -16.83 pp | 43.27 % | -17.53 pp | 42.21 % | -18.59 pp | 27.82 % | -32.98 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 22.00 register/thread | +4 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 | 6.39 | -6.39 |
| No Eligible (%) | 29.54 % | 52.83 % | +23.29 pp | 52.63 % | +23.09 pp | 56.81 % | +27.27 pp | 70.65 % | +41.11 pp |
| IPC Active | 2.80 inst/cycle | 1.87 inst/cycle | -0.93 | 1.88 inst/cycle | -0.92 | 1.70 inst/cycle | -1.10 | 1.15 inst/cycle | -1.65 |

---

## Cumulative Progress

```
Baseline:  19260 ns  (1.000×)
Iter 5     19140 ns  (1.006×)  — increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding

Final best: 19140 ns  (1.0063×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -13,7 +14,7 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 96)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
@@ -21,21 +22,14 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    inv_std = tl.math.rsqrt(tmp5)
+    x_norm = tmp2 * inv_std
+    tmp15 = tl.math.fma(x_norm, tmp12, tmp14)
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
