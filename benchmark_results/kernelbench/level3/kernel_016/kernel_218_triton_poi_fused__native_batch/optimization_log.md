# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6750 ns — **Best latency**: 6780 ns (**0.9956×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_218_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6493,1,1), Waves/SM: 3.07
- **Registers/thread**: 22
- **Duration**: 6750 ns
- **SM Throughput**: 42.24 % | **Memory**: 16.19 % | **DRAM**: 10.16 %
- **Achieved Occupancy**: 64.36 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 35.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.935× | 0.790× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.912× | 0.802× | ✓ |  |
| C | add_autotune_with_vectorized_loads_via_larger_xblock | 0.998× | 0.284× | ✓ |  |
| D | increase_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.914× | 0.802× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | autotune_xblock_e... | Δ | add_autotune_with... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 8540 ns | +1790 ns (+26.5%) | 8420 ns | +1670 ns (+24.7%) | 23740 ns | +16990 ns (+251.7%) | 8420 ns | +1670 ns (+24.7%) |
| SM Throughput (%) | 42.24 % | 63.32 % | +21.08 pp | 63.24 % | +21.00 pp | 78.21 % | +35.97 pp | 63.37 % | +21.13 pp |
| Memory Throughput (%) | 16.19 % | 17.26 % | +1.07 pp | 17.22 % | +1.03 pp | 16.06 % | -0.13 pp | 17.28 % | +1.09 pp |
| DRAM Throughput (%) | 10.16 % | 8.03 % | -2.13 pp | 8.16 % | -2.00 pp | 2.88 % | -7.28 pp | 8.15 % | -2.01 pp |
| Achieved Occupancy (%) | 64.36 % | 79.68 % | +15.32 pp | 79.79 % | +15.43 pp | 54.82 % | -9.54 pp | 79.15 % | +14.79 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 48.00 register/thread | +26 | 32.00 register/thread | +10 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 | 9.84 | +6.77 | 3.07 | +0.00 |
| No Eligible (%) | 35.13 % | 16.83 % | -18.30 pp | 16.89 % | -18.24 pp | 14.11 % | -21.02 pp | 17.66 % | -17.47 pp |
| IPC Active | 2.48 inst/cycle | 3.30 inst/cycle | +0.82 | 3.30 inst/cycle | +0.82 | 3.43 inst/cycle | +0.95 | 3.29 inst/cycle | +0.81 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.802×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that was already achieving reasonable occupancy caused a 25% slowdown. The autotune selected a config that nearly doubled register usage per thread (22→32), which increased total instruction count by ~89% (2.2M→4.2M instructions) without a proportional reduction in kernel launches, resulting in longer elapsed cycles (10010→12464) despite higher compute throughput (42%→63%). The partial-wave tail overhead flagged by LaunchConfiguration was real, but the autotuner's selected config introduced more overhead via register pressure and instruction bloat than it saved by reducing grid size.

> **Applicable when**: Never — this specific approach is an anti-pattern. The LaunchConfiguration rule flags partial-wave overhead, but autotuning XBLOCK upward on a pointwise kernel with low register count does not reliably eliminate the overhead and can increase register pressure and instruction count.

> **Avoid when**: Do not apply @triton.autotune with large XBLOCK configs (512–2048) to short-duration pointwise kernels (~6-8 µs) where: (1) the grid is already fixed and not updated to use meta['XBLOCK'], causing the grid to remain at the original size while the kernel processes larger tiles redundantly; (2) register count is already low (≤22) and larger tiles cause register spill or bloat; (3) the kernel body has no shared memory or software pipelining benefit from larger tiles.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_bn_arithmetic_with_fma_and_fix_broken_autotune | 0.985× | 1.004× | ✓ |  |
| B | fix_autotune_grid_and_fuse_rsqrt_instruction | 0.982× | 1.000× | ✓ |  |
| C | replace_sqrt_divide_with_rsqrt_to_enable_fma_fusion_and_reduce_register_pressure | 1.009× | 1.009× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_rsqrt_simplification_and_restore_small_xblock | 0.985× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_bn_arithm... | Δ | fix_autotune_grid... | Δ | replace_sqrt_divi... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 6720 ns | -30 ns (-0.4%) | 6750 ns | +0 ns (+0.0%) | 6690 ns | -60 ns (-0.9%) | 6820 ns | +70 ns (+1.0%) |
| SM Throughput (%) | 42.24 % | 31.97 % | -10.27 pp | 31.87 % | -10.37 pp | 31.97 % | -10.27 pp | 31.76 % | -10.48 pp |
| Memory Throughput (%) | 16.19 % | 16.22 % | +0.03 pp | 16.16 % | -0.03 pp | 16.24 % | +0.05 pp | 16.14 % | -0.05 pp |
| DRAM Throughput (%) | 10.16 % | 10.18 % | +0.02 pp | 10.19 % | +0.03 pp | 10.28 % | +0.12 pp | 10.08 % | -0.08 pp |
| Achieved Occupancy (%) | 64.36 % | 48.13 % | -16.23 pp | 48.27 % | -16.09 pp | 47.50 % | -16.86 pp | 49.98 % | -14.38 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 | 3.07 | +0.00 | 3.07 | +0.00 |
| No Eligible (%) | 35.13 % | 59.81 % | +24.68 pp | 59.39 % | +24.26 pp | 61.33 % | +26.20 pp | 61.66 % | +26.53 pp |
| IPC Active | 2.48 inst/cycle | 1.56 inst/cycle | -0.92 | 1.60 inst/cycle | -0.88 | 1.56 inst/cycle | -0.92 | 1.60 inst/cycle | -0.88 |

---

## Iteration 3

**Best so far**: 1.009× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.985× | 1.009× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency | 1.009× | 1.004× | ✓ |  |
| C | increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding | 0.981× | 0.879× | ✓ |  |
| D | increase_num_warps_with_maxnreg_cap_to_hide_l1tex_latency | 1.011× | 0.996× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 6690 ns | -60 ns (-0.9%) | 6720 ns | -30 ns (-0.4%) | 7680 ns | +930 ns (+13.8%) | 6780 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 42.24 % | 31.30 % | -10.94 pp | 31.87 % | -10.37 pp | 54.94 % | +12.70 pp | 32.00 % | -10.24 pp |
| Memory Throughput (%) | 16.19 % | 15.75 % | -0.44 pp | 16.15 % | -0.04 pp | 25.46 % | +9.27 pp | 16.16 % | -0.03 pp |
| DRAM Throughput (%) | 10.16 % | 10.28 % | +0.12 pp | 10.25 % | +0.09 pp | 8.90 % | -1.26 pp | 10.16 % | +0.00 pp |
| Achieved Occupancy (%) | 64.36 % | 49.58 % | -14.78 pp | 48.12 % | -16.24 pp | 74.52 % | +10.16 pp | 46.48 % | -17.88 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 19.00 register/thread | -3 | 18.00 register/thread | -4 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 | 6.15 | +3.08 | 3.07 | +0.00 |
| No Eligible (%) | 35.13 % | 61.73 % | +26.60 pp | 59.17 % | +24.04 pp | 38.36 % | +3.23 pp | 60.37 % | +25.24 pp |
| IPC Active | 2.48 inst/cycle | 1.61 inst/cycle | -0.87 | 1.60 inst/cycle | -0.88 | 2.49 inst/cycle | +0.01 | 1.57 inst/cycle | -0.91 |

---

## Iteration 4

**Best so far**: 1.009× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_arithmetic_with_explicit_fma_to_reduce_instruction_count | 0.985× | 1.009× | ✓ |  |
| B | add_evict_first_store_policy_to_reduce_l1tex_cache_pollution | 0.983× | 1.004× | ✓ |  |
| C | streaming_eviction_policy_on_main_data_load_to_improve_bn_param_cache_hit_rate | 1.009× | 0.990× | ✓ |  |
| D | fuse_bn_arithmetic_chain_with_explicit_fma_to_reduce_non_fused_fp32_instructions | 1.008× | 0.879× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_arithme... | Δ | add_evict_first_s... | Δ | streaming_evictio... | Δ | fuse_bn_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 6690 ns | -60 ns (-0.9%) | 6720 ns | -30 ns (-0.4%) | 6820 ns | +70 ns (+1.0%) | 7680 ns | +930 ns (+13.8%) |
| SM Throughput (%) | 42.24 % | 31.74 % | -10.50 pp | 31.31 % | -10.93 pp | 31.66 % | -10.58 pp | 55.00 % | +12.76 pp |
| Memory Throughput (%) | 16.19 % | 16.04 % | -0.15 pp | 15.87 % | -0.32 pp | 15.90 % | -0.29 pp | 25.48 % | +9.29 pp |
| DRAM Throughput (%) | 10.16 % | 10.26 % | +0.10 pp | 10.19 % | +0.03 pp | 10.07 % | -0.09 pp | 8.93 % | -1.23 pp |
| Achieved Occupancy (%) | 64.36 % | 50.29 % | -14.07 pp | 51.25 % | -13.11 pp | 48.56 % | -15.80 pp | 74.45 % | +10.09 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 19.00 register/thread | -3 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 | 3.07 | +0.00 | 6.15 | +3.08 |
| No Eligible (%) | 35.13 % | 59.59 % | +24.46 pp | 59.45 % | +24.32 pp | 55.85 % | +20.72 pp | 37.14 % | +2.01 pp |
| IPC Active | 2.48 inst/cycle | 1.54 inst/cycle | -0.94 | 1.52 inst/cycle | -0.96 | 1.70 inst/cycle | -0.78 | 2.49 inst/cycle | +0.01 |

---

## Iteration 5

**Best so far**: 1.009× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_xblock_to_64_with_2_warps_for_l1tex_latency_hiding | 1.002× | 1.014× | ✓ |  |
| D | fix_broken_autotune_grid_and_pipeline_l1tex_loads_via_num_stages | 0.981× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 6660 ns | -90 ns (-1.3%) | 6910 ns | +160 ns (+2.4%) |
| SM Throughput (%) | 42.24 % | 31.60 % | -10.64 pp | 31.93 % | -10.31 pp |
| Memory Throughput (%) | 16.19 % | 16.02 % | -0.17 pp | 16.18 % | -0.01 pp |
| DRAM Throughput (%) | 10.16 % | 10.31 % | +0.15 pp | 9.95 % | -0.21 pp |
| Achieved Occupancy (%) | 64.36 % | 47.49 % | -16.87 pp | 50.12 % | -14.24 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 |
| No Eligible (%) | 35.13 % | 59.68 % | +24.55 pp | 59.22 % | +24.09 pp |
| IPC Active | 2.48 inst/cycle | 1.58 inst/cycle | -0.90 | 1.56 inst/cycle | -0.92 |

---

## Cumulative Progress

```
Baseline:  6750 ns  (1.000×)
Iter 2     6690 ns  (1.009×)  — replace_sqrt_divide_with_rsqrt_to_enable_fma_fusion_and_reduce_register_pressure

Final best: 6780 ns  (0.9956×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_218(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -21,19 +22,11 @@

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
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp_rsqrt = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp_rsqrt
+    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)  # fused: tmp11*tmp12 + tmp14
+    tmp17 = tl.maximum(tl.full([1], 0.0, tl.float32), tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
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
