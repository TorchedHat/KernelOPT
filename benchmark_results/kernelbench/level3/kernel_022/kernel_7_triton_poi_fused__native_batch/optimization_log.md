# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 60320 ns — **Best latency**: 60320 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (94080,1,1), Waves/SM: 44.55
- **Registers/thread**: 17
- **Duration**: 60320 ns
- **SM Throughput**: 57.61 % | **Memory**: 24.87 % | **DRAM**: 23.54 %
- **Achieved Occupancy**: 57.69 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 40.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.387× | 0.375× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.995× | 0.998× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.694× | 0.195× | ✓ |  |
| D | autotune_xblock_tile_size_to_hide_l1tex_latency | 0.995× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60320 ns | 160990 ns | +100670 ns (+166.9%) | 60450 ns | +130 ns (+0.2%) | 308960 ns | +248640 ns (+412.2%) | 60290 ns | -30 ns (-0.0%) |
| SM Throughput (%) | 57.61 % | 85.87 % | +28.26 pp | 68.27 % | +10.66 pp | 88.95 % | +31.34 pp | 66.96 % | +9.35 pp |
| Memory Throughput (%) | 24.87 % | 20.77 % | -4.10 pp | 25.18 % | +0.31 pp | 18.27 % | -6.60 pp | 24.70 % | -0.17 pp |
| DRAM Throughput (%) | 23.54 % | 8.84 % | -14.70 pp | 23.53 % | -0.01 pp | 4.61 % | -18.93 pp | 23.61 % | +0.07 pp |
| Achieved Occupancy (%) | 57.69 % | 56.35 % | -1.34 pp | 61.63 % | +3.94 pp | 85.92 % | +28.23 pp | 61.62 % | +3.93 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 44.55 | 71.27 | +26.72 | 44.55 | +0.00 | 178.18 | +133.63 | 44.55 | +0.00 |
| No Eligible (%) | 40.14 % | 13.08 % | -27.06 pp | 28.89 % | -11.25 pp | 10.46 % | -29.68 pp | 29.64 % | -10.50 pp |
| IPC Active | 2.40 inst/cycle | 3.48 inst/cycle | +1.08 | 2.83 inst/cycle | +0.43 | 3.58 inst/cycle | +1.18 | 2.82 inst/cycle | +0.42 |


### Key Learnings

#### ⚠️ Plan A: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.375×)

Adding @triton.autotune with large XBLOCK configs (1024–4096) to a kernel that originally used 17 registers/thread caused the compiler to allocate 48 registers/thread in the autotuned version. This tripled register pressure, cutting theoretical occupancy from 100% to 62.5% (Block_Limit_Registers dropped from 21 to 10 blocks/SM), and caused a 4x increase in executed instructions (27M → 109M), resulting in 2.67x longer kernel duration (60us → 161us) despite the scheduler stall rate improving (No Eligible: 40% → 13%). The net effect was a 0.37x regression: the compute throughput improvement (57% → 86% SM utilization) was completely overwhelmed by the instruction count explosion from unrolling large tiles.

> **Applicable when**: Kernel has very low register count (≤20 regs/thread) AND large XBLOCK configs are expected to unroll without register pressure growth AND the kernel body is truly minimal (few temporaries).

> **Avoid when**: The kernel body contains multiple intermediate temporaries (e.g., BN normalization with 10+ temp variables) that the compiler will unroll per XBLOCK element, causing register count to scale with tile size. When registers/thread jumps from <20 to >40, Block_Limit_Registers will halve theoretical occupancy and the instruction count explosion will dominate any latency-hiding benefit. Also avoid when the grid is already fixed and autotune changes the grid lambda incorrectly (here the grid was left as cdiv(xnumel, 128) instead of being updated to use meta['XBLOCK']).

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_latency_hiding` (0.195×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to a kernel that had a fixed, non-autotuned grid caused a critical bug: the launch grid remained hardcoded to cdiv(xnumel, 128) while autotune selected XBLOCK=512, causing each element to be processed ~4x redundantly. This inflated executed instructions from 27M to 217M (8x) and duration from 60 to 309 us (5x regression). When adding autotune, the grid lambda MUST use META['XBLOCK'] — a static grid is incompatible with a variable XBLOCK. The compute throughput improved (57% → 89%) but was irrelevant because the total work exploded.

> **Applicable when**: Kernel has CPIStall bottleneck from L1TEX latency, low eligible warps per scheduler (<2), and small fixed block size. Autotune is being added to explore larger tile sizes.

> **Avoid when**: The grid is hardcoded to a fixed XBLOCK value — if autotune changes XBLOCK but the grid stays fixed, elements will be processed multiple times causing instruction explosion and severe regression. Never mix static grid with variable XBLOCK autotune.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_occupancy | 0.995× | 1.000× | ✓ |  |
| B | replace_non_fused_fp32_division_with_rsqrt_for_fma_fusion | 1.002× | 1.001× | ✓ | **✓ best** |
| C | revert_autotune_register_explosion_restore_xblock128 | 0.999× | 1.000× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_element_processing | 0.696× | 0.195× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | replace_non_fused... | Δ | revert_autotune_r... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60320 ns | 60320 ns | +0 ns (+0.0%) | 60260 ns | -60 ns (-0.1%) | 60350 ns | +30 ns (+0.0%) | 308930 ns | +248610 ns (+412.2%) |
| SM Throughput (%) | 57.61 % | 67.99 % | +10.38 pp | 41.15 % | -16.46 pp | 68.45 % | +10.84 pp | 88.83 % | +31.22 pp |
| Memory Throughput (%) | 24.87 % | 25.08 % | +0.21 pp | 24.66 % | -0.21 pp | 25.24 % | +0.37 pp | 18.23 % | -6.64 pp |
| DRAM Throughput (%) | 23.54 % | 23.56 % | +0.02 pp | 23.62 % | +0.08 pp | 23.56 % | +0.02 pp | 4.60 % | -18.94 pp |
| Achieved Occupancy (%) | 57.69 % | 62.03 % | +4.34 pp | 42.88 % | -14.81 pp | 61.46 % | +3.77 pp | 85.92 % | +28.23 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 178.18 | +133.63 |
| No Eligible (%) | 40.14 % | 28.94 % | -11.20 pp | 57.23 % | +17.09 pp | 29.47 % | -10.67 pp | 10.50 % | -29.64 pp |
| IPC Active | 2.40 inst/cycle | 2.83 inst/cycle | +0.43 | 1.71 inst/cycle | -0.69 | 2.82 inst/cycle | +0.42 | 3.57 inst/cycle | +1.17 |

---

## Iteration 3

**Best so far**: 1.001× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_per_channel_scale_bias_to_collapse_bn_into_single_fma | 0.999× | 1.004× | ✓ |  |
| B | replace_integer_division_chain_with_rsqrt_to_enable_fma_fusion | 0.997× | 1.004× | ✓ |  |
| C | replace_nonfused_division_with_rsqrt_and_eliminate_redundant_multiply | 1.000× | 1.004× | ✓ |  |
| D | replace_integer_division_with_rsqrt_for_fma_fusion | 1.004× | 1.003× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.999× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_per_ch... | Δ | replace_integer_d... | Δ | replace_nonfused_... | Δ | replace_integer_d... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60320 ns | 60100 ns | -220 ns (-0.4%) | 60100 ns | -220 ns (-0.4%) | 60060 ns | -260 ns (-0.4%) | 60160 ns | -160 ns (-0.3%) | 60100 ns | -220 ns (-0.4%) |
| SM Throughput (%) | 57.61 % | 41.11 % | -16.50 pp | 41.16 % | -16.45 pp | 41.16 % | -16.45 pp | 41.05 % | -16.56 pp | 41.15 % | -16.46 pp |
| Memory Throughput (%) | 24.87 % | 24.58 % | -0.29 pp | 24.67 % | -0.20 pp | 24.67 % | -0.20 pp | 24.62 % | -0.25 pp | 24.69 % | -0.18 pp |
| DRAM Throughput (%) | 23.54 % | 23.60 % | +0.06 pp | 23.62 % | +0.08 pp | 23.66 % | +0.12 pp | 23.57 % | +0.03 pp | 23.64 % | +0.10 pp |
| Achieved Occupancy (%) | 57.69 % | 43.29 % | -14.40 pp | 42.99 % | -14.70 pp | 43.54 % | -14.15 pp | 42.99 % | -14.70 pp | 43.61 % | -14.08 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 |
| No Eligible (%) | 40.14 % | 57.23 % | +17.09 pp | 57.20 % | +17.06 pp | 57.20 % | +17.06 pp | 57.57 % | +17.43 pp | 57.30 % | +17.16 pp |
| IPC Active | 2.40 inst/cycle | 1.71 inst/cycle | -0.69 | 1.71 inst/cycle | -0.69 | 1.72 inst/cycle | -0.68 | 1.71 inst/cycle | -0.69 | 1.70 inst/cycle | -0.70 |

---

## Iteration 4

**Best so far**: 1.0027× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 1.005× | 0.996× | ✓ | **✓ best** |
| B | fix_grid_lambda_and_sweep_num_warps_for_occupancy | 1.003× | 1.002× | ✓ |  |
| C | fix_broken_autotune_grid_lambda_and_add_num_stages_for_l1tex_latency_hiding | 1.002× | 0.999× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_latency_without_changing_xblock | 1.001× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_grid_lambda_a... | Δ | fix_broken_autotu... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60320 ns | 60580 ns | +260 ns (+0.4%) | 60190 ns | -130 ns (-0.2%) | 60380 ns | +60 ns (+0.1%) | 60100 ns | -220 ns (-0.4%) |
| SM Throughput (%) | 57.61 % | 41.14 % | -16.47 pp | 41.08 % | -16.53 pp | 41.15 % | -16.46 pp | 41.36 % | -16.25 pp |
| Memory Throughput (%) | 24.87 % | 24.68 % | -0.19 pp | 24.55 % | -0.32 pp | 24.66 % | -0.21 pp | 24.81 % | -0.06 pp |
| DRAM Throughput (%) | 23.54 % | 23.46 % | -0.08 pp | 23.57 % | +0.03 pp | 23.55 % | +0.01 pp | 23.62 % | +0.08 pp |
| Achieved Occupancy (%) | 57.69 % | 42.94 % | -14.75 pp | 44.01 % | -13.68 pp | 42.97 % | -14.72 pp | 43.19 % | -14.50 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 |
| No Eligible (%) | 40.14 % | 56.90 % | +16.76 pp | 57.19 % | +17.05 pp | 57.14 % | +17.00 pp | 57.09 % | +16.95 pp |
| IPC Active | 2.40 inst/cycle | 1.71 inst/cycle | -0.69 | 1.70 inst/cycle | -0.70 | 1.72 inst/cycle | -0.68 | 1.71 inst/cycle | -0.69 |

---

## Iteration 5

**Best so far**: 1.0027× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_bn_chain_into_explicit_fma_to_reduce_nonfused_fp32_instructions | 1.005× | 1.000× | ✓ | **✓ best** |
| B | add_evict_first_to_streaming_main_tensor_load_to_preserve_l1_for_channel_params | 1.003× | 1.003× | ✓ |  |
| C | preload_per_channel_bn_params_into_shared_memory_to_eliminate_l1tex_stalls | 0.994× | 1.000× | ✓ |  |
| D | fuse_fp32_multiply_add_into_fma_and_add_evict_first_on_streaming_load | 1.000× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_bn_ch... | Δ | add_evict_first_t... | Δ | preload_per_chann... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60320 ns | 60320 ns | +0 ns (+0.0%) | 60130 ns | -190 ns (-0.3%) | 60320 ns | +0 ns (+0.0%) | 60060 ns | -260 ns (-0.4%) |
| SM Throughput (%) | 57.61 % | 40.55 % | -17.06 pp | 45.01 % | -12.60 pp | 94.93 % | +37.32 pp | 45.08 % | -12.53 pp |
| Memory Throughput (%) | 24.87 % | 24.97 % | +0.10 pp | 24.54 % | -0.33 pp | 61.81 % | +36.94 pp | 24.73 % | -0.14 pp |
| DRAM Throughput (%) | 23.54 % | 23.70 % | +0.16 pp | 23.83 % | +0.29 pp | 23.73 % | +0.19 pp | 23.61 % | +0.07 pp |
| Achieved Occupancy (%) | 57.69 % | 43.81 % | -13.88 pp | 44.32 % | -13.37 pp | 55.33 % | -2.36 pp | 43.57 % | -14.12 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 |
| No Eligible (%) | 40.14 % | 57.98 % | +17.84 pp | 53.13 % | +12.99 pp | 48.01 % | +7.87 pp | 53.00 % | +12.86 pp |
| IPC Active | 2.40 inst/cycle | 1.68 inst/cycle | -0.72 | 1.88 inst/cycle | -0.52 | 2.07 inst/cycle | -0.33 | 1.87 inst/cycle | -0.53 |

---

## Cumulative Progress

```
Baseline:  60320 ns  (1.000×)
Iter 2     60260 ns  (1.001×)  — replace_non_fused_fp32_division_with_rsqrt_for_fma_fusion
Iter 3     60160 ns  (1.003×)  — replace_integer_division_with_rsqrt_for_fma_fusion

Final best: 60320 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 12042240
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 96)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
+    tmp6 = tl.math.rsqrt(tmp5)
+    tmp_scale = tmp12 * tmp6
+    tmp15 = tl.math.fma(tmp2, tmp_scale, tmp14)
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
