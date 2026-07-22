# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 126300 ns — **Best latency**: 125470 ns (**1.0066×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (200704,1,1), Waves/SM: 95.03
- **Registers/thread**: 19
- **Duration**: 126300 ns
- **SM Throughput**: 63.03 % | **Memory**: 29.02 % | **DRAM**: 28.82 %
- **Achieved Occupancy**: 60.88 %
- **IPC Active**: 2.57 inst/cycle | **No Eligible**: 35.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.629× | 0.617× | ✓ |  |
| B | autotune_xblock_num_warps_to_improve_occupancy_and_hide_l1tex_latency | 0.999× | 1.000× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.877× | 0.863× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.605× | 0.602× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | add_autotune_with... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 204640 ns | +78340 ns (+62.0%) | 126270 ns | -30 ns (-0.0%) | 146430 ns | +20130 ns (+15.9%) | 209790 ns | +83490 ns (+66.1%) |
| SM Throughput (%) | 63.03 % | 87.01 % | +23.98 pp | 76.86 % | +13.83 pp | 86.35 % | +23.32 pp | 83.63 % | +20.60 pp |
| Memory Throughput (%) | 29.02 % | 27.25 % | -1.77 pp | 29.16 % | +0.14 pp | 34.03 % | +5.01 pp | 30.99 % | +1.97 pp |
| DRAM Throughput (%) | 28.82 % | 17.77 % | -11.05 pp | 28.81 % | -0.01 pp | 24.83 % | -3.99 pp | 17.33 % | -11.49 pp |
| Achieved Occupancy (%) | 60.88 % | 87.78 % | +26.90 pp | 66.00 % | +5.12 pp | 79.35 % | +18.47 pp | 81.17 % | +20.29 pp |
| Registers/Thread | 19.00 register/thread | 32.00 register/thread | +13 | 20.00 register/thread | +1 | 28.00 register/thread | +9 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 190.06 | +95.03 |
| No Eligible (%) | 35.49 % | 12.32 % | -23.17 pp | 22.18 % | -13.31 pp | 12.20 % | -23.29 pp | 15.54 % | -19.95 pp |
| IPC Active | 2.57 inst/cycle | 3.51 inst/cycle | +0.94 | 3.11 inst/cycle | +0.54 | 3.52 inst/cycle | +0.95 | 3.38 inst/cycle | +0.81 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.617×)

Adding @triton.autotune with large XBLOCK configs (512–4096) caused registers per thread to jump from 19 to 32, increasing total executed instructions by 2.2x (62.6M → 139.1M) and kernel duration by 62% (126 µs → 205 µs). Although achieved occupancy improved (60.9% → 87.8%) and compute SM throughput rose (63% → 87%), the massive instruction count increase dominated, resulting in a 0.62x slowdown. The root cause is that larger XBLOCK tiles force the compiler to keep more live values in registers simultaneously, inflating register pressure and instruction count beyond what the occupancy gain can compensate for in this elementwise-fused kernel.

> **Applicable when**: Kernel has low achieved occupancy (<65%) and high L1TEX stall share (>40%) with small XBLOCK (≤128) and low registers per thread (≤20), suggesting room to increase tile size without register pressure issues.

> **Avoid when**: Kernel already uses near-maximum registers per thread (>20) or the XBLOCK increase would push registers per thread above ~24, as the resulting instruction count explosion (2x+) will dominate any occupancy or cache-reuse gains. Also avoid when the kernel fuses many operations (BN+conv+ReLU) that multiply live values per thread. Additionally, do not introduce xmask bounds checks when the original kernel used tl.full([XBLOCK], True) — the added predication overhead compounds the regression.

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.863×)

Adding @triton.autotune with larger XBLOCK values (256–2048) for a pointwise kernel that was originally XBLOCK=128 caused a 16% regression. Although occupancy improved (60.9%→79.4%) and eligible warps per scheduler increased (1.33→5.04), the register count jumped from 19 to 28 per thread, and total executed instructions increased by ~56% (62.6M→97.9M), indicating the autotune selected a config that generated significantly more instructions without a proportional reduction in stall cycles (Warp Cycles Per Issued Instruction barely changed: 14.87→14.34). The CPIStall bottleneck was not resolved because the stall was not primarily due to insufficient vectorization width but rather an inherent memory latency pattern; the extra instruction overhead from larger tiles outweighed any vectorization benefit.

> **Applicable when**: Never — this specific strategy (autotune larger XBLOCK for pointwise kernels already at XBLOCK=128 with CPIStall) caused regression. Avoid when registers per thread are already near 20 and the kernel is compute-bound (SM throughput >60%), as larger tiles increase register pressure and instruction count without resolving the underlying memory latency.

> **Avoid when**: Kernel is already compute-bound (SM throughput >60%); registers per thread are near 20 (headroom is limited); CPIStall is present but warp cycles per issued instruction is already low (~15); the kernel is a fused pointwise op where larger tiles generate proportionally more instructions without reducing stall cycles. Also avoid when the grid is already large (waves_per_sm ~95) — the parallelism is sufficient and reducing grid size does not help.

#### ❌ Plan D: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.602×)

Doubling block size from 128 to 256 (the autotune winner) doubled waves_per_sm from 95 to 190 and more than doubled total executed instructions (62M → 138M), causing duration to increase from 126 µs to 210 µs despite occupancy improving from 61% to 81%. The regression occurred because the kernel was already compute-bound (63% SM throughput before), and the larger block size introduced unnecessary masking overhead (xmask = xindex < xnumel replacing tl.full([XBLOCK], True)) and doubled instruction count without reducing memory latency. When a pointwise kernel already has high waves_per_sm (≥90) and is compute-bottlenecked rather than latency-bottlenecked, increasing XBLOCK inflates instruction count and grid serialization without benefit.

> **Applicable when**: Never: this strategy caused regression. Avoid increasing XBLOCK via autotune when waves_per_sm is already high (≥90), the kernel is compute-bound (SM throughput ≥60%), and the grid is already large enough to saturate all SMs.

> **Avoid when**: Kernel already has high waves_per_sm (≥90); kernel is compute-bound rather than memory-latency-bound; xnumel is exactly divisible by current XBLOCK (making masking unnecessary overhead); adding xmask introduces extra instructions that inflate total instruction count.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_integer_reciprocal_sqrt_with_rsqrt_and_add_evict_first_on_streaming_load | 1.005× | 1.004× | ✓ | **✓ best** |
| B | reduce_non_fused_fp32_instructions_via_rsqrt_and_fma | 1.003× | 1.003× | ✓ |  |
| C | force_fma_instruction_fusion_in_bn_relu_compute | 0.629× | 0.617× | ✓ |  |
| D | reduce_instruction_count_via_rsqrt_and_fma_simplification | 1.004× | 1.003× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_integer_r... | Δ | reduce_non_fused_... | Δ | force_fma_instruc... | Δ | reduce_instructio... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 125860 ns | -440 ns (-0.3%) | 125950 ns | -350 ns (-0.3%) | 204860 ns | +78560 ns (+62.2%) | 125920 ns | -380 ns (-0.3%) | 125760 ns | -540 ns (-0.4%) |
| SM Throughput (%) | 63.03 % | 53.27 % | -9.76 pp | 57.32 % | -5.71 pp | 87.22 % | +24.19 pp | 45.40 % | -17.63 pp | 76.42 % | +13.39 pp |
| Memory Throughput (%) | 29.02 % | 28.90 % | -0.12 pp | 39.76 % | +10.74 pp | 27.32 % | -1.70 pp | 28.89 % | -0.13 pp | 28.99 % | -0.03 pp |
| DRAM Throughput (%) | 28.82 % | 28.90 % | +0.08 pp | 28.89 % | +0.07 pp | 17.76 % | -11.06 pp | 28.89 % | +0.07 pp | 28.95 % | +0.13 pp |
| Achieved Occupancy (%) | 60.88 % | 47.30 % | -13.58 pp | 37.03 % | -23.85 pp | 87.45 % | +26.57 pp | 43.37 % | -17.51 pp | 66.05 % | +5.17 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 26.00 register/thread | +7 | 32.00 register/thread | +13 | 18.00 register/thread | -1 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.49 % | 45.35 % | +9.86 pp | 41.18 % | +5.69 pp | 11.87 % | -23.62 pp | 62.98 % | +27.49 pp | 21.69 % | -13.80 pp |
| IPC Active | 2.57 inst/cycle | 2.18 inst/cycle | -0.39 | 2.37 inst/cycle | -0.20 | 3.52 inst/cycle | +0.95 | 1.48 inst/cycle | -1.09 | 3.11 inst/cycle | +0.54 |

---

## Iteration 3

**Best so far**: 1.0035× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | arithmetic_simplification_rsqrt_fma_isolated_no_xblock_change | 1.004× | 1.004× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.005× | 1.007× | ✓ | **✓ best** |
| C | fuse_fp32_multiply_add_to_fma_at_fixed_xblock | 1.002× | 1.002× | ✓ |  |
| D | restore_small_xblock_config_to_escape_register_pressure_regression | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | arithmetic_simpli... | Δ | increase_num_warp... | Δ | fuse_fp32_multipl... | Δ | restore_small_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 125820 ns | -480 ns (-0.4%) | 125470 ns | -830 ns (-0.7%) | 126110 ns | -190 ns (-0.2%) | 126460 ns | +160 ns (+0.1%) |
| SM Throughput (%) | 63.03 % | 49.32 % | -13.71 pp | 45.53 % | -17.50 pp | 57.81 % | -5.22 pp | 76.88 % | +13.85 pp |
| Memory Throughput (%) | 29.02 % | 28.93 % | -0.09 pp | 29.02 % | +0.00 pp | 40.08 % | +11.06 pp | 29.19 % | +0.17 pp |
| DRAM Throughput (%) | 28.82 % | 28.93 % | +0.11 pp | 29.02 % | +0.20 pp | 28.84 % | +0.02 pp | 28.76 % | -0.06 pp |
| Achieved Occupancy (%) | 60.88 % | 45.83 % | -15.05 pp | 43.50 % | -17.38 pp | 37.12 % | -23.76 pp | 66.27 % | +5.39 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 18.00 register/thread | -1 | 26.00 register/thread | +7 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.49 % | 49.51 % | +14.02 pp | 62.76 % | +27.27 pp | 41.39 % | +5.90 pp | 22.15 % | -13.34 pp |
| IPC Active | 2.57 inst/cycle | 2.01 inst/cycle | -0.56 | 1.48 inst/cycle | -1.09 | 2.36 inst/cycle | -0.21 | 3.12 inst/cycle | +0.55 |

---

## Iteration 4

**Best so far**: 1.0066× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_first_on_large_streaming_tensor_to_reduce_l1tex_cache_pollution | 1.004× | 1.002× | ✓ |  |
| B | evict_first_on_streaming_load_and_fix_autotune_grid | 1.004× | 1.004× | ✓ |  |
| C | fix_dead_autotune_and_increase_num_warps_at_fixed_xblock_128 | 1.004× | 1.003× | ✓ |  |
| D | replace_sqrt_fdiv_pair_with_single_rsqrt_at_fixed_xblock | 1.005× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_first_o... | Δ | evict_first_on_st... | Δ | fix_dead_autotune... | Δ | replace_sqrt_fdiv... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 126110 ns | -190 ns (-0.2%) | 125820 ns | -480 ns (-0.4%) | 125950 ns | -350 ns (-0.3%) | 125790 ns | -510 ns (-0.4%) |
| SM Throughput (%) | 63.03 % | 45.21 % | -17.82 pp | 52.27 % | -10.76 pp | 49.17 % | -13.86 pp | 49.26 % | -13.77 pp |
| Memory Throughput (%) | 29.02 % | 28.84 % | -0.18 pp | 28.92 % | -0.10 pp | 28.91 % | -0.11 pp | 28.93 % | -0.09 pp |
| DRAM Throughput (%) | 28.82 % | 28.84 % | +0.02 pp | 28.92 % | +0.10 pp | 28.91 % | +0.09 pp | 28.93 % | +0.11 pp |
| Achieved Occupancy (%) | 60.88 % | 44.37 % | -16.51 pp | 46.91 % | -13.97 pp | 45.98 % | -14.90 pp | 45.83 % | -15.05 pp |
| Registers/Thread | 19.00 register/thread | 18.00 register/thread | -1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.49 % | 58.75 % | +23.26 pp | 46.42 % | +10.93 pp | 49.66 % | +14.17 pp | 49.53 % | +14.04 pp |
| IPC Active | 2.57 inst/cycle | 1.65 inst/cycle | -0.92 | 2.14 inst/cycle | -0.43 | 2.02 inst/cycle | -0.55 | 2.00 inst/cycle | -0.57 |

---

## Iteration 5

**Best so far**: 1.0066× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_software_pipelining_via_num_stages_with_fixed_grid_lambda | 1.003× | 1.007× | ✓ |  |
| B | cap_registers_per_thread_with_maxnreg_to_increase_sm_occupancy | 1.004× | 1.007× | ✓ |  |
| C | reduce_num_warps_to_increase_block_count_and_scheduler_eligible_warps | 1.004× | 1.007× | ✓ |  |
| D | increase_warps_per_block_via_xblock256_no_mask_to_hide_l1tex_latency | 1.004× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_software_p... | Δ | cap_registers_per... | Δ | reduce_num_warps_... | Δ | increase_warps_pe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 125440 ns | -860 ns (-0.7%) | 125440 ns | -860 ns (-0.7%) | 125440 ns | -860 ns (-0.7%) | 126300 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 63.03 % | 49.60 % | -13.43 pp | 53.24 % | -9.79 pp | 49.35 % | -13.68 pp | 45.42 % | -17.61 pp |
| Memory Throughput (%) | 29.02 % | 28.99 % | -0.03 pp | 28.99 % | -0.03 pp | 28.99 % | -0.03 pp | 28.80 % | -0.22 pp |
| DRAM Throughput (%) | 28.82 % | 28.99 % | +0.17 pp | 28.99 % | +0.17 pp | 28.99 % | +0.17 pp | 28.80 % | -0.02 pp |
| Achieved Occupancy (%) | 60.88 % | 45.77 % | -15.11 pp | 46.41 % | -14.47 pp | 45.83 % | -15.05 pp | 43.94 % | -16.94 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 18.00 register/thread | -1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.49 % | 49.87 % | +14.38 pp | 45.41 % | +9.92 pp | 49.82 % | +14.33 pp | 58.95 % | +23.46 pp |
| IPC Active | 2.57 inst/cycle | 2.00 inst/cycle | -0.57 | 2.19 inst/cycle | -0.38 | 2.00 inst/cycle | -0.57 | 1.65 inst/cycle | -0.92 |

---

## Cumulative Progress

```
Baseline:  126300 ns  (1.000×)
Iter 2     125860 ns  (1.004×)  — replace_integer_reciprocal_sqrt_with_rsqrt_and_add_evict_first_on_streaming_load
Iter 3     125470 ns  (1.007×)  — increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency

Final best: 125470 ns  (1.0066×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,40 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 25690112
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 1e-05, tl.float32)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.sqrt_rn(tmp7)
-    tmp9 = tl.full([1], 1, tl.int32)
-    tmp10 = (tmp9 / tmp8)
-    tmp11 = tl.full([1], 1.0, tl.float32)
-    tmp12 = tmp10 * tmp11
-    tmp13 = tmp4 * tmp12
-    tmp15 = tmp13 * tmp14
-    tmp17 = tmp15 + tmp16
-    tmp18 = tl.full([1], 0, tl.int32)
-    tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tmp_inv = tl.math.rsqrt(tmp7)
+    tmp_norm = tmp4 * tmp_inv
+    tmp17 = tl.math.fma(tmp_norm, tmp14, tmp16)
+    tmp19 = tl.maximum(tl.zeros([1], tl.float32), tmp17)
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +49,7 @@

     arg_5 = inputs[5]
     xnumel = 25690112
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 25690112, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 25690112)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
