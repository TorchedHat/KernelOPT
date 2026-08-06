# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9280 ns — **Best latency**: 9500 ns (**0.9768×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_18`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_18_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9800,1,1), Waves/SM: 4.64
- **Registers/thread**: 28
- **Duration**: 9280 ns
- **SM Throughput**: 59.35 % | **Memory**: 37.50 % | **DRAM**: 37.50 %
- **Achieved Occupancy**: 76.53 %
- **IPC Active**: 2.99 inst/cycle | **No Eligible**: 23.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave_and_improve_cache_reuse | 0.836× | 0.709× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.842× | 0.709× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_partial_waves | 0.835× | 0.696× | ✓ |  |
| D | increase_xblock_and_autotune_for_occupancy_and_wave_elimination | 0.845× | 0.702× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9280 ns | 13090 ns | +3810 ns (+41.1%) | 13090 ns | +3810 ns (+41.1%) | 13340 ns | +4060 ns (+43.8%) | 13220 ns | +3940 ns (+42.5%) |
| SM Throughput (%) | 59.35 % | 56.59 % | -2.76 pp | 56.75 % | -2.60 pp | 56.59 % | -2.76 pp | 55.75 % | -3.60 pp |
| Memory Throughput (%) | 37.50 % | 26.70 % | -10.80 pp | 26.78 % | -10.72 pp | 26.52 % | -10.98 pp | 26.40 % | -11.10 pp |
| DRAM Throughput (%) | 37.50 % | 26.65 % | -10.85 pp | 26.58 % | -10.92 pp | 26.08 % | -11.42 pp | 26.40 % | -11.10 pp |
| Achieved Occupancy (%) | 76.53 % | 63.37 % | -13.16 pp | 63.34 % | -13.19 pp | 63.29 % | -13.24 pp | 63.31 % | -13.22 pp |
| Registers/Thread | 28.00 register/thread | 34.00 register/thread | +6 | 34.00 register/thread | +6 | 34.00 register/thread | +6 | 34.00 register/thread | +6 |
| Waves/SM | 4.64 | 6.19 | +1.55 | 6.19 | +1.55 | 6.19 | +1.55 | 6.19 | +1.55 |
| No Eligible (%) | 23.48 % | 33.30 % | +9.82 pp | 33.51 % | +10.03 pp | 33.50 % | +10.02 pp | 34.50 % | +11.02 pp |
| IPC Active | 2.99 inst/cycle | 2.65 inst/cycle | -0.34 | 2.67 inst/cycle | -0.32 | 2.67 inst/cycle | -0.32 | 2.67 inst/cycle | -0.32 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_eliminate_partial_wave` (0.709×)

Adding @triton.autotune with larger XBLOCK values (512–2048) caused the compiler to allocate more registers per thread (28→34), which reduced Block Limit Registers from 16 to 12 and dropped theoretical occupancy from 100% to 75% and achieved occupancy from 76.5% to 63.3%. The increased register pressure more than offset any partial-wave tail elimination benefit, resulting in 41% more elapsed cycles (13688→19480) and 38% more executed instructions (4.2M→5.8M), producing a 0.71x regression. When a kernel is already near 100% theoretical occupancy with a small register footprint, forcing larger tiles via autotune can silently increase register usage and destroy occupancy.

> **Applicable when**: Kernel has a partial-wave tail AND register pressure is low (≤24 regs/thread) AND theoretical occupancy is already below 80% so register increase cannot hurt further.

> **Avoid when**: Kernel already achieves near-100% theoretical occupancy with its current register count (≤32 regs/thread at block_limit_registers=16); in this regime, larger XBLOCK tiles increase register allocation, reduce block_limit_registers, and drop theoretical occupancy, causing net regression. Also avoid when the call-site grid is not updated to use meta['XBLOCK'] — the autotune will select a larger XBLOCK but the grid remains sized for the original small XBLOCK, wasting work.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_fixed_xblock128_and_fix_eviction_policy_for_streaming_loads | 1.035× | 0.977× | ✓ | **✓ best** |
| B | restore_occupancy_by_removing_autotune_and_fixing_grid | 1.017× | 0.989× | ✓ |  |
| C | reduce_register_pressure_by_removing_large_xblock_autotune_to_restore_occupancy | 1.016× | 1.011× | ✓ |  |
| D | remove_large_xblock_autotune_to_restore_register_pressure_and_occupancy | 1.005× | 0.967× | ✓ |  |
| E | add_triton_autotune | 1.001× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_fixed_xbl... | Δ | restore_occupancy... | Δ | reduce_register_p... | Δ | remove_large_xblo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9280 ns | 9500 ns | +220 ns (+2.4%) | 9380 ns | +100 ns (+1.1%) | 9180 ns | -100 ns (-1.1%) | 9600 ns | +320 ns (+3.4%) | 9600 ns | +320 ns (+3.4%) |
| SM Throughput (%) | 59.35 % | 58.58 % | -0.77 pp | 59.33 % | -0.02 pp | 59.52 % | +0.17 pp | 59.01 % | -0.34 pp | 58.37 % | -0.98 pp |
| Memory Throughput (%) | 37.50 % | 36.70 % | -0.80 pp | 37.11 % | -0.39 pp | 37.95 % | +0.45 pp | 36.44 % | -1.06 pp | 36.38 % | -1.12 pp |
| DRAM Throughput (%) | 37.50 % | 36.70 % | -0.80 pp | 37.11 % | -0.39 pp | 37.95 % | +0.45 pp | 36.44 % | -1.06 pp | 36.38 % | -1.12 pp |
| Achieved Occupancy (%) | 76.53 % | 76.81 % | +0.28 pp | 76.64 % | +0.11 pp | 76.66 % | +0.13 pp | 77.43 % | +0.90 pp | 75.37 % | -1.16 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 4.64 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 |
| No Eligible (%) | 23.48 % | 24.00 % | +0.52 pp | 23.25 % | -0.23 pp | 24.41 % | +0.93 pp | 23.64 % | +0.16 pp | 25.10 % | +1.62 pp |
| IPC Active | 2.99 inst/cycle | 3.00 inst/cycle | +0.01 | 3.01 inst/cycle | +0.02 | 3.02 inst/cycle | +0.03 | 2.96 inst/cycle | -0.03 | 2.94 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_eviction_policy_to_evict_first_for_streaming_avg_pool_loads | 1.022× | 0.983× | ✓ |  |
| B | change_eviction_policy_to_evict_first_for_streaming_loads_and_enable_fma_fusion | 1.018× | 0.980× | ✓ |  |
| C | fuse_non_fused_fp32_multiply_adds_into_fma_instructions | 1.019× | 0.961× | ✓ |  |
| D | fix_eviction_policy_for_large_streaming_tensors_to_improve_l2_hit_rate_and_reduce_cache_pollution | 1.018× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | change_eviction_p... | Δ | fuse_non_fused_fp... | Δ | fix_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9280 ns | 9440 ns | +160 ns (+1.7%) | 9470 ns | +190 ns (+2.0%) | 9660 ns | +380 ns (+4.1%) | 9250 ns | -30 ns (-0.3%) |
| SM Throughput (%) | 59.35 % | 57.04 % | -2.31 pp | 59.23 % | -0.12 pp | 58.73 % | -0.62 pp | 58.46 % | -0.89 pp |
| Memory Throughput (%) | 37.50 % | 36.90 % | -0.60 pp | 36.81 % | -0.69 pp | 36.13 % | -1.37 pp | 37.68 % | +0.18 pp |
| DRAM Throughput (%) | 37.50 % | 36.90 % | -0.60 pp | 36.81 % | -0.69 pp | 36.13 % | -1.37 pp | 37.68 % | +0.18 pp |
| Achieved Occupancy (%) | 76.53 % | 76.21 % | -0.32 pp | 76.59 % | +0.06 pp | 75.07 % | -1.46 pp | 77.42 % | +0.89 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 4.64 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 |
| No Eligible (%) | 23.48 % | 23.45 % | -0.03 pp | 24.68 % | +1.20 pp | 23.42 % | -0.06 pp | 25.35 % | +1.87 pp |
| IPC Active | 2.99 inst/cycle | 3.02 inst/cycle | +0.03 | 3.02 inst/cycle | +0.03 | 3.01 inst/cycle | +0.02 | 3.02 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fused_fp32_multiply_adds_into_fma_instructions_via_libdevice | 1.030× | 0.957× | ✓ |  |
| B | fuse_non_fused_fp32_multiply_adds_into_fma_instructions_via_explicit_expression_rewrite | 1.020× | 0.983× | ✓ |  |
| C | reduce_num_warps_to_alleviate_scheduler_not_selected_stall_and_improve_cache_locality | 1.017× | 1.000× | ✓ |  |
| D | fuse_avgpool_accumulation_and_batchnorm_arithmetic_into_fma_chains | 1.012× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fused_fp... | Δ | fuse_non_fused_fp... | Δ | reduce_num_warps_... | Δ | fuse_avgpool_accu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9280 ns | 9700 ns | +420 ns (+4.5%) | 9440 ns | +160 ns (+1.7%) | 9280 ns | +0 ns (+0.0%) | 9380 ns | +100 ns (+1.1%) |
| SM Throughput (%) | 59.35 % | 59.16 % | -0.19 pp | 58.14 % | -1.21 pp | 57.32 % | -2.03 pp | 58.25 % | -1.10 pp |
| Memory Throughput (%) | 37.50 % | 35.96 % | -1.54 pp | 37.00 % | -0.50 pp | 37.55 % | +0.05 pp | 37.11 % | -0.39 pp |
| DRAM Throughput (%) | 37.50 % | 35.96 % | -1.54 pp | 37.00 % | -0.50 pp | 37.55 % | +0.05 pp | 37.11 % | -0.39 pp |
| Achieved Occupancy (%) | 76.53 % | 78.17 % | +1.64 pp | 76.84 % | +0.31 pp | 77.11 % | +0.58 pp | 77.61 % | +1.08 pp |
| Registers/Thread | 28.00 register/thread | 26.00 register/thread | -2 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 26.00 register/thread | -2 |
| Waves/SM | 4.64 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 | 4.64 | +0.00 |
| No Eligible (%) | 23.48 % | 23.13 % | -0.35 pp | 23.43 % | -0.05 pp | 23.22 % | -0.26 pp | 24.42 % | +0.94 pp |
| IPC Active | 2.99 inst/cycle | 3.03 inst/cycle | +0.04 | 2.99 inst/cycle | +0.00 | 3.01 inst/cycle | +0.02 | 2.93 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  9280 ns  (1.000×)

Final best: 9500 ns  (0.9768×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
@@ -25,12 +26,12 @@

     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 128, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -39,7 +40,7 @@

     tmp16 = tmp0 >= tmp3
     tmp17 = tl.full([1], 160, tl.int64)
     tmp18 = tmp0 < tmp17
-    tmp19 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp16 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp19 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp16 & xmask, eviction_policy='evict_first', other=0.0)
     tmp20 = tl.where(tmp4, tmp15, tmp19)
     tmp22 = tmp20 - tmp21
     tmp24 = tl.full([1], 1e-05, tl.float32)
@@ -49,7 +50,6 @@

     tmp33 = tmp31 * tmp32 + tmp34
     tmp37 = tl.maximum(tl.zeros_like(tmp33), tmp33)
     tl.store(in_out_ptr0 + (x4), tmp37, mask=xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
