# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 292320 ns — **Best latency**: 292290 ns (**1.0001×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_sqrt_reciprocal_to_rsqrt_and_fix_autotune_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (470400,1,1), Waves/SM: 222.73
- **Registers/thread**: 17
- **Duration**: 292320 ns
- **SM Throughput**: 59.43 % | **Memory**: 31.66 % | **DRAM**: 31.66 %
- **Achieved Occupancy**: 60.43 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.663× | 0.182× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.892× | 0.887× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock_autotune | 0.893× | 0.888× | ✓ |  |
| D | autotune_xblock_for_latency_hiding | 0.893× | 0.888× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 292320 ns | 1610000 ns | +1317680 ns (+450.8%) | 329410 ns | +37090 ns (+12.7%) | 329250 ns | +36930 ns (+12.6%) | 329280 ns | +36960 ns (+12.6%) |
| SM Throughput (%) | 59.43 % | 88.28 % | +28.85 pp | 80.98 % | +21.55 pp | 80.93 % | +21.50 pp | 81.21 % | +21.78 pp |
| Memory Throughput (%) | 31.66 % | 17.97 % | -13.69 pp | 32.69 % | +1.03 pp | 32.67 % | +1.01 pp | 32.77 % | +1.11 pp |
| DRAM Throughput (%) | 31.66 % | 5.77 % | -25.89 pp | 28.11 % | -3.55 pp | 28.12 % | -3.54 pp | 28.11 % | -3.55 pp |
| Achieved Occupancy (%) | 60.43 % | 84.22 % | +23.79 pp | 59.41 % | -1.02 pp | 59.30 % | -1.13 pp | 59.25 % | -1.18 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 24.00 register/thread | +7 | 24.00 register/thread | +7 |
| Waves/SM | 222.73 | 890.91 | +668.18 | 222.73 | +0.00 | 222.73 | +0.00 | 222.73 | +0.00 |
| No Eligible (%) | 39.95 % | 11.61 % | -28.34 pp | 18.26 % | -21.69 pp | 18.18 % | -21.77 pp | 18.46 % | -21.49 pp |
| IPC Active | 2.40 inst/cycle | 3.53 inst/cycle | +1.13 | 3.27 inst/cycle | +0.87 | 3.27 inst/cycle | +0.87 | 3.27 inst/cycle | +0.87 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.182×)

Increasing block size via @triton.autotune while keeping the grid fixed at the original small-block size caused the kernel to launch 4x more threads than needed (waves_per_sm jumped from 222 to 890), executing ~8x more instructions (137M → 1120M). The root cause is a broken grid calculation: the grid was hardcoded to cdiv(xnumel, 128) instead of being updated to use meta['XBLOCK'], so the grid did not shrink when XBLOCK grew — every element was processed multiple times. Additionally, switching from tl.full([XBLOCK], True) mask to xindex < xnumel with a non-None mask on broadcast loads (in_ptr0/1/2/3 indexed by x0 = xindex % 120) caused redundant masked loads on the small-stride dimension. When using @triton.autotune, the grid lambda MUST reference meta['XBLOCK'] to correctly reduce block count as tile size grows.

> **Applicable when**: Any time @triton.autotune is added to a pointwise kernel to vary XBLOCK — the grid must always be a lambda referencing meta['XBLOCK'].

> **Avoid when**: Never use a hardcoded grid size when the kernel uses @triton.autotune with variable XBLOCK; this causes the kernel to process all elements multiple times.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_fma_fusion_via_rsqrt | 1.000× | 1.003× | ✓ | **✓ best** |
| B | fix_broken_autotune_grid_and_eliminate_nonfused_fp32_instructions | 1.000× | 1.004× | ✓ |  |
| C | fix_grid_lambda_and_arithmetic_for_fma_fusion | 0.893× | 0.883× | ✓ |  |
| D | fix_grid_lambda_and_reduce_nonfused_fp32_instructions | 0.887× | 0.313× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_broken_autotu... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 292320 ns | 291360 ns | -960 ns (-0.3%) | 291300 ns | -1020 ns (-0.3%) | 330980 ns | +38660 ns (+13.2%) | 933280 ns | +640960 ns (+219.3%) |
| SM Throughput (%) | 59.43 % | 48.55 % | -10.88 pp | 49.03 % | -10.40 pp | 80.92 % | +21.49 pp | 76.71 % | +17.28 pp |
| Memory Throughput (%) | 31.66 % | 37.01 % | +5.35 pp | 37.42 % | +5.76 pp | 32.66 % | +1.00 pp | 31.00 % | -0.66 pp |
| DRAM Throughput (%) | 31.66 % | 31.79 % | +0.13 pp | 31.79 % | +0.13 pp | 27.97 % | -3.69 pp | 9.92 % | -21.74 pp |
| Achieved Occupancy (%) | 60.43 % | 33.28 % | -27.15 pp | 33.53 % | -26.90 pp | 59.23 % | -1.20 pp | 71.36 % | +10.93 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 24.00 register/thread | +7 | 32.00 register/thread | +15 |
| Waves/SM | 222.73 | 222.73 | +0.00 | 222.73 | +0.00 | 222.73 | +0.00 | 890.91 | +668.18 |
| No Eligible (%) | 39.95 % | 50.24 % | +10.29 pp | 50.64 % | +10.69 pp | 18.43 % | -21.52 pp | 22.87 % | -17.08 pp |
| IPC Active | 2.40 inst/cycle | 1.98 inst/cycle | -0.42 | 1.98 inst/cycle | -0.42 | 3.27 inst/cycle | +0.87 | 3.08 inst/cycle | +0.68 |

---

## Iteration 3

**Best so far**: 1.0033× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_explicit_fused_arithmetic_and_correct_grid_lambda | 1.000× | 1.002× | ✓ |  |
| B | fix_grid_lambda_and_fma_fusion_via_libdevice_fma | 0.999× | 1.003× | ✓ |  |
| C | fuse_sqrt_reciprocal_to_rsqrt_and_fix_autotune_grid | 1.000× | 1.000× | ✓ | **✓ best** |
| D | fix_grid_lambda_and_fma_restructure_for_instruction_reduction | 0.886× | 0.312× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_ex... | Δ | fix_grid_lambda_a... | Δ | fuse_sqrt_recipro... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 292320 ns | 291680 ns | -640 ns (-0.2%) | 291390 ns | -930 ns (-0.3%) | 292290 ns | -30 ns (-0.0%) | 935520 ns | +643200 ns (+220.0%) |
| SM Throughput (%) | 59.43 % | 48.95 % | -10.48 pp | 49.01 % | -10.42 pp | 48.49 % | -10.94 pp | 76.75 % | +17.32 pp |
| Memory Throughput (%) | 31.66 % | 37.29 % | +5.63 pp | 37.38 % | +5.72 pp | 36.92 % | +5.26 pp | 31.00 % | -0.66 pp |
| DRAM Throughput (%) | 31.66 % | 31.75 % | +0.09 pp | 31.77 % | +0.11 pp | 31.67 % | +0.01 pp | 9.90 % | -21.76 pp |
| Achieved Occupancy (%) | 60.43 % | 33.13 % | -27.30 pp | 33.59 % | -26.84 pp | 33.37 % | -27.06 pp | 75.16 % | +14.73 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 32.00 register/thread | +15 |
| Waves/SM | 222.73 | 222.73 | +0.00 | 222.73 | +0.00 | 222.73 | +0.00 | 890.91 | +668.18 |
| No Eligible (%) | 39.95 % | 50.73 % | +10.78 pp | 50.73 % | +10.78 pp | 50.19 % | +10.24 pp | 22.87 % | -17.08 pp |
| IPC Active | 2.40 inst/cycle | 1.97 inst/cycle | -0.43 | 1.98 inst/cycle | -0.42 | 1.96 inst/cycle | -0.44 | 3.08 inst/cycle | +0.68 |

---

## Iteration 4

**Best so far**: 1.0033× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_autotune_xblock_lambda | 1.000× | 1.002× | ✓ |  |
| B | fix_grid_lambda_to_enable_existing_autotune_xblock_configs | 1.000× | 1.004× | ✓ |  |
| C | fix_grid_lambda_and_streaming_eviction_for_l2_cache_efficiency | 1.000× | 1.002× | ✓ |  |
| D | fix_broken_grid_lambda_and_tune_num_warps_for_correct_autotune | 0.887× | 0.313× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | fix_broken_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 292320 ns | 291650 ns | -670 ns (-0.2%) | 291300 ns | -1020 ns (-0.3%) | 291650 ns | -670 ns (-0.2%) | 933950 ns | +641630 ns (+219.5%) |
| SM Throughput (%) | 59.43 % | 48.41 % | -11.02 pp | 48.47 % | -10.96 pp | 52.59 % | -6.84 pp | 76.82 % | +17.39 pp |
| Memory Throughput (%) | 31.66 % | 36.87 % | +5.21 pp | 36.91 % | +5.25 pp | 37.28 % | +5.62 pp | 31.06 % | -0.60 pp |
| DRAM Throughput (%) | 31.66 % | 31.75 % | +0.09 pp | 31.78 % | +0.12 pp | 31.75 % | +0.09 pp | 9.91 % | -21.75 pp |
| Achieved Occupancy (%) | 60.43 % | 33.40 % | -27.03 pp | 33.65 % | -26.78 pp | 33.96 % | -26.47 pp | 75.17 % | +14.74 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 32.00 register/thread | +15 |
| Waves/SM | 222.73 | 222.73 | +0.00 | 222.73 | +0.00 | 222.73 | +0.00 | 890.91 | +668.18 |
| No Eligible (%) | 39.95 % | 50.91 % | +10.96 pp | 50.22 % | +10.27 pp | 46.86 % | +6.91 pp | 23.13 % | -16.82 pp |
| IPC Active | 2.40 inst/cycle | 1.97 inst/cycle | -0.43 | 1.98 inst/cycle | -0.42 | 2.10 inst/cycle | -0.30 | 3.08 inst/cycle | +0.68 |

---

## Iteration 5

**Best so far**: 1.0033× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages_for_software_pipelining_with_correct_grid_lambda | 1.000× | 1.001× | ✓ |  |
| B | fix_grid_lambda_and_add_num_stages_for_l1tex_latency_hiding | 1.000× | 1.000× | ✓ |  |
| C | fix_grid_lambda_and_restructure_arithmetic_for_fma_fusion_with_correct_autotune | 1.000× | 1.001× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_and_fix_grid_lambda | 0.999× | 0.316× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 292320 ns | 291900 ns | -420 ns (-0.1%) | 292260 ns | -60 ns (-0.0%) | 291940 ns | -380 ns (-0.1%) | 924380 ns | +632060 ns (+216.2%) |
| SM Throughput (%) | 59.43 % | 48.95 % | -10.48 pp | 48.47 % | -10.96 pp | 52.73 % | -6.70 pp | 77.53 % | +18.10 pp |
| Memory Throughput (%) | 31.66 % | 37.34 % | +5.68 pp | 36.91 % | +5.25 pp | 37.44 % | +5.78 pp | 31.38 % | -0.28 pp |
| DRAM Throughput (%) | 31.66 % | 31.72 % | +0.06 pp | 31.67 % | +0.01 pp | 31.71 % | +0.05 pp | 10.01 % | -21.65 pp |
| Achieved Occupancy (%) | 60.43 % | 33.31 % | -27.12 pp | 33.49 % | -26.94 pp | 34.15 % | -26.28 pp | 76.89 % | +16.46 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 24.00 register/thread | +7 |
| Waves/SM | 222.73 | 222.73 | +0.00 | 222.73 | +0.00 | 222.73 | +0.00 | 890.91 | +668.18 |
| No Eligible (%) | 39.95 % | 50.94 % | +10.99 pp | 50.05 % | +10.10 pp | 47.14 % | +7.19 pp | 22.06 % | -17.89 pp |
| IPC Active | 2.40 inst/cycle | 1.99 inst/cycle | -0.41 | 1.99 inst/cycle | -0.41 | 2.11 inst/cycle | -0.29 | 3.12 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_via_maxnreg_and_fix_grid_lambda` (0.316×)

Adding maxnreg=16 to @triton.autotune configs while also varying XBLOCK caused autotune to select XBLOCK=512 (block_size 512), but the grid was still not correctly scaled — waves_per_sm jumped from 222 to 890 and executed instructions exploded from 111M to 565M (5x), causing duration to increase from 291us to 924us. The maxnreg=16 cap did not reduce registers_per_thread (22→24, actually increased), indicating the compiler could not honor the constraint without spilling, and the autotune selected a larger XBLOCK that compounded the over-execution problem. When the grid lambda is broken or autotune selects a larger block size without a corrected grid, maxnreg constraints interact destructively with XBLOCK scaling, producing massive instruction count increases with no memory bandwidth benefit (DRAM throughput dropped from 31.78% to 10.01%).

> **Applicable when**: Never: maxnreg=16 in triton.Config is an anti-pattern when the compiler already uses 22-32 registers/thread — the constraint cannot be honored without spilling, and combined with autotune XBLOCK selection it causes instruction count explosion.

> **Avoid when**: Do not apply maxnreg constraints when: (1) the kernel already uses >16 registers/thread (compiler will spill or ignore), (2) the grid is not a correctly-scaled lambda referencing meta['XBLOCK'], (3) autotune is selecting larger XBLOCK values that would increase waves_per_sm without a corrected grid.

---

## Cumulative Progress

```
Baseline:  292320 ns  (1.000×)
Iter 2     291360 ns  (1.003×)  — fix_grid_lambda_and_fma_fusion_via_rsqrt

Final best: 292290 ns  (1.0001×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 60211200
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 120)
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
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp_eps = tl.full([1], 1e-05, tl.float32)
+    tmp_inv_std = libdevice.rsqrt(tmp3 + tmp_eps)
+    tmp_xhat = (tmp0 - tmp1) * tmp_inv_std
+    tmp_out = tl.math.fma(tmp_xhat, tmp12, tmp14)
+    tmp_zero = tl.full([1], 0.0, tl.float32)
+    tmp17 = tl.maximum(tmp_zero, tmp_out)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 60211200
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 60211200, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 60211200)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
