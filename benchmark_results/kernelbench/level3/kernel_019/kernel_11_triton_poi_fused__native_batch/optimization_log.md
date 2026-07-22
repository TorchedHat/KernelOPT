# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12290 ns — **Best latency**: 12220 ns (**1.0057×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`explicit_fma_and_scalar_epsilon_to_enable_fused_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12290 ns
- **SM Throughput**: 47.74 % | **Memory**: 21.07 % | **DRAM**: 13.35 %
- **Achieved Occupancy**: 56.11 %
- **IPC Active**: 2.35 inst/cycle | **No Eligible**: 41.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.767× | 0.164× | ✓ |  |
| B | autotune_xblock_num_warps_to_hide_l1tex_latency | 0.781× | 0.719× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 1.010× | 0.998× | ✓ | **✓ best** |
| D | autotune_xblock_to_hide_l1tex_latency | 0.998× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 74780 ns | +62490 ns (+508.5%) | 17090 ns | +4800 ns (+39.1%) | 12320 ns | +30 ns (+0.2%) | 12350 ns | +60 ns (+0.5%) |
| SM Throughput (%) | 47.74 % | 81.51 % | +33.77 pp | 76.42 % | +28.68 pp | 56.40 % | +8.66 pp | 56.27 % | +8.53 pp |
| Memory Throughput (%) | 21.07 % | 22.15 % | +1.08 pp | 21.91 % | +0.84 pp | 21.32 % | +0.25 pp | 21.14 % | +0.07 pp |
| DRAM Throughput (%) | 13.35 % | 2.19 % | -11.16 pp | 9.61 % | -3.74 pp | 13.32 % | -0.03 pp | 13.30 % | -0.05 pp |
| Achieved Occupancy (%) | 56.11 % | 77.20 % | +21.09 pp | 85.23 % | +29.12 pp | 60.18 % | +4.07 pp | 59.79 % | +3.68 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 59.39 | +51.97 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 41.58 % | 16.06 % | -25.52 pp | 15.48 % | -26.10 pp | 30.35 % | -11.23 pp | 30.73 % | -10.85 pp |
| IPC Active | 2.35 inst/cycle | 3.36 inst/cycle | +1.01 | 3.44 inst/cycle | +1.09 | 2.78 inst/cycle | +0.43 | 2.75 inst/cycle | +0.40 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.246×)

Introducing @triton.autotune with larger XBLOCK configs (256–2048) while the kernel_function grid remains hardcoded to `triton.cdiv(xnumel, 128)` causes the autotuner to select XBLOCK=1024 but launch 15680 blocks, processing 16M elements instead of ~2M — an 8x work explosion confirmed by waves_per_sm jumping from 7.42 to 59.39 and executed instructions rising from 6M to 28.7M. Duration increased 4x (12.45µs → 49.95µs) despite marginal improvements in eligible warps (2.43→5.26) and compute throughput (62%→72%). The grid MUST be a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` whenever autotune varies XBLOCK; a hardcoded integer grid is a correctness bug that causes proportional work duplication.

> **Applicable when**: Any time @triton.autotune is added or modified to vary XBLOCK in a pointwise Triton kernel and the kernel_function wrapper defines the grid as a pre-computed integer tuple.

> **Avoid when**: N/A — this is a correctness constraint. There is no scenario where hardcoding the old tile size in the grid is correct when autotune varies XBLOCK. Waves_per_sm exploding (e.g., 7→59) after adding autotune is the definitive diagnostic signal.

#### ❌ Plan B: `autotune_xblock_num_warps_to_hide_l1tex_latency` (0.719×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) and higher num_warps (4–16) doubled registers per thread from 17 to 32 and increased executed instructions by 2.2x (4.6M → 10.1M), causing a 39% latency increase despite improving occupancy (56% → 85%) and eligible warps (1.26 → 6.9). The autotuner selected a config that increased compute throughput (47% → 76%) but the instruction count explosion dominated, indicating the larger tile sizes caused redundant computation or unrolling overhead that outweighed latency-hiding benefits. When the original kernel already has very low register usage (≤17/thread) and the bottleneck is L1TEX stalls, simply enlarging XBLOCK via autotune can backfire if the selected config inflates register pressure and instruction count beyond what the latency hiding gain can compensate.

> **Applicable when**: Never: this specific pattern (autotune XBLOCK upward for pointwise kernels with L1TEX stalls) caused regression here. Avoid when the kernel has very low register count (≤17/thread) that may balloon under larger tile sizes, and when instruction count is already proportional to work done.

> **Avoid when**: Kernel is a simple pointwise op with low register pressure (≤20 registers/thread); increasing XBLOCK via autotune can cause the compiler to allocate significantly more registers (e.g., 17→32) and explode instruction count (2x+), negating any latency-hiding benefit. Also avoid when the grid is already well-matched to SM count (waves_per_sm ~7) and the bottleneck may be partially self-correcting.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 1.012× | 0.985× | ✓ | **✓ best** |
| B | add_evict_first_on_streaming_tensor_to_improve_l1_hit_rate | 1.009× | 0.987× | ✓ |  |
| C | fix_grid_lambda_and_eliminate_int_to_float_conversions_for_fma_fusion | 0.774× | 0.727× | ✓ |  |
| D | fix_grid_lambda_and_simplify_arithmetic_for_fma_fusion | 0.911× | 0.246× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | add_evict_first_o... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 12480 ns | +190 ns (+1.5%) | 12450 ns | +160 ns (+1.3%) | 16900 ns | +4610 ns (+37.5%) | 49980 ns | +37690 ns (+306.7%) |
| SM Throughput (%) | 47.74 % | 56.15 % | +8.41 pp | 62.51 % | +14.77 pp | 75.66 % | +27.92 pp | 72.16 % | +24.42 pp |
| Memory Throughput (%) | 21.07 % | 21.21 % | +0.14 pp | 21.15 % | +0.08 pp | 21.60 % | +0.53 pp | 33.38 % | +12.31 pp |
| DRAM Throughput (%) | 13.35 % | 13.15 % | -0.20 pp | 13.19 % | -0.16 pp | 9.71 % | -3.64 pp | 3.27 % | -10.08 pp |
| Achieved Occupancy (%) | 56.11 % | 59.39 % | +3.28 pp | 64.08 % | +7.97 pp | 85.55 % | +29.44 pp | 65.72 % | +9.61 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 22.00 register/thread | +5 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 59.39 | +51.97 |
| No Eligible (%) | 41.58 % | 30.59 % | -10.99 pp | 23.81 % | -17.77 pp | 13.53 % | -28.05 pp | 24.10 % | -17.48 pp |
| IPC Active | 2.35 inst/cycle | 2.74 inst/cycle | +0.39 | 3.03 inst/cycle | +0.68 | 3.43 inst/cycle | +1.08 | 3.03 inst/cycle | +0.68 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_int_cast_and_dead_multiply_to_enable_fma_fusion | 1.025× | 1.016× | ✓ |  |
| B | eliminate_int_to_float_cast_to_enable_fma_fusion | 1.027× | 1.019× | ✓ | **✓ best** |
| C | eliminate_redundant_multiply_by_one_to_enable_fma_fusion | 0.776× | 0.723× | ✓ |  |
| D | fix_grid_lambda_and_fuse_fp32_multiply_add_to_fma | 0.900× | 0.246× | ✓ |  |
| E | add_triton_autotune | 0.994× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_int_cas... | Δ | eliminate_int_to_... | Δ | eliminate_redunda... | Δ | fix_grid_lambda_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 12100 ns | -190 ns (-1.5%) | 12060 ns | -230 ns (-1.9%) | 16990 ns | +4700 ns (+38.2%) | 49950 ns | +37660 ns (+306.4%) | 12320 ns | +30 ns (+0.2%) |
| SM Throughput (%) | 47.74 % | 36.35 % | -11.39 pp | 34.25 % | -13.49 pp | 76.18 % | +28.44 pp | 72.64 % | +24.90 pp | 62.72 % | +14.98 pp |
| Memory Throughput (%) | 21.07 % | 21.14 % | +0.07 pp | 21.36 % | +0.29 pp | 21.84 % | +0.77 pp | 33.62 % | +12.55 pp | 21.23 % | +0.16 pp |
| DRAM Throughput (%) | 13.35 % | 13.54 % | +0.19 pp | 13.58 % | +0.23 pp | 9.63 % | -3.72 pp | 3.27 % | -10.08 pp | 13.33 % | -0.02 pp |
| Achieved Occupancy (%) | 56.11 % | 45.11 % | -11.00 pp | 43.16 % | -12.95 pp | 85.22 % | +29.11 pp | 65.74 % | +9.63 pp | 64.26 % | +8.15 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 59.39 | +51.97 | 7.42 | +0.00 |
| No Eligible (%) | 41.58 % | 53.81 % | +12.23 pp | 57.21 % | +15.63 pp | 13.35 % | -28.23 pp | 24.14 % | -17.44 pp | 22.82 % | -18.76 pp |
| IPC Active | 2.35 inst/cycle | 1.80 inst/cycle | -0.55 | 1.68 inst/cycle | -0.67 | 3.44 inst/cycle | +1.09 | 3.03 inst/cycle | +0.68 | 3.08 inst/cycle | +0.73 |


### Key Learnings

#### ❌ Plan C: `eliminate_redundant_multiply_by_one_to_enable_fma_fusion` (0.723×)

Removing redundant tl.full([1], 1.0) scalar multiplies to enable FMA fusion did not reduce instruction count — instead, the simultaneous autotune config change (XBLOCK 128→512-2048) caused registers per thread to jump from 18 to 32 and executed instructions to increase by 1.67x (6.0M → 10.1M), raising duration from 12.45us to 16.99us (a 36% regression). The arithmetic simplification itself was likely neutral or marginally positive, but the confounded autotune config change dominated by inflating register pressure and instruction count, identical to the pattern seen in iteration 1. The causal agent was the XBLOCK/num_warps autotune change, not the FMA arithmetic path — the two changes were bundled together, making it impossible to isolate the FMA effect, but the instruction count explosion (matching iteration 1's 10.1M figure exactly) confirms the autotune config is the culprit.

> **Applicable when**: Never in isolation for this kernel: the FMA simplification was bundled with a harmful autotune config change. The arithmetic simplification alone (removing dead tl.full multiplies) may be neutral or slightly positive, but cannot be confirmed without isolating it from the XBLOCK change.

> **Avoid when**: Do not bundle arithmetic simplifications with autotune config changes in the same iteration — the confounding makes it impossible to attribute causation. Do not apply large XBLOCK autotune configs (512-2048) to pointwise kernels with low register pressure (≤20 registers/thread); this pattern has now caused regression in two consecutive iterations (iter 1 and iter 3), both producing exactly 10.1M executed instructions and 32 registers/thread.

---

## Iteration 4

**Best so far**: 1.0191× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_rsqrt_to_eliminate_transcendental_from_hot_loop | 1.024× | 1.011× | ✓ |  |
| B | replace_libdevice_rsqrt_and_fix_relu_type_mismatch_at_fixed_xblock128 | 1.010× | 1.019× | ✓ |  |
| C | revert_xblock_to_128_and_replace_sqrt_with_rsqrt_for_fma_fusion | 1.012× | 1.019× | ✓ |  |
| D | fix_grid_lambda_and_constrain_xblock_to_prevent_work_explosion | 1.012× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_rsqrt_... | Δ | replace_libdevice... | Δ | revert_xblock_to_... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 12160 ns | -130 ns (-1.1%) | 12060 ns | -230 ns (-1.9%) | 12060 ns | -230 ns (-1.9%) | 12450 ns | +160 ns (+1.3%) |
| SM Throughput (%) | 47.74 % | 32.89 % | -14.85 pp | 36.90 % | -10.84 pp | 34.21 % | -13.53 pp | 38.09 % | -9.65 pp |
| Memory Throughput (%) | 21.07 % | 21.24 % | +0.17 pp | 21.37 % | +0.30 pp | 21.32 % | +0.25 pp | 28.34 % | +7.27 pp |
| DRAM Throughput (%) | 13.35 % | 13.49 % | +0.14 pp | 13.58 % | +0.23 pp | 13.58 % | +0.23 pp | 13.17 % | -0.18 pp |
| Achieved Occupancy (%) | 56.11 % | 42.64 % | -13.47 pp | 44.30 % | -11.81 pp | 43.64 % | -12.47 pp | 34.96 % | -21.15 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 41.58 % | 58.39 % | +16.81 pp | 53.39 % | +11.81 pp | 56.84 % | +15.26 pp | 48.86 % | +7.28 pp |
| IPC Active | 2.35 inst/cycle | 1.66 inst/cycle | -0.69 | 1.84 inst/cycle | -0.51 | 1.70 inst/cycle | -0.65 | 2.02 inst/cycle | -0.33 |

---

## Iteration 5

**Best so far**: 1.0191× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_and_scalar_epsilon_to_enable_fused_fp32_instructions | 1.029× | 1.006× | ✓ | **✓ best** |
| B | reduce_num_warps_to_increase_blocks_per_sm_for_latency_hiding | 1.013× | 1.019× | ✓ |  |
| C | fix_hardcoded_grid_and_add_num_stages_pipelining | 1.011× | 1.013× | ✓ |  |
| D | enable_l2_compression_on_relu_output_via_cache_streaming_store | 1.017× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_and_... | Δ | reduce_num_warps_... | Δ | fix_hardcoded_gri... | Δ | enable_l2_compres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 12220 ns | -70 ns (-0.6%) | 12060 ns | -230 ns (-1.9%) | 12130 ns | -160 ns (-1.3%) | 12260 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 47.74 % | 20.97 % | -26.77 pp | 34.22 % | -13.52 pp | 32.53 % | -15.21 pp | 38.22 % | -9.52 pp |
| Memory Throughput (%) | 21.07 % | 19.60 % | -1.47 pp | 21.28 % | +0.21 pp | 21.02 % | -0.05 pp | 28.44 % | +7.37 pp |
| DRAM Throughput (%) | 13.35 % | 13.43 % | +0.08 pp | 13.58 % | +0.23 pp | 13.53 % | +0.18 pp | 13.39 % | +0.04 pp |
| Achieved Occupancy (%) | 56.11 % | 30.72 % | -25.39 pp | 43.10 % | -13.01 pp | 42.55 % | -13.56 pp | 36.98 % | -19.13 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 7.42 | 3.71 | -3.71 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 41.58 % | 69.42 % | +27.84 pp | 56.94 % | +15.36 pp | 58.56 % | +16.98 pp | 48.97 % | +7.39 pp |
| IPC Active | 2.35 inst/cycle | 1.17 inst/cycle | -1.18 | 1.71 inst/cycle | -0.64 | 1.61 inst/cycle | -0.74 | 2.05 inst/cycle | -0.30 |

---

## Cumulative Progress

```
Baseline:  12290 ns  (1.000×)
Iter 3     12060 ns  (1.019×)  — eliminate_int_to_float_cast_to_enable_fma_fusion

Final best: 12220 ns  (1.0057×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 256)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp5 = tmp3 + 1e-5
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tmp15 = tl.fma(tmp11, tmp12, tmp14)
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +42,7 @@

     arg_4 = inputs[4]
     xnumel = 2007040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
