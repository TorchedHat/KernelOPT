# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 2850 ns (**1.0877×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fuse_fp32_instructions_via_rsqrt_reciprocal`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (784,1,1), Waves/SM: 0.37
- **Registers/thread**: 18
- **Duration**: 3100 ns
- **SM Throughput**: 10.57 % | **Memory**: 4.30 % | **DRAM**: 2.74 %
- **Achieved Occupancy**: 30.59 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 1.020× | 1.010× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 1.026× | 0.650× | ✓ | **✓ best** |
| C | vectorized_loads_stores_with_larger_xblock | 1.026× | 1.000× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.995× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 3070 ns | -30 ns (-1.0%) | 4770 ns | +1670 ns (+53.9%) | 3100 ns | +0 ns (+0.0%) | 3170 ns | +70 ns (+2.3%) |
| SM Throughput (%) | 10.57 % | 12.85 % | +2.28 pp | 46.14 % | +35.57 pp | 13.91 % | +3.34 pp | 7.76 % | -2.81 pp |
| Memory Throughput (%) | 4.30 % | 4.36 % | +0.06 pp | 9.64 % | +5.34 pp | 4.71 % | +0.41 pp | 3.96 % | -0.34 pp |
| DRAM Throughput (%) | 2.74 % | 2.75 % | +0.01 pp | 1.77 % | -0.97 pp | 2.73 % | -0.01 pp | 2.67 % | -0.07 pp |
| Achieved Occupancy (%) | 30.59 % | 26.27 % | -4.32 pp | 47.30 % | +16.71 pp | 27.85 % | -2.74 pp | 15.82 % | -14.77 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 48.00 register/thread | +30 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 0.37 | 0.37 | +0.00 | 1.19 | +0.82 | 0.37 | +0.00 | 0.19 | -0.18 |
| No Eligible (%) | 71.96 % | 63.43 % | -8.53 pp | 24.62 % | -47.34 pp | 62.80 % | -9.16 pp | 78.82 % | +6.86 pp |
| IPC Active | 1.14 inst/cycle | 1.34 inst/cycle | +0.20 | 2.83 inst/cycle | +1.69 | 1.24 inst/cycle | +0.10 | 0.80 inst/cycle | -0.34 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_increase_tile_size` (0.650×)

Adding @triton.autotune with larger XBLOCK configs caused the autotuner to select XBLOCK=256 with num_warps=4/num_stages=2, which ballooned registers_per_thread from 18 to 48. This register pressure capped theoretical occupancy at 62.5% (Block Limit Registers dropped to 5 blocks/SM) and increased total executed instructions by 6.3x (269K → 1.71M), causing duration to increase from 3.1 µs to 4.77 µs despite better scheduler utilization. The root cause is that larger XBLOCK configs in the autotune search space force the compiler to allocate more registers to hold intermediate values across the wider tile, creating a register-pressure-induced occupancy ceiling that outweighs the wave-count benefit.

> **Applicable when**: Never apply this specific pattern. Document as anti-pattern: do not add autotune with large XBLOCK configs to short-duration pointwise kernels where the baseline register count is already low (≤20 regs/thread) and the kernel body contains multiple intermediate FP32 temporaries.

> **Avoid when**: Do not add @triton.autotune with XBLOCK configs larger than the baseline when: (1) baseline registers_per_thread is low (≤20) suggesting the compiler is already near register budget, (2) the kernel is a short pointwise op (duration <5 µs) where autotune overhead and instruction count inflation dominate, (3) the grid is already small enough that larger tiles create partial-wave tail effects (LaunchConfiguration rule fires with 50% tail penalty). The autotune search space must be constrained to configs that do not increase register pressure beyond the SM's register file budget.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_enable_autotune_xblock_coverage | 0.985× | 0.978× | ✓ |  |
| B | fuse_fp32_instructions_via_rsqrt_reciprocal | 1.069× | 1.088× | ✓ |  |
| C | fix_autotune_bypass_and_simplify_bn_arithmetic | 1.074× | 1.088× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_arithmetic_simplification | 1.064× | 1.076× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fuse_fp32_instruc... | Δ | fix_autotune_bypa... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 3170 ns | +70 ns (+2.3%) | 2850 ns | -250 ns (-8.1%) | 2850 ns | -250 ns (-8.1%) | 2880 ns | -220 ns (-7.1%) |
| SM Throughput (%) | 10.57 % | 7.89 % | -2.68 pp | 8.62 % | -1.95 pp | 5.10 % | -5.47 pp | 7.97 % | -2.60 pp |
| Memory Throughput (%) | 4.30 % | 4.00 % | -0.30 pp | 5.42 % | +1.12 pp | 4.38 % | +0.08 pp | 4.54 % | +0.24 pp |
| DRAM Throughput (%) | 2.74 % | 2.67 % | -0.07 pp | 2.94 % | +0.20 pp | 2.93 % | +0.19 pp | 2.93 % | +0.19 pp |
| Achieved Occupancy (%) | 30.59 % | 15.73 % | -14.86 pp | 25.97 % | -4.62 pp | 15.06 % | -15.53 pp | 31.42 % | +0.83 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 22.00 register/thread | +4 | 22.00 register/thread | +4 | 18.00 register/thread | +0 |
| Waves/SM | 0.37 | 0.19 | -0.18 | 0.37 | +0.00 | 0.19 | -0.18 | 0.37 | +0.00 |
| No Eligible (%) | 71.96 % | 78.15 % | +6.19 pp | 75.56 % | +3.60 pp | 85.67 % | +13.71 pp | 78.34 % | +6.38 pp |
| IPC Active | 1.14 inst/cycle | 0.80 inst/cycle | -0.34 | 0.96 inst/cycle | -0.18 | 0.47 inst/cycle | -0.67 | 0.70 inst/cycle | -0.44 |


### Key Learnings

#### ✅ Plan B: `fuse_fp32_instructions_via_rsqrt_reciprocal` (1.088×)

When a Triton kernel computes 1/sqrt(x) via an integer-constant division path (tl.full([1], 1, tl.int32) / tl.sqrt_rn(x)), the type-conversion forces the compiler through the integer pipeline and breaks FMA fusion, producing a high ratio of non-fused FP32 instructions (75% non-fused in this case, flagged by the FPInstructions NCU rule). Replacing the 6-instruction sequence with a single tl.math.rsqrt(x) call collapses it to one MUFU.RSQ instruction, halving total executed instructions (348K → 185K) and reducing kernel duration by ~7%. A secondary benefit is eliminating integer-zero constants in tl.maximum (replacing tl.full([1], 0, tl.int32) with tl.full([1], 0.0, tl.float32)) which removes additional type-promotion branches, dropping divergent branches from 5.94 to 0 and branch instructions from 36K to 4.7K.

> **Applicable when**: FPInstructions NCU rule fires with high non-fused FP32 ratio (>50% non-fused); kernel source contains integer-constant division to compute reciprocal or reciprocal-sqrt (pattern: tl.full([1], 1, tl.int32) / tl.sqrt_rn(x)); kernel also uses integer zero constants in tl.maximum for ReLU. Particularly effective for short pointwise kernels (duration <5 µs) where instruction count reduction directly maps to latency reduction.

> **Avoid when**: Do not apply if the computation requires double-precision accuracy (rsqrt is single-precision only). Do not apply if the epsilon value is not already added before the rsqrt call (must verify x > 0 to avoid NaN from rsqrt(0)). Do not apply if the NCU FPInstructions rule is not triggered or non-fused ratio is low (<20%), as the gain will be negligible.

#### ✅ Plan D: `reduce_register_pressure_via_arithmetic_simplification` (1.076×)

Replacing a multi-step batch normalization computation chain (sqrt + integer division + no-op multiply, ~7 temporaries) with a single tl.math.rsqrt call reduced registers_per_thread from 24 to 18 and halved executed instructions (348K → 163K). This lowered kernel duration from 3.07 µs to 2.88 µs (1.076x speedup) and improved achieved occupancy from 26.3% to 31.4% by reducing live register pressure. When a pointwise kernel shows elevated registers_per_thread alongside redundant arithmetic intermediates (e.g., 1/sqrt(x) computed as sqrt then integer divide then multiply-by-1.0), collapsing them to a single hardware SFU instruction (rsqrt) reduces the compiler's live-value count and frees registers.

> **Applicable when**: Kernel shows registers_per_thread higher than expected for its arithmetic complexity; NCU reports AchievedOccupancy well below TheoreticalOccupancy; the kernel body contains redundant intermediate temporaries for computing 1/sqrt(x) via separate sqrt + divide + multiply-by-1 steps; kernel is a short pointwise op (duration <5 µs) where instruction count reduction directly translates to latency savings.

> **Avoid when**: Do not apply when the arithmetic chain is not redundant (each step is semantically necessary); when the kernel is already at minimum register count (≤16 regs/thread); when numerical precision requirements differ between rsqrt and the explicit sqrt+divide path (rsqrt uses SFU approximation with ~23-bit precision, which is sufficient for BN but may not be for all use cases); when the kernel is already compute-bound and register pressure is not the bottleneck.

---

## Iteration 3

**Best so far**: 1.0877× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_xblock_coverage | 1.069× | 1.099× | ✓ |  |
| B | fix_hardcoded_grid_bypass_and_add_pid_swizzle_for_l2_locality | 1.008× | 0.987× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency_in_underutilized_pointwise_kernel | 1.029× | 0.923× | ✓ |  |
| D | rsqrt_arithmetic_simplification_with_corrected_lambda_grid | 1.062× | 1.076× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | increase_num_warp... | Δ | rsqrt_arithmetic_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2820 ns | -280 ns (-9.0%) | 3140 ns | +40 ns (+1.3%) | 3360 ns | +260 ns (+8.4%) | 2880 ns | -220 ns (-7.1%) |
| SM Throughput (%) | 10.57 % | 8.51 % | -2.06 pp | 5.97 % | -4.60 pp | 13.68 % | +3.11 pp | 5.15 % | -5.42 pp |
| Memory Throughput (%) | 4.30 % | 5.37 % | +1.07 pp | 4.14 % | -0.16 pp | 7.07 % | +2.77 pp | 4.58 % | +0.28 pp |
| DRAM Throughput (%) | 2.74 % | 2.98 % | +0.24 pp | 2.67 % | -0.07 pp | 2.50 % | -0.24 pp | 2.94 % | +0.20 pp |
| Achieved Occupancy (%) | 30.59 % | 26.59 % | -4.00 pp | 16.02 % | -14.57 pp | 57.78 % | +27.19 pp | 14.92 % | -15.67 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 24.00 register/thread | +6 | 19.00 register/thread | +1 | 22.00 register/thread | +4 |
| Waves/SM | 0.37 | 0.37 | +0.00 | 0.19 | -0.18 | 0.74 | +0.37 | 0.19 | -0.18 |
| No Eligible (%) | 71.96 % | 73.50 % | +1.54 pp | 81.35 % | +9.39 pp | 66.29 % | -5.67 pp | 86.14 % | +14.18 pp |
| IPC Active | 1.14 inst/cycle | 1.00 inst/cycle | -0.14 | 0.70 inst/cycle | -0.44 | 1.38 inst/cycle | +0.24 | 0.48 inst/cycle | -0.66 |

---

## Iteration 4

**Best so far**: 1.0877× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_to_hide_scheduler_stalls | 1.062× | 1.115× | ✓ |  |
| B | fix_hardcoded_grid_and_add_low_register_large_xblock_config | 1.030× | 1.030× | ✓ |  |
| C | fix_hardcoded_grid_and_remove_broken_swizzle_for_lambda_autotune | 1.055× | 1.115× | ✓ |  |
| D | pid_swizzle_for_l2_locality_and_workload_balance | 1.057× | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | pid_swizzle_for_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2780 ns | -320 ns (-10.3%) | 3010 ns | -90 ns (-2.9%) | 2780 ns | -320 ns (-10.3%) | 3230 ns | +130 ns (+4.2%) |
| SM Throughput (%) | 10.57 % | 4.36 % | -6.21 pp | 5.16 % | -5.41 pp | 4.24 % | -6.33 pp | 14.37 % | +3.80 pp |
| Memory Throughput (%) | 4.30 % | 2.55 % | -1.75 pp | 4.14 % | -0.16 pp | 2.51 % | -1.79 pp | 7.05 % | +2.75 pp |
| DRAM Throughput (%) | 2.74 % | 1.55 % | -1.19 pp | 2.79 % | +0.05 pp | 1.55 % | -1.19 pp | 2.60 % | -0.14 pp |
| Achieved Occupancy (%) | 30.59 % | 15.06 % | -15.53 pp | 14.85 % | -15.74 pp | 15.02 % | -15.57 pp | 57.69 % | +27.10 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 0.37 | 0.19 | -0.18 | 0.19 | -0.18 | 0.19 | -0.18 | 0.74 | +0.37 |
| No Eligible (%) | 71.96 % | 87.43 % | +15.47 pp | 87.24 % | +15.28 pp | 88.01 % | +16.05 pp | 69.32 % | -2.64 pp |
| IPC Active | 1.14 inst/cycle | 0.43 inst/cycle | -0.71 | 0.48 inst/cycle | -0.66 | 0.43 inst/cycle | -0.71 | 1.14 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)
Iter 2     2850 ns  (1.088×)  — fix_autotune_bypass_and_simplify_bn_arithmetic

Final best: 2850 ns  (1.0877×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 100352
@@ -19,21 +32,12 @@

     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp5 = tmp3 + tl.full([1], 1e-05, tl.float32)
+    tmp11 = tmp2 * tl.math.rsqrt(tmp5)
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp17 = tl.maximum(tl.full([1], 0.0, tl.float32), tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +47,7 @@

     arg_4 = inputs[4]
     xnumel = 100352
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
