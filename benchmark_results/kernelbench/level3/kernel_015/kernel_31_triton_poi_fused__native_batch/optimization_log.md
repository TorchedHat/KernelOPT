# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7620 ns — **Best latency**: 7330 ns (**1.0396×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_redundant_ops_to_enable_fma_fusion_via_rsqrt`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_31_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7620 ns
- **SM Throughput**: 39.03 % | **Memory**: 17.11 % | **DRAM**: 10.81 %
- **Achieved Occupancy**: 57.00 %
- **IPC Active**: 2.25 inst/cycle | **No Eligible**: 42.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.916× | 0.288× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.983× | 1.013× | ✓ |  |
| C | vectorized_loads_larger_xblock_autotune | 0.904× | 0.285× | ✓ |  |
| D | increase_xblock_with_autotune_to_eliminate_partial_wave_and_reduce_l1tex_stalls | 0.918× | 0.783× | ✓ |  |
| E | add_triton_autotune | 0.984× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | increase_xblock_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7620 ns | 26460 ns | +18840 ns (+247.2%) | 7520 ns | -100 ns (-1.3%) | 26750 ns | +19130 ns (+251.0%) | 9730 ns | +2110 ns (+27.7%) | 7490 ns | -130 ns (-1.7%) |
| SM Throughput (%) | 39.03 % | 81.72 % | +42.69 pp | 44.92 % | +5.89 pp | 81.61 % | +42.58 pp | 65.77 % | +26.74 pp | 43.95 % | +4.92 pp |
| Memory Throughput (%) | 17.11 % | 17.18 % | +0.07 pp | 16.78 % | -0.33 pp | 17.16 % | +0.05 pp | 17.90 % | +0.79 pp | 16.29 % | -0.82 pp |
| DRAM Throughput (%) | 10.81 % | 3.10 % | -7.71 pp | 10.96 % | +0.15 pp | 3.07 % | -7.74 pp | 8.45 % | -2.36 pp | 10.98 % | +0.17 pp |
| Achieved Occupancy (%) | 57.00 % | 54.83 % | -2.17 pp | 58.67 % | +1.67 pp | 54.85 % | -2.15 pp | 81.55 % | +24.55 pp | 61.57 % | +4.57 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 18.00 register/thread | +1 | 48.00 register/thread | +31 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 11.88 | +8.17 | 3.71 | +0.00 | 11.88 | +8.17 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.62 % | 11.27 % | -31.35 pp | 35.09 % | -7.53 pp | 11.25 % | -31.37 pp | 15.74 % | -26.88 pp | 36.19 % | -6.43 pp |
| IPC Active | 2.25 inst/cycle | 3.54 inst/cycle | +1.29 | 2.42 inst/cycle | +0.17 | 3.54 inst/cycle | +1.29 | 3.34 inst/cycle | +1.09 | 2.57 inst/cycle | +0.32 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_eliminate_partial_wave` (0.783×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate nearly 2x more registers per thread (17→32), reducing the register-limited block count per SM from 21 to 16. Although occupancy improved (57%→81.5%) and eligible warps per scheduler increased (1.39→6.47), the instruction count more than doubled (2.29M→5.03M) and kernel duration increased by 28% (7.62→9.73 µs). The structural anti-pattern is that larger XBLOCK tiles in pointwise kernels can cause the compiler to spill or allocate more registers to hold intermediate values across the wider tile, negating the latency-hiding benefit of higher occupancy.

> **Applicable when**: Kernel has very low registers/thread (≤20) AND block limit by registers is high (≥20 blocks/SM), suggesting true register headroom exists without compiler reallocation. Autotune XBLOCK increase is safe only when the compiler does NOT increase register count for larger tiles.

> **Avoid when**: The compiler increases registers/thread when XBLOCK grows (observable as Block Limit Registers decreasing in the after profile). If executed instructions more than double with larger XBLOCK for a pointwise kernel, the compiler is generating redundant or unrolled code rather than amortizing loads. Also avoid when the kernel already has low memory throughput (<20% DRAM) — the bottleneck is not memory latency hiding.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.000× | 0.979× | ✓ |  |
| B | fix_grid_wiring_and_increase_num_warps_for_latency_hiding | 1.000× | 1.017× | ✓ | **✓ best** |
| C | eliminate_redundant_ops_to_enable_fma_fusion | 0.903× | 0.812× | ✓ |  |
| D | remove_autotune_restore_static_mask_to_recover_register_occupancy | 1.000× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_grid_wiring_a... | Δ | eliminate_redunda... | Δ | remove_autotune_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7620 ns | 7780 ns | +160 ns (+2.1%) | 7490 ns | -130 ns (-1.7%) | 9380 ns | +1760 ns (+23.1%) | 7550 ns | -70 ns (-0.9%) |
| SM Throughput (%) | 39.03 % | 32.81 % | -6.22 pp | 45.65 % | +6.62 pp | 65.17 % | +26.14 pp | 38.46 % | -0.57 pp |
| Memory Throughput (%) | 17.11 % | 14.93 % | -2.18 pp | 17.03 % | -0.08 pp | 18.44 % | +1.33 pp | 16.75 % | -0.36 pp |
| DRAM Throughput (%) | 10.81 % | 10.60 % | -0.21 pp | 10.99 % | +0.18 pp | 8.76 % | -2.05 pp | 10.92 % | +0.11 pp |
| Achieved Occupancy (%) | 57.00 % | 48.51 % | -8.49 pp | 60.71 % | +3.71 pp | 80.84 % | +23.84 pp | 57.45 % | +0.45 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 17.00 register/thread | +0 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.62 % | 42.05 % | -0.57 pp | 33.68 % | -8.94 pp | 16.20 % | -26.42 pp | 42.48 % | -0.14 pp |
| IPC Active | 2.25 inst/cycle | 2.31 inst/cycle | +0.06 | 2.64 inst/cycle | +0.39 | 3.33 inst/cycle | +1.08 | 2.27 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0174× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_ops_to_enable_fma_fusion_via_rsqrt | 1.015× | 1.040× | ✓ | **✓ best** |
| B | simplify_batchnorm_arithmetic_to_enable_fma_without_autotune | 0.984× | 1.009× | ✓ |  |
| C | fix_grid_lambda_wiring_and_safe_num_warps_autotune_at_xblock128 | 0.999× | 1.005× | ✓ |  |
| D | remove_large_xblock_autotune_configs_to_restore_low_register_pressure | 1.001× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | simplify_batchnor... | Δ | fix_grid_lambda_w... | Δ | remove_large_xblo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7620 ns | 7330 ns | -290 ns (-3.8%) | 7550 ns | -70 ns (-0.9%) | 7580 ns | -40 ns (-0.5%) | 7650 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 39.03 % | 28.24 % | -10.79 pp | 38.48 % | -0.55 pp | 45.54 % | +6.51 pp | 31.90 % | -7.13 pp |
| Memory Throughput (%) | 17.11 % | 17.60 % | +0.49 pp | 16.68 % | -0.43 pp | 16.95 % | -0.16 pp | 15.03 % | -2.08 pp |
| DRAM Throughput (%) | 10.81 % | 11.23 % | +0.42 pp | 10.95 % | +0.14 pp | 10.82 % | +0.01 pp | 10.74 % | -0.07 pp |
| Achieved Occupancy (%) | 57.00 % | 48.81 % | -8.19 pp | 57.50 % | +0.50 pp | 61.30 % | +4.30 pp | 48.04 % | -8.96 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 17.00 register/thread | +0 | 18.00 register/thread | +1 | 24.00 register/thread | +7 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 1.86 | -1.85 |
| No Eligible (%) | 42.62 % | 60.17 % | +17.55 pp | 43.18 % | +0.56 pp | 33.68 % | -8.94 pp | 42.14 % | -0.48 pp |
| IPC Active | 2.25 inst/cycle | 1.62 inst/cycle | -0.63 | 2.26 inst/cycle | +0.01 | 2.62 inst/cycle | +0.37 | 2.23 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0396× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_replace_libdevice_rsqrt_with_native_intrinsic_for_fma_fusion | 0.994× | 1.027× | ✓ |  |
| B | replace_div_sqrt_with_rsqrt_to_enable_fma_fusion | 1.015× | 1.050× | ✓ |  |
| C | fma_fusion_via_arithmetic_simplification_of_batchnorm_chain | 0.995× | 1.050× | ✓ |  |
| D | fix_autotune_block_size_to_128_threads_with_safe_num_warps | 1.003× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | replace_div_sqrt_... | Δ | fma_fusion_via_ar... | Δ | fix_autotune_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7620 ns | 7420 ns | -200 ns (-2.6%) | 7260 ns | -360 ns (-4.7%) | 7260 ns | -360 ns (-4.7%) | 7550 ns | -70 ns (-0.9%) |
| SM Throughput (%) | 39.03 % | 30.94 % | -8.09 pp | 26.05 % | -12.98 pp | 28.24 % | -10.79 pp | 44.47 % | +5.44 pp |
| Memory Throughput (%) | 17.11 % | 17.61 % | +0.50 pp | 17.01 % | -0.10 pp | 17.62 % | +0.51 pp | 16.88 % | -0.23 pp |
| DRAM Throughput (%) | 10.81 % | 11.06 % | +0.25 pp | 11.34 % | +0.53 pp | 11.36 % | +0.55 pp | 10.93 % | +0.12 pp |
| Achieved Occupancy (%) | 57.00 % | 47.94 % | -9.06 pp | 46.42 % | -10.58 pp | 47.56 % | -9.44 pp | 59.91 % | +2.91 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.62 % | 53.32 % | +10.70 pp | 65.84 % | +23.22 pp | 57.11 % | +14.49 pp | 37.57 % | -5.05 pp |
| IPC Active | 2.25 inst/cycle | 1.87 inst/cycle | -0.38 | 1.33 inst/cycle | -0.92 | 1.71 inst/cycle | -0.54 | 2.58 inst/cycle | +0.33 |


### Key Learnings

#### ✅ Plan B: `replace_div_sqrt_with_rsqrt_to_enable_fma_fusion` (1.050×)

Replacing `1.0 / tl.sqrt_rn(x)` with `tl.math.rsqrt(x)` collapses a 3-instruction sequence (FADD + MUFU.RSQ + RCP) into a single MUFU.RSQ instruction, and chaining with `tl.math.fma()` further reduces a FMUL+FADD pair into one FFMA. This reduced total executed instructions by ~21% (1.63M → 1.29M) and registers per thread from 18 to 16, improving Block Limit Registers from 21 to 32 blocks/SM. The kernel achieved a modest 4.96% speedup (7.33 µs → 7.26 µs) despite the kernel remaining latency-bound (L1TEX stall ~57% of cycles), confirming that arithmetic instruction reduction helps even when memory latency is the primary bottleneck.

> **Applicable when**: Kernel has FPInstructions NCU rule triggered with high non-fused FP32 instruction fraction (>50% non-fused); kernel computes batch norm or similar normalize-scale-shift pattern using `1.0 / sqrt(x)` followed by multiply-add; kernel is partially compute-bound or has moderate instruction count overhead.

> **Avoid when**: Kernel requires exact IEEE 754 double-rounding semantics for the multiply-add (tl.math.fma has single rounding vs two-rounding for separate FMUL+FADD); kernel is purely memory-latency bound with negligible compute throughput (<5%) where instruction reduction has no effect on duration.

#### ✅ Plan C: `fma_fusion_via_arithmetic_simplification_of_batchnorm_chain` (1.050×)

When a batch norm normalization chain uses an integer literal (e.g., tl.full([1], 1, tl.int32)) to compute a reciprocal sqrt, the int-to-float cast breaks the FP32 dependency chain and prevents the PTX compiler from emitting FMA instructions. Replacing the multi-step chain (sqrt → int(1) → int/float divide → multiply-by-1.0) with a single libdevice.rsqrt() call eliminates 4 intermediate temporaries, reduces SM active cycles (~5% reduction: 7609 → 7234 cycles), and improves IPC active (1.62 → 1.71) and issue slot utilization (40.7% → 42.9%). The speedup is modest (~5%) because the kernel remains latency-bound on L1TEX scoreboard stalls (CPIStall still dominant), but the arithmetic simplification is correct and reduces instruction-level overhead.

> **Applicable when**: NCU FPInstructions rule fires with high non-fused to fused FP32 instruction ratio (e.g., 3:1 or worse). Kernel contains a reciprocal-sqrt pattern using integer literals (tl.full([1], 1, tl.int32)) that force int-to-float casts, breaking the FP32 dependency chain. Kernel is compute or latency bound with CPIStall on L1TEX. libdevice is already imported.

> **Avoid when**: When the kernel is purely memory-bandwidth bound (DRAM throughput >80%) and arithmetic simplification cannot move the bottleneck. Also avoid if rsqrt precision is insufficient for the use case (though for batch norm with eps>0 it is mathematically equivalent). Do not apply if the speedup from FPInstructions rule is already captured by other changes.

---

## Iteration 5

**Best so far**: 1.0396× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_tensor_to_reduce_l1tex_stalls | 1.014× | 1.021× | ✓ |  |
| B | fix_int_cast_in_relu_and_fuse_scale_shift_to_fma | 0.994× | 1.050× | ✓ |  |
| C | explicit_fma_chain_with_static_mask_to_reduce_non_fused_fp32_instructions | 0.995× | 1.050× | ✓ |  |
| D | fma_fusion_via_rsqrt_replacing_sqrt_division | 0.996× | 1.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_int_cast_in_r... | Δ | explicit_fma_chai... | Δ | fma_fusion_via_rs... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7620 ns | 7460 ns | -160 ns (-2.1%) | 7260 ns | -360 ns (-4.7%) | 7260 ns | -360 ns (-4.7%) | 7260 ns | -360 ns (-4.7%) |
| SM Throughput (%) | 39.03 % | 26.45 % | -12.58 pp | 27.76 % | -11.27 pp | 26.58 % | -12.45 pp | 28.24 % | -10.79 pp |
| Memory Throughput (%) | 17.11 % | 17.34 % | +0.23 pp | 17.31 % | +0.20 pp | 17.42 % | +0.31 pp | 17.55 % | +0.44 pp |
| DRAM Throughput (%) | 10.81 % | 11.04 % | +0.23 pp | 11.34 % | +0.53 pp | 11.30 % | +0.49 pp | 11.36 % | +0.55 pp |
| Achieved Occupancy (%) | 57.00 % | 45.50 % | -11.50 pp | 46.36 % | -10.64 pp | 48.41 % | -8.59 pp | 46.35 % | -10.65 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.62 % | 63.30 % | +20.68 pp | 57.34 % | +14.72 pp | 61.87 % | +19.25 pp | 58.47 % | +15.85 pp |
| IPC Active | 2.25 inst/cycle | 1.44 inst/cycle | -0.81 | 1.69 inst/cycle | -0.56 | 1.43 inst/cycle | -0.82 | 1.70 inst/cycle | -0.55 |

---

## Cumulative Progress

```
Baseline:  7620 ns  (1.000×)
Iter 2     7490 ns  (1.017×)  — fix_grid_wiring_and_increase_num_warps_for_latency_hiding
Iter 3     7330 ns  (1.040×)  — eliminate_redundant_ops_to_enable_fma_fusion_via_rsqrt

Final best: 7330 ns  (1.0396×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -18,22 +19,11 @@

     tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
+    tmp_centered = tmp0 - tmp1
+    tmp_inv_std = tl.math.rsqrt(tmp3 + 1e-5)
+    tmp_scaled = tl.math.fma(tmp_centered * tmp_inv_std, tmp12, tmp14)
+    tmp_out = tl.maximum(0.0, tmp_scaled)
+    tl.store(in_out_ptr0 + (x2), tmp_out, None)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
