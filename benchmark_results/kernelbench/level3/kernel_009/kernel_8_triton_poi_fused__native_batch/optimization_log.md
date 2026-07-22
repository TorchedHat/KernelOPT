# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 3390 ns (**1.0383×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`streaming_eviction_policy_for_main_tensor_plus_fma_cleanup`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_8_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 17
- **Duration**: 3520 ns
- **SM Throughput**: 16.34 % | **Memory**: 6.81 % | **DRAM**: 4.73 %
- **Achieved Occupancy**: 57.18 %
- **IPC Active**: 1.78 inst/cycle | **No Eligible**: 53.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_increase_work_per_cta | 0.996× | 0.480× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_utilization | 0.961× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 7330 ns | +3810 ns (+108.2%) | 3740 ns | +220 ns (+6.2%) |
| SM Throughput (%) | 16.34 % | 60.30 % | +43.96 pp | 33.63 % | +17.29 pp |
| Memory Throughput (%) | 6.81 % | 12.62 % | +5.81 pp | 8.91 % | +2.10 pp |
| DRAM Throughput (%) | 4.73 % | 2.27 % | -2.46 pp | 4.44 % | -0.29 pp |
| Achieved Occupancy (%) | 57.18 % | 50.74 % | -6.44 pp | 53.09 % | -4.09 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 32.00 register/thread | +15 |
| Waves/SM | 0.74 | 2.38 | +1.64 | 0.74 | +0.00 |
| No Eligible (%) | 53.51 % | 16.03 % | -37.48 pp | 36.51 % | -17.00 pp |
| IPC Active | 1.78 inst/cycle | 3.30 inst/cycle | +1.52 | 2.72 inst/cycle | +0.94 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_increase_work_per_cta` (0.480×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) selected a larger tile (XBLOCK=256) that caused registers-per-thread to jump from 17 to 48, reducing Block Limit Registers from 21 to 5 and capping theoretical occupancy at 62.5% instead of 100%. Simultaneously, executed instructions exploded 7.5x (from 457K to 3.4M), indicating the larger tile introduced redundant or unrolled computation far exceeding the original work. The net effect was a 2x increase in kernel duration (3.52 µs → 7.33 µs) despite the scheduler idle rate improving from 53.5% to 16%, because the register pressure and instruction count increase dominated. When a kernel already has low register usage (≤20 registers/thread) and the grid is small due to a genuinely small problem size rather than a tuning artifact, increasing XBLOCK via autotune risks register pressure blowup that more than offsets any latency-hiding benefit.

> **Applicable when**: Never use this specific pattern when registers_per_thread is already low (≤20) and the kernel body is simple (no shared memory, no reductions); the autotune overhead and register pressure from larger tiles will dominate.

> **Avoid when**: Do not apply autotune with large XBLOCK configs when: (1) the kernel has very low register usage (≤20 reg/thread) indicating the compiler is already tightly optimized; (2) the total problem size is small (xnumel ≤ 200K) so even the baseline grid is not severely underutilized in absolute terms; (3) the kernel uses in-place writes (restore_value adds overhead); or (4) the instruction count is already minimal — larger tiles may cause the compiler to unroll loops and explode instruction count.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_honor_autotune_xblock | 0.994× | 0.992× | ✓ |  |
| B | replace_int_reciprocal_with_rsqrt_for_fma_fusion | 1.011× | 1.100× | ✓ |  |
| C | restore_small_xblock_reduce_register_pressure | 0.949× | 0.941× | ✓ |  |
| D | reduce_register_pressure_via_arithmetic_simplification | 1.030× | 1.048× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | replace_int_recip... | Δ | restore_small_xbl... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3550 ns | +30 ns (+0.9%) | 3200 ns | -320 ns (-9.1%) | 3740 ns | +220 ns (+6.2%) | 3360 ns | -160 ns (-4.5%) |
| SM Throughput (%) | 16.34 % | 19.20 % | +2.86 pp | 18.81 % | +2.47 pp | 13.58 % | -2.76 pp | 12.29 % | -4.05 pp |
| Memory Throughput (%) | 6.81 % | 6.51 % | -0.30 pp | 11.09 % | +4.28 pp | 6.32 % | -0.49 pp | 7.18 % | +0.37 pp |
| DRAM Throughput (%) | 4.73 % | 4.68 % | -0.05 pp | 5.17 % | +0.44 pp | 4.46 % | -0.27 pp | 4.96 % | +0.23 pp |
| Achieved Occupancy (%) | 57.18 % | 61.32 % | +4.14 pp | 52.94 % | -4.24 pp | 30.24 % | -26.94 pp | 53.56 % | -3.62 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 18.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 53.51 % | 50.07 % | -3.44 pp | 41.91 % | -11.60 pp | 64.78 % | +11.27 pp | 68.91 % | +15.40 pp |
| IPC Active | 1.78 inst/cycle | 1.76 inst/cycle | -0.02 | 2.16 inst/cycle | +0.38 | 1.46 inst/cycle | -0.32 | 1.15 inst/cycle | -0.63 |


### Key Learnings

#### ✅ Plan B: `replace_int_reciprocal_with_rsqrt_for_fma_fusion` (1.100×)

When a batch-norm normalization sequence uses an integer constant (tl.full([1], 1, tl.int32)) divided by a float sqrt result, the mixed int/float types block the compiler's FMA fusion pass, producing a high ratio of non-fused FP32 instructions (NCU FPInstructions rule). Replacing the sqrt + integer-reciprocal + redundant float multiply chain with a single libdevice.rsqrt() call eliminates the type mismatch, allows the compiler to fuse (x-mean)*inv_std*weight+bias into FFMA chains, and halves total executed instructions (1,011K → 480K) yielding a ~15% duration reduction (3.74 µs → 3.20 µs). Branch divergence also drops to zero as predicated-off threads are eliminated by the cleaner compute path.

> **Applicable when**: NCU FPInstructions rule fires with a high ratio of non-fused to fused FP32 instructions (e.g., >3:1 non-fused:fused). The kernel contains a reciprocal-sqrt pattern implemented via integer constant division (tl.full([1], 1, tl.int32) / sqrt_result) or a redundant multiply-by-1.0. The compute chain is a multiply-add sequence (BN normalization: (x-mean)*inv_std*weight+bias) that would benefit from FFMA fusion.

> **Avoid when**: The kernel is already using libdevice.rsqrt() or pure float32 reciprocal. The bottleneck is memory bandwidth rather than instruction throughput (DRAM Throughput near 100%). The kernel is so short that the overhead of libdevice function call exceeds savings.

#### ✅ Plan D: `reduce_register_pressure_via_arithmetic_simplification` (1.048×)

Replacing redundant arithmetic patterns (integer-divide-by-float, multiply-by-1.0 no-ops, tl.sqrt_rn+divide instead of tl.rsqrt, int32 zero for ReLU) with clean equivalents (tl.rsqrt, float literal 0.0) reduced registers-per-thread from 32 to 18 and executed instructions by ~3.1x (1.01M to 326K). This lowered Block Limit Registers from 16 to 21 blocks/SM and improved average active threads per warp from 29.95 to 32.0 (eliminating predicated-off threads from type-cast divergence), yielding a 10.7% duration reduction (3.74 µs → 3.36 µs). When NCU reports FPInstructions with non-fused FP32 instructions and Avg. Active Threads Per Warp below 32, inspecting the kernel body for redundant type-conversion arithmetic (int/float ops, multiply-by-one, int-zero ReLU) and replacing with native float ops is a reliable low-risk optimization.

> **Applicable when**: NCU FPInstructions rule fires with significant non-fused FP32 instructions; Avg. Active Threads Per Warp is below 32 (indicating predicated-off threads from type-cast divergence); registers-per-thread is elevated relative to the logical complexity of the kernel; the kernel body contains tl.full([1], 1, tl.int32) or tl.full([1], 0, tl.int32) used in arithmetic with float tensors, or tl.sqrt_rn followed by division.

> **Avoid when**: The arithmetic differences are intentional (e.g., integer division semantics differ from float rsqrt for non-positive inputs); the kernel already uses native float ops and tl.rsqrt; register pressure is not a bottleneck (Block Limit Registers is not the binding occupancy constraint).

---

## Iteration 3

**Best so far**: 1.0476× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_honor_autotune_xblock_and_replace_int_relu_with_float | 0.962× | 0.983× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency | 1.001× | 0.992× | ✓ |  |
| C | fuse_fp32_chain_via_rsqrt_and_float_relu_zero | 1.028× | 1.038× | ✓ |  |
| D | increase_num_warps_and_rsqrt_fma_fusion | 1.028× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | increase_num_warp... | Δ | fuse_fp32_chain_v... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3580 ns | +60 ns (+1.7%) | 3550 ns | +30 ns (+0.9%) | 3390 ns | -130 ns (-3.7%) | 3390 ns | -130 ns (-3.7%) |
| SM Throughput (%) | 16.34 % | 19.33 % | +2.99 pp | 19.08 % | +2.74 pp | 12.63 % | -3.71 pp | 11.84 % | -4.50 pp |
| Memory Throughput (%) | 6.81 % | 6.63 % | -0.18 pp | 6.67 % | -0.14 pp | 7.20 % | +0.39 pp | 6.98 % | +0.17 pp |
| DRAM Throughput (%) | 4.73 % | 4.67 % | -0.06 pp | 4.70 % | -0.03 pp | 4.93 % | +0.20 pp | 4.93 % | +0.20 pp |
| Achieved Occupancy (%) | 57.18 % | 53.38 % | -3.80 pp | 55.46 % | -1.72 pp | 50.47 % | -6.71 pp | 58.24 % | +1.06 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 53.51 % | 47.95 % | -5.56 pp | 48.20 % | -5.31 pp | 65.22 % | +11.71 pp | 67.88 % | +14.37 pp |
| IPC Active | 1.78 inst/cycle | 1.85 inst/cycle | +0.07 | 2.00 inst/cycle | +0.22 | 1.27 inst/cycle | -0.51 | 1.39 inst/cycle | -0.39 |

---

## Iteration 4

**Best so far**: 1.0476× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_use_dynamic_xblock_via_lambda | 0.997× | 0.992× | ✓ |  |
| B | fuse_relu_via_float_zero_to_eliminate_nonfused_fp32_instructions | 1.011× | 0.992× | ✓ |  |
| C | eliminate_all_int_float_type_mismatches_for_full_fma_chain_fusion | 1.027× | 1.057× | ✓ |  |
| D | streaming_eviction_policy_for_main_tensor_plus_fma_cleanup | 1.048× | 1.038× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.014× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_use_d... | Δ | fuse_relu_via_flo... | Δ | eliminate_all_int... | Δ | streaming_evictio... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3550 ns | +30 ns (+0.9%) | 3550 ns | +30 ns (+0.9%) | 3330 ns | -190 ns (-5.4%) | 3390 ns | -130 ns (-3.7%) | 3580 ns | +60 ns (+1.7%) |
| SM Throughput (%) | 16.34 % | 19.18 % | +2.84 pp | 19.04 % | +2.70 pp | 11.54 % | -4.80 pp | 13.68 % | -2.66 pp | 19.16 % | +2.82 pp |
| Memory Throughput (%) | 6.81 % | 6.53 % | -0.28 pp | 6.55 % | -0.26 pp | 6.92 % | +0.11 pp | 7.24 % | +0.43 pp | 6.53 % | -0.28 pp |
| DRAM Throughput (%) | 4.73 % | 4.70 % | -0.03 pp | 4.70 % | -0.03 pp | 4.96 % | +0.23 pp | 4.91 % | +0.18 pp | 4.68 % | -0.05 pp |
| Achieved Occupancy (%) | 57.18 % | 53.27 % | -3.91 pp | 52.32 % | -4.86 pp | 48.52 % | -8.66 pp | 49.79 % | -7.39 pp | 47.64 % | -9.54 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 53.51 % | 53.98 % | +0.47 pp | 45.74 % | -7.77 pp | 62.48 % | +8.97 pp | 67.14 % | +13.63 pp | 49.30 % | -4.21 pp |
| IPC Active | 1.78 inst/cycle | 1.84 inst/cycle | +0.06 | 1.88 inst/cycle | +0.10 | 1.28 inst/cycle | -0.50 | 1.35 inst/cycle | -0.43 | 1.53 inst/cycle | -0.25 |

---

## Iteration 5

**Best so far**: 1.0476× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | 2d_grid_launch_to_fix_l2_workload_imbalance | 0.964× | 0.972× | ✓ |  |
| B | annotate_broadcast_loads_with_contiguity_hints_to_reduce_l1tex_stalls | 0.992× | 0.992× | ✓ |  |
| C | grid_swizzle_for_l2_slice_balance | 0.979× | 1.029× | ✓ |  |
| D | increase_block_size_to_256_threads_for_latency_hiding | 0.998× | 0.903× | ✓ |  |


### Metric Comparison

| Metric | Baseline | 2d_grid_launch_to... | Δ | annotate_broadcas... | Δ | grid_swizzle_for_... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3620 ns | +100 ns (+2.8%) | 3550 ns | +30 ns (+0.9%) | 3420 ns | -100 ns (-2.8%) | 3900 ns | +380 ns (+10.8%) |
| SM Throughput (%) | 16.34 % | 19.30 % | +2.96 pp | 20.38 % | +4.04 pp | 13.68 % | -2.66 pp | 17.80 % | +1.46 pp |
| Memory Throughput (%) | 6.81 % | 6.42 % | -0.39 pp | 6.58 % | -0.23 pp | 6.95 % | +0.14 pp | 6.10 % | -0.71 pp |
| DRAM Throughput (%) | 4.73 % | 4.61 % | -0.12 pp | 4.66 % | -0.07 pp | 4.88 % | +0.15 pp | 4.27 % | -0.46 pp |
| Achieved Occupancy (%) | 57.18 % | 52.86 % | -4.32 pp | 53.74 % | -3.44 pp | 50.62 % | -6.56 pp | 53.39 % | -3.79 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 53.51 % | 47.46 % | -6.05 pp | 45.59 % | -7.92 pp | 61.01 % | +7.50 pp | 46.24 % | -7.27 pp |
| IPC Active | 1.78 inst/cycle | 1.94 inst/cycle | +0.16 | 1.93 inst/cycle | +0.15 | 1.43 inst/cycle | -0.35 | 1.91 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 2     3360 ns  (1.048×)  — reduce_register_pressure_via_arithmetic_simplification

Final best: 3390 ns  (1.0383×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 200704
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
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
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
