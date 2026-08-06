# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8160 ns — **Best latency**: 7870 ns (**1.0368×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_broken_autotune_grid_and_add_small_xblock_configs_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8820,1,1), Waves/SM: 4.18
- **Registers/thread**: 18
- **Duration**: 8160 ns
- **SM Throughput**: 47.80 % | **Memory**: 18.99 % | **DRAM**: 11.31 %
- **Achieved Occupancy**: 61.74 %
- **IPC Active**: 2.65 inst/cycle | **No Eligible**: 33.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.826× | 0.765× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy_and_latency_hiding | 0.818× | 0.765× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_occupancy | 0.949× | 0.944× | ✓ |  |
| D | autotune_xblock_size_to_reduce_partial_waves_and_l1tex_stalls | 0.990× | 0.765× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8160 ns | 10660 ns | +2500 ns (+30.6%) | 10660 ns | +2500 ns (+30.6%) | 8640 ns | +480 ns (+5.9%) | 10660 ns | +2500 ns (+30.6%) |
| SM Throughput (%) | 47.80 % | 66.07 % | +18.27 pp | 65.61 % | +17.81 pp | 58.05 % | +10.25 pp | 66.39 % | +18.59 pp |
| Memory Throughput (%) | 18.99 % | 19.85 % | +0.86 pp | 19.70 % | +0.71 pp | 22.74 % | +3.75 pp | 20.01 % | +1.02 pp |
| DRAM Throughput (%) | 11.31 % | 8.66 % | -2.65 pp | 8.66 % | -2.65 pp | 10.69 % | -0.62 pp | 8.68 % | -2.63 pp |
| Achieved Occupancy (%) | 61.74 % | 82.22 % | +20.48 pp | 83.24 % | +21.50 pp | 73.57 % | +11.83 pp | 82.23 % | +20.49 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.58 % | 18.35 % | -15.23 pp | 18.39 % | -15.19 pp | 21.68 % | -11.90 pp | 18.80 % | -14.78 pp |
| IPC Active | 2.65 inst/cycle | 3.21 inst/cycle | +0.56 | 3.21 inst/cycle | +0.56 | 3.21 inst/cycle | +0.56 | 3.21 inst/cycle | +0.56 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.765×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) increased registers per thread from 18 to 32, nearly doubling total executed instructions (3.03M → 5.55M) and increasing kernel duration from 8.16 µs to 10.66 µs (0.77x slowdown). Although achieved occupancy improved (61.7% → 82.2%) and eligible warps per scheduler increased (1.97 → 6.35), the instruction count explosion dominated, indicating the autotuner selected a config that caused significant loop unrolling or register spilling overhead. When a kernel already has low register pressure (≤20 registers/thread) and the bottleneck is L1TEX stall latency rather than occupancy, increasing tile size via autotune can backfire by inflating instruction count without proportional throughput gain.

> **Applicable when**: Never — this is a negative result. Do not apply autotune with large XBLOCK configs when: registers_per_thread ≤ 20, the kernel is purely pointwise with no shared memory, and the primary bottleneck is L1TEX stall latency (CPIStall rule, ~40% of cycles). Occupancy gap alone is insufficient justification.

> **Avoid when**: Kernel has low register pressure (≤20 registers/thread); kernel is purely pointwise (no reduction, no shared memory); L1TEX stall is the dominant bottleneck; increasing XBLOCK causes register count to jump (e.g. 18→32), which inflates instruction count without proportional memory access reduction.

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_occupancy_and_latency_hiding` (0.765×)

Adding triton.autotune with larger XBLOCK configs (512–2048) and more warps (4–16) improved achieved occupancy (61.7%→83.2%) and compute throughput (47.8%→65.6%), but nearly doubled total executed instructions (3.03M→5.55M) and increased registers per thread (18→32), causing a 31% slowdown. The instruction explosion likely stems from the autotuner selecting a config that unrolls more work per thread, generating more non-fused FP32 instructions and increasing per-warp register pressure. Despite better occupancy, the higher instruction count and longer warp cycles per instruction (14.66→16.11) resulted in a net 30% longer kernel duration (8.16→10.66 µs). For short-duration pointwise kernels already at 4+ waves, adding autotune overhead and larger XBLOCK can backfire if the selected config dramatically increases instruction count.

> **Applicable when**: Never — this specific approach caused regression. Autotune with large XBLOCK configs may be worth trying only if: (1) registers per thread are very low (<16) and won't increase with unrolling, (2) the kernel is purely memory-bound (not compute-bound), and (3) instruction count is verified not to explode with larger tile sizes.

> **Avoid when**: Kernel already has moderate occupancy (>60%) and is partially compute-bound (Compute SM Throughput >40%). Adding autotune with XBLOCK 4–16x larger than baseline causes instruction count to nearly double (due to increased unrolling of non-fused FP32 ops) and register pressure to jump (18→32 registers/thread), negating the occupancy gains. Avoid when the kernel duration is already short (<10 µs) and the autotune overhead itself may dominate.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_non_fused_fp32_instructions_via_rsqrt_algebraic_simplification | 1.012× | 0.996× | ✓ |  |
| B | reduce_non_fused_fp32_instructions_via_algebraic_simplification_and_restore_baseline_xblock | 1.009× | 1.016× | ✓ |  |
| C | reduce_non_fused_fp32_instructions_via_arithmetic_simplification | 1.026× | 1.016× | ✓ | **✓ best** |
| D | replace_sqrt_reciprocal_with_rsqrt_to_fuse_fp32_instructions | 1.007× | 0.951× | ✓ |  |
| E | add_triton_autotune | 1.008× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_non_fused_... | Δ | reduce_non_fused_... | Δ | reduce_non_fused_... | Δ | replace_sqrt_reci... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8160 ns | 8190 ns | +30 ns (+0.4%) | 8030 ns | -130 ns (-1.6%) | 8030 ns | -130 ns (-1.6%) | 8580 ns | +420 ns (+5.1%) | 8160 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.80 % | 32.31 % | -15.49 pp | 28.19 % | -19.61 pp | 28.73 % | -19.07 pp | 37.66 % | -10.14 pp | 47.08 % | -0.72 pp |
| Memory Throughput (%) | 18.99 % | 25.18 % | +6.19 pp | 18.97 % | -0.02 pp | 18.85 % | -0.14 pp | 26.49 % | +7.50 pp | 18.68 % | -0.31 pp |
| DRAM Throughput (%) | 11.31 % | 11.28 % | -0.03 pp | 11.48 % | +0.17 pp | 11.52 % | +0.21 pp | 10.78 % | -0.53 pp | 11.32 % | +0.01 pp |
| Achieved Occupancy (%) | 61.74 % | 41.87 % | -19.87 pp | 47.96 % | -13.78 pp | 47.26 % | -14.48 pp | 46.68 % | -15.06 pp | 60.53 % | -1.21 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.58 % | 49.01 % | +15.43 pp | 58.27 % | +24.69 pp | 59.63 % | +26.05 pp | 43.84 % | +10.26 pp | 33.91 % | +0.33 pp |
| IPC Active | 2.65 inst/cycle | 2.01 inst/cycle | -0.64 | 1.59 inst/cycle | -1.06 | 1.65 inst/cycle | -1.00 | 2.19 inst/cycle | -0.46 | 2.60 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0162× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | apply_evict_first_to_streaming_tensor_to_protect_bn_parameter_cache_residency | 1.024× | 1.033× | ✓ |  |
| B | persistent_kernel_to_eliminate_partial_wave_and_improve_l1tex_latency_hiding | 0.307× | 0.251× | ✓ |  |
| C | add_evict_first_to_streaming_tensor_load_to_reduce_l1tex_stalls | 0.987× | 0.996× | ✓ |  |
| D | fix_broken_autotune_grid_and_add_small_xblock_configs_for_occupancy | 1.028× | 1.037× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | apply_evict_first... | Δ | persistent_kernel... | Δ | add_evict_first_t... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8160 ns | 7900 ns | -260 ns (-3.2%) | 32510 ns | +24350 ns (+298.4%) | 8190 ns | +30 ns (+0.4%) | 7870 ns | -290 ns (-3.6%) |
| SM Throughput (%) | 47.80 % | 31.42 % | -16.38 pp | 6.58 % | -41.22 pp | 49.68 % | +1.88 pp | 14.75 % | -33.05 pp |
| Memory Throughput (%) | 18.99 % | 19.06 % | +0.07 pp | 3.90 % | -15.09 pp | 18.66 % | -0.33 pp | 9.43 % | -9.56 pp |
| DRAM Throughput (%) | 11.31 % | 11.69 % | +0.38 pp | 2.84 % | -8.47 pp | 11.31 % | +0.00 pp | 5.88 % | -5.43 pp |
| Achieved Occupancy (%) | 61.74 % | 47.93 % | -13.81 pp | 8.13 % | -53.61 pp | 63.81 % | +2.07 pp | 34.80 % | -26.94 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 2.09 | -2.09 |
| No Eligible (%) | 33.58 % | 54.80 % | +21.22 pp | 92.18 % | +58.60 pp | 29.73 % | -3.85 pp | 74.00 % | +40.42 pp |
| IPC Active | 2.65 inst/cycle | 1.82 inst/cycle | -0.83 | 0.31 inst/cycle | -2.34 | 2.68 inst/cycle | +0.03 | 0.99 inst/cycle | -1.66 |

---

## Iteration 4

**Best so far**: 1.0368× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | grid_align_to_eliminate_partial_wave_and_increase_warps_per_sm | 1.012× | 1.033× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.008× | 1.016× | ✓ |  |
| C | enable_l2_compression_on_output_store_to_reduce_dram_traffic | 1.006× | 1.000× | ✓ |  |
| D | fix_persistent_kernel_grid_to_activate_tile_loop_and_increase_num_warps | 0.309× | 0.255× | ✓ |  |


### Metric Comparison

| Metric | Baseline | grid_align_to_eli... | Δ | increase_num_warp... | Δ | enable_l2_compres... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8160 ns | 7900 ns | -260 ns (-3.2%) | 8030 ns | -130 ns (-1.6%) | 8160 ns | +0 ns (+0.0%) | 32000 ns | +23840 ns (+292.2%) |
| SM Throughput (%) | 47.80 % | 29.36 % | -18.44 pp | 31.36 % | -16.44 pp | 50.49 % | +2.69 pp | 6.57 % | -41.23 pp |
| Memory Throughput (%) | 18.99 % | 19.24 % | +0.25 pp | 18.98 % | -0.01 pp | 18.96 % | -0.03 pp | 3.91 % | -15.08 pp |
| DRAM Throughput (%) | 11.31 % | 11.65 % | +0.34 pp | 11.51 % | +0.20 pp | 11.35 % | +0.04 pp | 2.88 % | -8.43 pp |
| Achieved Occupancy (%) | 61.74 % | 46.33 % | -15.41 pp | 47.65 % | -14.09 pp | 63.81 % | +2.07 pp | 8.14 % | -53.60 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.58 % | 58.33 % | +24.75 pp | 55.90 % | +22.32 pp | 29.69 % | -3.89 pp | 92.19 % | +58.61 pp |
| IPC Active | 2.65 inst/cycle | 1.66 inst/cycle | -0.99 | 1.78 inst/cycle | -0.87 | 2.67 inst/cycle | +0.02 | 0.31 inst/cycle | -2.34 |

---

## Iteration 5

**Best so far**: 1.0368× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_bn_scale_offset_to_fuse_fp32_instructions_via_fma | 1.014× | 1.016× | ✓ |  |
| B | align_xblock_to_channel_group_size_for_bn_parameter_reuse | 1.026× | 1.033× | ✓ |  |
| C | replace_sqrt_div_with_libdevice_rsqrt_to_fuse_fp32_instructions | 1.006× | 1.024× | ✓ |  |
| D | revert_broken_persistent_kernel_to_flat_kernel_restore_register_budget | 1.027× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_bn_scale_of... | Δ | align_xblock_to_c... | Δ | replace_sqrt_div_... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8160 ns | 8030 ns | -130 ns (-1.6%) | 7900 ns | -260 ns (-3.2%) | 7970 ns | -190 ns (-2.3%) | 8130 ns | -30 ns (-0.4%) |
| SM Throughput (%) | 47.80 % | 28.49 % | -19.31 pp | 31.57 % | -16.23 pp | 31.88 % | -15.92 pp | 28.86 % | -18.94 pp |
| Memory Throughput (%) | 18.99 % | 18.93 % | -0.06 pp | 19.23 % | +0.24 pp | 19.13 % | +0.14 pp | 18.97 % | -0.02 pp |
| DRAM Throughput (%) | 11.31 % | 11.49 % | +0.18 pp | 11.66 % | +0.35 pp | 11.60 % | +0.29 pp | 11.38 % | +0.07 pp |
| Achieved Occupancy (%) | 61.74 % | 46.72 % | -15.02 pp | 47.44 % | -14.30 pp | 49.46 % | -12.28 pp | 48.05 % | -13.69 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.58 % | 59.65 % | +26.07 pp | 55.70 % | +22.12 pp | 54.23 % | +20.65 pp | 57.76 % | +24.18 pp |
| IPC Active | 2.65 inst/cycle | 1.60 inst/cycle | -1.05 | 1.74 inst/cycle | -0.91 | 1.74 inst/cycle | -0.91 | 1.62 inst/cycle | -1.03 |

---

## Cumulative Progress

```
Baseline:  8160 ns  (1.000×)
Iter 2     8030 ns  (1.016×)  — reduce_non_fused_fp32_instructions_via_arithmetic_simplification
Iter 3     7870 ns  (1.037×)  — fix_broken_autotune_grid_and_add_small_xblock_configs_for_occupancy

Final best: 7870 ns  (1.0368×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1128960
@@ -21,12 +30,8 @@

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
     tmp16 = tl.full([1], 0.0, tl.float32)
@@ -34,8 +39,6 @@

     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 1128960
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
