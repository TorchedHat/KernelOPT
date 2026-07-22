# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13790 ns — **Best latency**: 13660 ns (**1.0095×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_bn_arithmetic_into_fma_and_add_missing_hardtanh_clamp`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18000,1,1), Waves/SM: 8.52
- **Registers/thread**: 18
- **Duration**: 13790 ns
- **SM Throughput**: 57.25 % | **Memory**: 21.28 % | **DRAM**: 13.64 %
- **Achieved Occupancy**: 59.36 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 32.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_persistent_tile_scheduling | 0.772× | 0.743× | ✓ |  |
| B | increase_xblock_with_autotune_for_pointwise_kernel | 0.778× | 0.741× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_latency_hiding | 0.782× | 0.743× | ✓ |  |
| D | increase_xblock_and_add_autotune_for_vectorized_loads | 0.792× | 0.755× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13790 ns | 18560 ns | +4770 ns (+34.6%) | 18620 ns | +4830 ns (+35.0%) | 18560 ns | +4770 ns (+34.6%) | 18270 ns | +4480 ns (+32.5%) |
| SM Throughput (%) | 57.25 % | 77.51 % | +20.26 pp | 77.03 % | +19.78 pp | 77.56 % | +20.31 pp | 77.47 % | +20.22 pp |
| Memory Throughput (%) | 21.28 % | 23.30 % | +2.02 pp | 22.93 % | +1.65 pp | 23.13 % | +1.85 pp | 23.68 % | +2.40 pp |
| DRAM Throughput (%) | 13.64 % | 10.14 % | -3.50 pp | 10.09 % | -3.55 pp | 10.12 % | -3.52 pp | 10.29 % | -3.35 pp |
| Achieved Occupancy (%) | 59.36 % | 85.00 % | +25.64 pp | 85.27 % | +25.91 pp | 85.38 % | +26.02 pp | 84.76 % | +25.40 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 32.42 % | 13.22 % | -19.20 pp | 13.02 % | -19.40 pp | 13.19 % | -19.23 pp | 13.82 % | -18.60 pp |
| IPC Active | 2.76 inst/cycle | 3.43 inst/cycle | +0.67 | 3.46 inst/cycle | +0.70 | 3.46 inst/cycle | +0.70 | 3.46 inst/cycle | +0.70 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_for_persistent_tile_scheduling` (0.743×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) caused the autotuner to select a configuration that nearly doubled registers per thread (18→32), reducing Block Limit Registers from 21 to 16 blocks/SM and increasing elapsed cycles by 35% (20493→27611 cycles). Although achieved occupancy improved (59%→85%) and eligible warps per scheduler rose (1.72→7.02), the instruction count nearly doubled (6.12M→11.16M), indicating the selected config introduced redundant computation or unrolling overhead that outweighed the latency-hiding benefit. The net result was a 34% slowdown despite better occupancy metrics.

> **Applicable when**: Never apply this exact strategy when the kernel has low register count (≤20 registers/thread) and the autotune search space includes very large XBLOCK values that may trigger aggressive unrolling and register spilling.

> **Avoid when**: Kernel already has low register usage (≤20 reg/thread) and low instruction count; autotuning with large XBLOCK configs risks doubling register pressure and instruction count even if occupancy improves. Also avoid when the kernel's instruction count is sensitive to XBLOCK-driven loop unrolling.

#### ❌ Plan B: `increase_xblock_with_autotune_for_pointwise_kernel` (0.741×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) improved achieved occupancy (59% → 85%) and eligible warps per scheduler (1.72 → 7.06), but caused a 35% regression in kernel duration (13.79 µs → 18.62 µs). The root cause is that the selected autotune config doubled register usage per thread (18 → 32 registers), which increased total instruction count by ~84% (6.12M → 11.25M executed instructions) and raised warp cycles per issued instruction (13.74 → 15.84). When a pointwise kernel is already compute-bound at small tile sizes and the register file is a limiting resource, forcing larger tiles via autotune can select configs that spill or increase instruction count enough to outweigh any scheduling benefit.

> **Applicable when**: Never apply this specific pattern when: registers_per_thread is already near the SM register limit and increasing XBLOCK causes the compiler to allocate significantly more registers per thread. Specifically, avoid when baseline registers_per_thread <= 20 and the autotune search space includes XBLOCK >= 512 without constraining num_warps to keep register pressure low.

> **Avoid when**: Kernel is a simple pointwise op with low register usage (<=20 regs/thread) at baseline; increasing XBLOCK causes register count to jump (e.g., 18 → 32), which inflates instruction count and warp cycles per instruction even if occupancy improves. Also avoid when the kernel is already achieving reasonable throughput (>50% compute SM) and the bottleneck is L1TEX stalls rather than pure scheduling starvation.

#### ⚠️ Plan C: `autotune_xblock_and_num_warps_for_latency_hiding` (0.743×)

Adding @triton.autotune with larger XBLOCK (512–2048) and more warps (8–16) improved achieved occupancy from 59% to 85% and eligible warps per scheduler from 1.72 to 6.99, but caused a 34% slowdown. The root cause is that registers per thread nearly doubled (18→32), indicating register pressure increased — likely from the compiler needing to keep more live values across the larger tile, or from autotune overhead itself. Despite better scheduling metrics, the 84% increase in executed instructions (6.12M→11.25M) and 15% higher warp cycles per issued instruction (13.74→15.77) show the larger tile introduced more work and did not reduce the dominant stall. When the baseline kernel uses very few registers (≤20/thread) and the bottleneck is L1TEX stall, increasing XBLOCK without verifying that register count stays flat can cause register spill or compiler-induced instruction bloat that overwhelms the latency-hiding benefit.

> **Applicable when**: Kernel has low eligible warps per scheduler (<2.0), high CPIStall fraction (>40%), and low achieved occupancy (<65%), suggesting latency-hiding via more warps could help.

> **Avoid when**: Baseline kernel already uses very few registers (≤20/thread) and the autotune configs cause register count to increase significantly (>25% jump). Also avoid when the kernel body has many live values that scale with XBLOCK, as larger tiles can cause compiler register spill or instruction count bloat that negates the scheduling benefit. Additionally, do not use a static grid that ignores META['XBLOCK'] — the grid must be a lambda consuming META to match the autotuned block size.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_fma_fusion | 1.001× | 1.004× | ✓ |  |
| B | replace_int_div_sqrt_with_rsqrt_to_enable_fma_fusion | 0.938× | 0.966× | ✓ |  |
| C | fuse_fp32_instructions_via_arithmetic_rewrite_and_restore_small_xblock | 1.009× | 1.022× | ✓ | **✓ best** |
| D | fuse_fp32_instructions_via_rsqrt_and_fma | 1.002× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | replace_int_div_s... | Δ | fuse_fp32_instruc... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13790 ns | 13730 ns | -60 ns (-0.4%) | 14270 ns | +480 ns (+3.5%) | 13500 ns | -290 ns (-2.1%) | 13630 ns | -160 ns (-1.2%) |
| SM Throughput (%) | 57.25 % | 57.20 % | -0.05 pp | 45.88 % | -11.37 pp | 34.43 % | -22.82 pp | 34.52 % | -22.73 pp |
| Memory Throughput (%) | 21.28 % | 21.28 % | +0.00 pp | 31.52 % | +10.24 pp | 21.38 % | +0.10 pp | 21.43 % | +0.15 pp |
| DRAM Throughput (%) | 13.64 % | 13.70 % | +0.06 pp | 13.18 % | -0.46 pp | 13.93 % | +0.29 pp | 13.80 % | +0.16 pp |
| Achieved Occupancy (%) | 59.36 % | 59.54 % | +0.18 pp | 38.55 % | -20.81 pp | 42.51 % | -16.85 pp | 42.49 % | -16.87 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 32.42 % | 30.96 % | -1.46 pp | 42.87 % | +10.45 pp | 57.70 % | +25.28 pp | 57.97 % | +25.55 pp |
| IPC Active | 2.76 inst/cycle | 2.74 inst/cycle | -0.02 | 2.29 inst/cycle | -0.47 | 1.67 inst/cycle | -1.09 | 1.69 inst/cycle | -1.07 |

---

## Iteration 3

**Best so far**: 1.0215× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_tensor_to_reduce_l1tex_stalls | 1.013× | 1.010× | ✓ |  |
| B | increase_num_warps_with_fixed_xblock_for_l1tex_latency_hiding | 1.015× | 1.024× | ✓ | **✓ best** |
| C | persistent_tile_scheduling_for_bn_pointwise_kernel | 0.985× | 0.989× | ✓ |  |
| D | fix_autotune_grid_lambda_and_add_small_xblock_configs_with_streaming_eviction | 1.003× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | increase_num_warp... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13790 ns | 13660 ns | -130 ns (-0.9%) | 13470 ns | -320 ns (-2.3%) | 13950 ns | +160 ns (+1.2%) | 13630 ns | -160 ns (-1.2%) |
| SM Throughput (%) | 57.25 % | 37.88 % | -19.37 pp | 34.51 % | -22.74 pp | 64.87 % | +7.62 pp | 37.20 % | -20.05 pp |
| Memory Throughput (%) | 21.28 % | 21.38 % | +0.10 pp | 21.37 % | +0.09 pp | 20.95 % | -0.33 pp | 21.39 % | +0.11 pp |
| DRAM Throughput (%) | 13.64 % | 13.77 % | +0.13 pp | 13.95 % | +0.31 pp | 13.49 % | -0.15 pp | 13.80 % | +0.16 pp |
| Achieved Occupancy (%) | 59.36 % | 44.83 % | -14.53 pp | 42.24 % | -17.12 pp | 73.05 % | +13.69 pp | 43.22 % | -16.14 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 32.42 % | 53.68 % | +21.26 pp | 57.81 % | +25.39 pp | 22.83 % | -9.59 pp | 54.49 % | +22.07 pp |
| IPC Active | 2.76 inst/cycle | 1.83 inst/cycle | -0.93 | 1.66 inst/cycle | -1.10 | 3.01 inst/cycle | +0.25 | 1.82 inst/cycle | -0.94 |

---

## Iteration 4

**Best so far**: 1.0238× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_broadcast_arrays_into_registers_to_eliminate_l1tex_stalls | 0.996× | 0.993× | ✓ |  |
| B | fuse_bn_arithmetic_into_fma_and_add_missing_hardtanh_clamp | 1.016× | 1.010× | ✓ | **✓ best** |
| C | restructure_channel_index_and_add_missing_hardtanh_clamp_for_fma_fusion | 1.016× | 1.022× | ✓ |  |
| D | fuse_sqrt_reciprocal_into_rsqrt_for_fp32_instruction_reduction | 0.997× | 1.014× | ✓ |  |
| E | add_triton_autotune | 1.002× | 1.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_broadcast... | Δ | fuse_bn_arithmeti... | Δ | restructure_chann... | Δ | fuse_sqrt_recipro... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13790 ns | 13890 ns | +100 ns (+0.7%) | 13660 ns | -130 ns (-0.9%) | 13500 ns | -290 ns (-2.1%) | 13600 ns | -190 ns (-1.4%) | 13470 ns | -320 ns (-2.3%) |
| SM Throughput (%) | 57.25 % | 74.45 % | +17.20 pp | 39.22 % | -18.03 pp | 39.17 % | -18.08 pp | 43.46 % | -13.79 pp | 34.34 % | -22.91 pp |
| Memory Throughput (%) | 21.28 % | 50.51 % | +29.23 pp | 21.24 % | -0.04 pp | 21.10 % | -0.18 pp | 21.27 % | -0.01 pp | 21.35 % | +0.07 pp |
| DRAM Throughput (%) | 13.64 % | 13.55 % | -0.09 pp | 13.77 % | +0.13 pp | 13.91 % | +0.27 pp | 13.82 % | +0.18 pp | 13.94 % | +0.30 pp |
| Achieved Occupancy (%) | 59.36 % | 55.78 % | -3.58 pp | 43.81 % | -15.55 pp | 44.86 % | -14.50 pp | 48.24 % | -11.12 pp | 42.69 % | -16.67 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 17.00 register/thread | -1 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 32.42 % | 53.31 % | +20.89 pp | 52.66 % | +20.24 pp | 52.22 % | +19.80 pp | 49.12 % | +16.70 pp | 57.79 % | +25.37 pp |
| IPC Active | 2.76 inst/cycle | 1.87 inst/cycle | -0.89 | 1.92 inst/cycle | -0.84 | 1.87 inst/cycle | -0.89 | 2.01 inst/cycle | -0.75 | 1.69 inst/cycle | -1.07 |

---

## Iteration 5

**Best so far**: 1.0238× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_and_explore_small_xblock_with_more_warps | 0.837× | 1.028× | ✓ |  |
| B | increase_num_warps_fixed_xblock_for_l1tex_latency_hiding | 1.012× | 1.018× | ✓ |  |
| C | reduce_xblock_to_increase_sm_parallelism_via_register_relief | 0.987× | 0.986× | ✓ |  |
| D | persistent_tile_scheduling_inner_loop_per_sm | 0.999× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | increase_num_warp... | Δ | reduce_xblock_to_... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13790 ns | 13410 ns | -380 ns (-2.8%) | 13540 ns | -250 ns (-1.8%) | 13980 ns | +190 ns (+1.4%) | 13570 ns | -220 ns (-1.6%) |
| SM Throughput (%) | 57.25 % | 34.11 % | -23.14 pp | 39.21 % | -18.04 pp | 33.43 % | -23.82 pp | 46.61 % | -10.64 pp |
| Memory Throughput (%) | 21.28 % | 17.98 % | -3.30 pp | 21.13 % | -0.15 pp | 18.99 % | -2.29 pp | 21.22 % | -0.06 pp |
| DRAM Throughput (%) | 13.64 % | 3.51 % | -10.13 pp | 13.89 % | +0.25 pp | 13.44 % | -0.20 pp | 13.88 % | +0.24 pp |
| Achieved Occupancy (%) | 59.36 % | 42.00 % | -17.36 pp | 45.95 % | -13.41 pp | 44.37 % | -14.99 pp | 48.97 % | -10.39 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 28.00 register/thread | +10 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 4.26 | -4.26 | 8.52 | +0.00 |
| No Eligible (%) | 32.42 % | 58.49 % | +26.07 pp | 52.26 % | +19.84 pp | 57.24 % | +24.82 pp | 43.33 % | +10.91 pp |
| IPC Active | 2.76 inst/cycle | 1.68 inst/cycle | -1.08 | 1.90 inst/cycle | -0.86 | 1.71 inst/cycle | -1.05 | 2.22 inst/cycle | -0.54 |

---

## Cumulative Progress

```
Baseline:  13790 ns  (1.000×)
Iter 2     13500 ns  (1.022×)  — fuse_fp32_instructions_via_arithmetic_rewrite_and_restore_small_xblock
Iter 3     13470 ns  (1.024×)  — increase_num_warps_with_fixed_xblock_for_l1tex_latency_hiding

Final best: 13660 ns  (1.0095×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2304000
@@ -21,16 +32,11 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -40,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 2304000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2304000, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2304000)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
