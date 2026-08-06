# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17120 ns — **Best latency**: 16990 ns (**1.0077×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_modulo96_overhead_with_max_contiguous_hints_for_channel_broadcast_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_8_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 17
- **Duration**: 17120 ns
- **SM Throughput**: 51.44 % | **Memory**: 22.72 % | **DRAM**: 14.34 %
- **Achieved Occupancy**: 57.06 %
- **IPC Active**: 2.36 inst/cycle | **No Eligible**: 40.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.744× | 0.714× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.999× | 0.216× | ✓ |  |
| C | increase_xblock_vectorized_loads_autotune | 0.746× | 0.713× | ✓ |  |
| D | autotune_xblock_to_hide_l1tex_latency | 0.748× | 0.714× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_v... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 23970 ns | +6850 ns (+40.0%) | 79390 ns | +62270 ns (+363.7%) | 24000 ns | +6880 ns (+40.2%) | 23970 ns | +6850 ns (+40.0%) |
| SM Throughput (%) | 51.44 % | 78.84 % | +27.40 pp | 86.57 % | +35.13 pp | 78.86 % | +27.42 pp | 78.94 % | +27.50 pp |
| Memory Throughput (%) | 22.72 % | 23.45 % | +0.73 pp | 17.77 % | -4.95 pp | 23.41 % | +0.69 pp | 23.51 % | +0.79 pp |
| DRAM Throughput (%) | 14.34 % | 10.33 % | -4.01 pp | 3.09 % | -11.25 pp | 10.23 % | -4.11 pp | 10.25 % | -4.09 pp |
| Achieved Occupancy (%) | 57.06 % | 86.28 % | +29.22 pp | 85.91 % | +28.85 pp | 86.53 % | +29.47 pp | 86.34 % | +29.28 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 44.55 | +33.41 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.40 % | 13.73 % | -26.67 pp | 10.89 % | -29.51 pp | 14.00 % | -26.40 pp | 14.09 % | -26.31 pp |
| IPC Active | 2.36 inst/cycle | 3.43 inst/cycle | +1.07 | 3.56 inst/cycle | +1.20 | 3.41 inst/cycle | +1.05 | 3.44 inst/cycle | +1.08 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.714×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512–2048) caused the compiler to allocate significantly more registers per thread (17→32), which reduced the block limit from registers (21→16 blocks/SM) and more than doubled the total executed instruction count (6.87M→14.79M). Although occupancy improved (57%→86%) and no-eligible-warp cycles dropped (40.4%→13.7%), the 2.15× instruction count increase dominated, raising kernel duration from 17.12 µs to 23.97 µs. The autotune configs introduced additional predication overhead (xmask guards on all loads/stores) and likely caused the compiler to unroll loops more aggressively, inflating register pressure beyond what the original 17-register budget suggested was available.

> **Applicable when**: Never apply this specific pattern when the kernel already has low register usage (≤20 registers/thread) but the autotune configs cause register count to nearly double; the instruction count explosion outweighs latency-hiding benefits.

> **Avoid when**: Do not add large-XBLOCK autotune configs to elementwise pointwise kernels when: (1) the compiler responds by doubling register allocation, (2) the resulting instruction count more than doubles, or (3) the kernel is already achieving adequate throughput with small tiles. The xmask predication on per-channel broadcast loads (indexed by x0 = xindex % C) adds unnecessary overhead when xnumel is already a multiple of the original XBLOCK.

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_latency_hiding` (0.216×)

Increasing block size from 128 to 512+ threads via autotune dramatically improved scheduler eligibility (No Eligible dropped from 40.4% to 10.9%) and occupancy (57% → 86%), but caused a 4.6x regression in wall time. The executed instruction count exploded from 6.9M to 54.3M (8x increase), and waves per SM jumped from 11 to 44.5, indicating the larger block size caused massive work redundancy or predicated-off thread overhead. The register count also doubled (17 → 32 per thread), suggesting the autotuner selected a config that triggered register spilling or additional compiler-generated instructions, overwhelming any latency-hiding benefit.

> **Applicable when**: Kernel is latency-bound with low eligible warps per scheduler (<2), high No-Eligible% (>35%), and L1TEX scoreboard stalls dominating CPI. Block size is small (≤128 threads, ≤4 warps). Register count per thread is low (≤20), suggesting room to increase block size without spilling.

> **Avoid when**: The kernel uses restore_value in autotune (causes extra save/restore passes multiplying instruction count). The kernel's xnumel is exactly divisible by all candidate XBLOCK values but masking (xmask = xindex < xnumel) is added anyway — this adds predicated instructions unnecessarily. Registers per thread are already moderate (≥20) since larger blocks may double register usage. The kernel is an in-place update (in_out_ptr0) — restore_value overhead is especially costly here.

#### ❌ Plan D: `autotune_xblock_to_hide_l1tex_latency` (0.714×)

Increasing XBLOCK via @triton.autotune (from 128 to 512–2048) successfully reduced no-eligible-warp cycles from 40.4% to 14.1% and raised achieved occupancy from 57% to 86%, but the executed instruction count more than doubled (6.87M → 14.79M) and registers per thread jumped from 17 to 32. The net effect was a 40% longer kernel duration (17.12 µs → 23.97 µs), because the extra register pressure and instruction overhead outweighed the latency-hiding benefit. When the original kernel already has very low register usage (≤20 regs/thread) and the bottleneck is L1TEX stalls, simply enlarging the tile size can cause the compiler to allocate significantly more registers, increasing per-warp cost and total instruction count enough to negate any scheduling improvement.

> **Applicable when**: Kernel has high no-eligible-warp % (>30%) due to L1TEX scoreboard stalls AND register count stays stable (does not increase) when XBLOCK is enlarged AND instruction count does not grow super-linearly with tile size.

> **Avoid when**: The compiler allocates significantly more registers when XBLOCK is increased (e.g., from 17 to 32 regs/thread), causing instruction count to more than double. Also avoid when the kernel is already memory-bandwidth bound rather than latency-bound, or when the grid is already large enough to provide sufficient occupancy through wave parallelism.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_to_fma_in_batchnorm_arithmetic | 0.741× | 0.710× | ✓ |  |
| B | reduce_non_fused_fp32_instructions_via_fma_and_rsqrt | 0.924× | 0.952× | ✓ |  |
| C | fuse_non_fused_fp32_instructions_via_rsqrt_and_fma_rewrite | 0.925× | 0.945× | ✓ |  |
| D | remove_restore_value_autotune_and_fix_xblock128_with_rsqrt_fusion | 1.001× | 1.011× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.989× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | reduce_non_fused_... | Δ | fuse_non_fused_fp... | Δ | remove_restore_va... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 24100 ns | +6980 ns (+40.8%) | 17980 ns | +860 ns (+5.0%) | 18110 ns | +990 ns (+5.8%) | 16930 ns | -190 ns (-1.1%) | 17150 ns | +30 ns (+0.2%) |
| SM Throughput (%) | 51.44 % | 78.71 % | +27.27 pp | 48.38 % | -3.06 pp | 48.43 % | -3.01 pp | 33.73 % | -17.71 pp | 60.42 % | +8.98 pp |
| Memory Throughput (%) | 22.72 % | 23.45 % | +0.73 pp | 33.36 % | +10.64 pp | 33.11 % | +10.39 pp | 22.27 % | -0.45 pp | 22.93 % | +0.21 pp |
| DRAM Throughput (%) | 14.34 % | 10.19 % | -4.15 pp | 13.66 % | -0.68 pp | 13.55 % | -0.79 pp | 14.50 % | +0.16 pp | 14.34 % | +0.00 pp |
| Achieved Occupancy (%) | 57.06 % | 86.30 % | +29.24 pp | 40.79 % | -16.27 pp | 39.56 % | -17.50 pp | 41.47 % | -15.59 pp | 61.20 % | +4.14 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.40 % | 14.93 % | -25.47 pp | 42.82 % | +2.42 pp | 42.86 % | +2.46 pp | 66.57 % | +26.17 pp | 29.59 % | -10.81 pp |
| IPC Active | 2.36 inst/cycle | 3.43 inst/cycle | +1.07 | 2.29 inst/cycle | -0.07 | 2.30 inst/cycle | -0.06 | 1.36 inst/cycle | -1.00 | 2.78 inst/cycle | +0.42 |


### Key Learnings

#### ❌ Plan A: `fuse_fp32_multiply_add_to_fma_in_batchnorm_arithmetic` (0.710×)

Replacing a 4-step batch norm arithmetic chain (sub, mul, mul, add) with explicit libdevice.fma() calls and eliminating dead multiply-by-1.0 code produced zero measurable reduction in executed instruction count (14,794,080 before and after) and no improvement in kernel duration (23.97 µs → 24.1 µs, a slight regression). Despite the NCU FPInstructions rule flagging 3.39M non-fused FP32 instructions and estimating a 38% throughput gain, the Triton JIT compiler already performs equivalent FMA fusion internally, so explicit libdevice.fma() calls are redundant and do not change the emitted PTX/SASS. When the kernel is already compute-bound at ~79% SM throughput with no change in instruction count, this arithmetic restructuring provides no benefit.

> **Applicable when**: Never — explicit libdevice.fma() calls in Triton do not reduce instruction count when the Triton JIT already performs FMA fusion. The NCU FPInstructions rule may flag non-fused instructions that are already being fused at the PTX/SASS level.

> **Avoid when**: When the NCU FPInstructions rule flags non-fused FP32 instructions in a Triton kernel that is already compute-bound (~79% SM throughput) and the executed instruction count is stable across iterations. The Triton compiler already fuses multiply-add pairs into FFMA instructions internally; explicit libdevice.fma() calls are redundant and do not change the emitted instruction count or kernel duration.

---

## Iteration 3

**Best so far**: 1.0112× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.002× | 1.008× | ✓ |  |
| B | add_streaming_eviction_policy_to_main_tensor_load | 0.997× | 0.991× | ✓ |  |
| C | evict_first_streaming_load_for_main_tensor_cache_pressure_relief | 1.009× | 1.002× | ✓ | **✓ best** |
| D | fix_broken_autotune_grid_and_add_evict_first_for_streaming_load | 0.924× | 0.954× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | add_streaming_evi... | Δ | evict_first_strea... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 16990 ns | -130 ns (-0.8%) | 17280 ns | +160 ns (+0.9%) | 17090 ns | -30 ns (-0.2%) | 17950 ns | +830 ns (+4.8%) |
| SM Throughput (%) | 51.44 % | 34.14 % | -17.30 pp | 63.19 % | +11.75 pp | 40.27 % | -11.17 pp | 51.23 % | -0.21 pp |
| Memory Throughput (%) | 22.72 % | 22.54 % | -0.18 pp | 22.63 % | -0.09 pp | 22.77 % | +0.05 pp | 33.28 % | +10.56 pp |
| DRAM Throughput (%) | 14.34 % | 14.46 % | +0.12 pp | 14.21 % | -0.13 pp | 14.39 % | +0.05 pp | 13.68 % | -0.66 pp |
| Achieved Occupancy (%) | 57.06 % | 41.33 % | -15.73 pp | 63.78 % | +6.72 pp | 43.04 % | -14.02 pp | 40.59 % | -16.47 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.40 % | 65.94 % | +25.54 pp | 25.69 % | -14.71 pp | 53.09 % | +12.69 pp | 38.90 % | -1.50 pp |
| IPC Active | 2.36 inst/cycle | 1.36 inst/cycle | -1.00 | 2.93 inst/cycle | +0.57 | 1.85 inst/cycle | -0.51 | 2.42 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.0112× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_prevent_register_pressure_regression_at_larger_xblock | 0.998× | 1.004× | ✓ |  |
| B | align_xblock_to_channel_count_multiple_to_eliminate_modulo_overhead | 1.000× | 1.004× | ✓ |  |
| C | fix_autotune_callsite_to_enable_xblock_selection | 0.996× | 0.993× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_with_multiple_of_96_xblock_configs | 1.004× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | align_xblock_to_c... | Δ | fix_autotune_call... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 17060 ns | -60 ns (-0.4%) | 17060 ns | -60 ns (-0.4%) | 17250 ns | +130 ns (+0.8%) | 17220 ns | +100 ns (+0.6%) |
| SM Throughput (%) | 51.44 % | 34.14 % | -17.30 pp | 40.26 % | -11.18 pp | 63.99 % | +12.55 pp | 44.28 % | -7.16 pp |
| Memory Throughput (%) | 22.72 % | 22.53 % | -0.19 pp | 22.79 % | +0.07 pp | 22.94 % | +0.22 pp | 30.73 % | +8.01 pp |
| DRAM Throughput (%) | 14.34 % | 14.39 % | +0.05 pp | 14.40 % | +0.06 pp | 14.25 % | -0.09 pp | 14.26 % | -0.08 pp |
| Achieved Occupancy (%) | 57.06 % | 41.41 % | -15.65 pp | 43.73 % | -13.33 pp | 63.67 % | +6.61 pp | 34.96 % | -22.10 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.40 % | 66.28 % | +25.88 pp | 52.69 % | +12.29 pp | 25.78 % | -14.62 pp | 46.08 % | +5.68 pp |
| IPC Active | 2.36 inst/cycle | 1.35 inst/cycle | -1.01 | 1.89 inst/cycle | -0.47 | 2.95 inst/cycle | +0.59 | 2.17 inst/cycle | -0.19 |

---

## Iteration 5

**Best so far**: 1.0112× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_xblock_to_96_to_eliminate_non_power_of_two_modulo | 0.995× | 0.998× | ✓ |  |
| B | eliminate_modulo96_overhead_with_max_contiguous_hints_for_channel_broadcast_loads | 1.010× | 1.008× | ✓ | **✓ best** |
| C | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency_via_warp_doubling | 0.999× | 1.009× | ✓ |  |
| D | eliminate_integer_modulo_scoreboard_stall_via_constexpr_channel_index | 0.971× | 0.966× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_xblock_to_9... | Δ | eliminate_modulo9... | Δ | increase_num_warp... | Δ | eliminate_integer... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 17150 ns | +30 ns (+0.2%) | 16990 ns | -130 ns (-0.8%) | 16960 ns | -160 ns (-0.9%) | 17730 ns | +610 ns (+3.6%) |
| SM Throughput (%) | 51.44 % | 33.51 % | -17.93 pp | 40.24 % | -11.20 pp | 40.06 % | -11.38 pp | 72.21 % | +20.77 pp |
| Memory Throughput (%) | 22.72 % | 22.37 % | -0.35 pp | 22.78 % | +0.06 pp | 22.66 % | -0.06 pp | 22.17 % | -0.55 pp |
| DRAM Throughput (%) | 14.34 % | 14.33 % | -0.01 pp | 14.45 % | +0.11 pp | 14.49 % | +0.15 pp | 13.88 % | -0.46 pp |
| Achieved Occupancy (%) | 57.06 % | 41.68 % | -15.38 pp | 43.75 % | -13.31 pp | 44.93 % | -12.13 pp | 73.98 % | +16.92 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.40 % | 69.78 % | +29.38 pp | 52.81 % | +12.41 pp | 53.44 % | +13.04 pp | 17.02 % | -23.38 pp |
| IPC Active | 2.36 inst/cycle | 1.20 inst/cycle | -1.16 | 1.89 inst/cycle | -0.47 | 1.87 inst/cycle | -0.49 | 3.30 inst/cycle | +0.94 |

---

## Cumulative Progress

```
Baseline:  17120 ns  (1.000×)
Iter 2     16930 ns  (1.011×)  — remove_restore_value_autotune_and_fix_xblock128_with_rsqrt_fusion

Final best: 16990 ns  (1.0077×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
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
+    x0_hint = tl.max_contiguous(tl.multiple_of(x0, 1), 96)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + x0_hint, xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + x0_hint, xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + x0_hint, xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + x0_hint, xmask, eviction_policy='evict_last')
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
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
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
