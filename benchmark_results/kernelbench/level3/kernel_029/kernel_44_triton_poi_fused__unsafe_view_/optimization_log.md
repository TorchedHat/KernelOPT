# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7420 ns — **Best latency**: 7490 ns (**0.9907×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_44_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 18
- **Duration**: 7420 ns
- **SM Throughput**: 50.06 % | **Memory**: 24.92 % | **DRAM**: 24.92 %
- **Achieved Occupancy**: 76.53 %
- **IPC Active**: 2.77 inst/cycle | **No Eligible**: 31.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 0.858× | 0.765× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 1.027× | 0.983× | ✓ | **✓ best** |
| C | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.677× | 0.564× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 1.026× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 9700 ns | +2280 ns (+30.7%) | 7550 ns | +130 ns (+1.8%) | 13150 ns | +5730 ns (+77.2%) | 7420 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 50.06 % | 62.77 % | +12.71 pp | 50.12 % | +0.06 pp | 66.10 % | +16.04 pp | 49.69 % | -0.37 pp |
| Memory Throughput (%) | 24.92 % | 19.06 % | -5.86 pp | 24.52 % | -0.40 pp | 18.51 % | -6.41 pp | 24.88 % | -0.04 pp |
| DRAM Throughput (%) | 24.92 % | 19.06 % | -5.86 pp | 24.52 % | -0.40 pp | 14.04 % | -10.88 pp | 24.88 % | -0.04 pp |
| Achieved Occupancy (%) | 76.53 % | 78.83 % | +2.30 pp | 76.21 % | -0.32 pp | 80.61 % | +4.08 pp | 76.51 % | -0.02 pp |
| Registers/Thread | 18.00 register/thread | 26.00 register/thread | +8 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 31.23 % | 22.52 % | -8.71 pp | 31.20 % | -0.03 pp | 22.12 % | -9.11 pp | 29.12 % | -2.11 pp |
| IPC Active | 2.77 inst/cycle | 3.06 inst/cycle | +0.29 | 2.74 inst/cycle | -0.03 | 3.10 inst/cycle | +0.33 | 2.77 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_eliminate_partial_wave` (0.765×)

Adding @triton.autotune with larger XBLOCK configs (256/512/1024) to a small pointwise kernel that NCU flags for a partial-wave tail (LaunchConfiguration rule, 33.3% speedup_pct) caused a 31% slowdown. The autotune overhead and the selected larger XBLOCK increased register pressure from 18 to 26 registers/thread, which reduced Block Limit Registers from 21 to 16 blocks/SM. More critically, executed instructions nearly doubled (2.94M → 4.66M) while duration increased from 7.42 µs to 9.7 µs, suggesting the autotuner selected a suboptimal config or the autotune overhead itself dominated for this short-running kernel. The partial-wave penalty flagged by NCU (33.3%) was not realized as a gain because the instruction count explosion and register pressure increase outweighed any tile-size benefit.

> **Applicable when**: Never apply this pattern when: kernel duration is < 10 µs, register count is near the block limit boundary (18 regs/thread with block limit registers = 21), and the partial-wave penalty is the only flagged issue. The autotune overhead and register pressure increase from larger tiles will dominate.

> **Avoid when**: Do not apply @triton.autotune with larger XBLOCK configs to short-duration pointwise kernels (< 10 µs) where registers/thread is already near the block-limit-registers threshold. Increasing XBLOCK raises register pressure, reduces blocks/SM, and the autotune calibration cost itself can exceed the partial-wave savings for kernels with < 3 waves. Also avoid when executed instruction count is already low (< 3M) — the autotune overhead is disproportionate.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.992× | 0.974× | ✓ |  |
| B | persistent_tile_scheduling | 1.006× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 1.001× | 0.983× | ✓ |  |
| D | persistent_tile_scheduling | 0.989× | 0.939× | ✓ |  |
| E | add_triton_autotune | 1.020× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7620 ns | +200 ns (+2.7%) | 7420 ns | +0 ns (+0.0%) | 7550 ns | +130 ns (+1.8%) | 7900 ns | +480 ns (+6.5%) | 7580 ns | +160 ns (+2.2%) |
| SM Throughput (%) | 50.06 % | 53.19 % | +3.13 pp | 51.61 % | +1.55 pp | 51.61 % | +1.55 pp | 52.88 % | +2.82 pp | 49.85 % | -0.21 pp |
| Memory Throughput (%) | 24.92 % | 24.33 % | -0.59 pp | 24.95 % | +0.03 pp | 24.43 % | -0.49 pp | 23.39 % | -1.53 pp | 24.36 % | -0.56 pp |
| DRAM Throughput (%) | 24.92 % | 24.33 % | -0.59 pp | 24.95 % | +0.03 pp | 24.43 % | -0.49 pp | 23.39 % | -1.53 pp | 24.36 % | -0.56 pp |
| Achieved Occupancy (%) | 76.53 % | 76.69 % | +0.16 pp | 75.75 % | -0.78 pp | 76.03 % | -0.50 pp | 76.10 % | -0.43 pp | 75.81 % | -0.72 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 24.00 register/thread | +6 | 24.00 register/thread | +6 | 25.00 register/thread | +7 | 18.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 31.23 % | 25.26 % | -5.97 pp | 27.68 % | -3.55 pp | 26.84 % | -4.39 pp | 27.29 % | -3.94 pp | 28.93 % | -2.30 pp |
| IPC Active | 2.77 inst/cycle | 2.90 inst/cycle | +0.13 | 2.85 inst/cycle | +0.08 | 2.88 inst/cycle | +0.11 | 2.85 inst/cycle | +0.08 | 2.75 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_adds_into_fma_instructions | 1.019× | 1.022× | ✓ |  |
| B | fma_fusion_and_streaming_eviction_for_large_tensor | 1.011× | 0.970× | ✓ |  |
| C | fuse_sequential_additions_to_reduce_non_fused_fp32_instructions | 1.019× | 1.008× | ✓ |  |
| D | fma_fusion_and_address_arithmetic_reduction | 0.973× | 0.946× | ✓ |  |
| E | add_triton_autotune | 1.049× | 0.991× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_fp32_adds_in... | Δ | fma_fusion_and_st... | Δ | fuse_sequential_a... | Δ | fma_fusion_and_ad... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7260 ns | -160 ns (-2.2%) | 7650 ns | +230 ns (+3.1%) | 7360 ns | -60 ns (-0.8%) | 7840 ns | +420 ns (+5.7%) | 7490 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 50.06 % | 51.02 % | +0.96 pp | 53.31 % | +3.25 pp | 50.69 % | +0.63 pp | 52.65 % | +2.59 pp | 51.73 % | +1.67 pp |
| Memory Throughput (%) | 24.92 % | 25.48 % | +0.56 pp | 24.16 % | -0.76 pp | 25.08 % | +0.16 pp | 23.58 % | -1.34 pp | 24.64 % | -0.28 pp |
| DRAM Throughput (%) | 24.92 % | 25.48 % | +0.56 pp | 24.16 % | -0.76 pp | 25.08 % | +0.16 pp | 23.58 % | -1.34 pp | 24.64 % | -0.28 pp |
| Achieved Occupancy (%) | 76.53 % | 75.60 % | -0.93 pp | 75.88 % | -0.65 pp | 75.90 % | -0.63 pp | 76.23 % | -0.30 pp | 75.33 % | -1.20 pp |
| Registers/Thread | 18.00 register/thread | 25.00 register/thread | +7 | 26.00 register/thread | +8 | 18.00 register/thread | +0 | 26.00 register/thread | +8 | 24.00 register/thread | +6 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 31.23 % | 26.93 % | -4.30 pp | 26.63 % | -4.60 pp | 27.81 % | -3.42 pp | 27.37 % | -3.86 pp | 26.81 % | -4.42 pp |
| IPC Active | 2.77 inst/cycle | 2.87 inst/cycle | +0.10 | 2.90 inst/cycle | +0.13 | 2.81 inst/cycle | +0.04 | 2.88 inst/cycle | +0.11 | 2.83 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_last_to_reused_strided_tensor_and_simplify_address_arithmetic | 0.997× | 0.991× | ✓ |  |
| B | reduce_address_arithmetic_for_strided_gather_load | 0.989× | 0.979× | ✓ |  |
| C | reduce_address_arithmetic_complexity_for_strided_load | 1.010× | 0.983× | ✓ |  |
| D | streaming_eviction_for_large_non_reused_tensor_to_improve_l2_hit_rate | 0.949× | 0.906× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_last_to... | Δ | reduce_address_ar... | Δ | reduce_address_ar... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7490 ns | +70 ns (+0.9%) | 7580 ns | +160 ns (+2.2%) | 7550 ns | +130 ns (+1.8%) | 8190 ns | +770 ns (+10.4%) |
| SM Throughput (%) | 50.06 % | 53.24 % | +3.18 pp | 49.79 % | -0.27 pp | 52.92 % | +2.86 pp | 54.45 % | +4.39 pp |
| Memory Throughput (%) | 24.92 % | 24.63 % | -0.29 pp | 24.35 % | -0.57 pp | 24.48 % | -0.44 pp | 22.58 % | -2.34 pp |
| DRAM Throughput (%) | 24.92 % | 24.63 % | -0.29 pp | 24.35 % | -0.57 pp | 24.48 % | -0.44 pp | 22.58 % | -2.34 pp |
| Achieved Occupancy (%) | 76.53 % | 76.85 % | +0.32 pp | 76.74 % | +0.21 pp | 77.17 % | +0.64 pp | 76.44 % | -0.09 pp |
| Registers/Thread | 18.00 register/thread | 23.00 register/thread | +5 | 18.00 register/thread | +0 | 26.00 register/thread | +8 | 26.00 register/thread | +8 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 31.23 % | 24.32 % | -6.91 pp | 29.95 % | -1.28 pp | 25.90 % | -5.33 pp | 24.46 % | -6.77 pp |
| IPC Active | 2.77 inst/cycle | 2.84 inst/cycle | +0.07 | 2.78 inst/cycle | +0.01 | 2.96 inst/cycle | +0.19 | 2.97 inst/cycle | +0.20 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_alu_register_pressure_via_index_hoisting | 1.020× | 0.995× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 1.015× | 0.983× | ✓ |  |
| C | increase_num_warps_to_hide_address_arithmetic_latency | 0.853× | 0.642× | ✓ |  |
| D | replace_trivial_fma_with_add_and_hoist_invariant_index_arithmetic | 0.913× | 0.840× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_alu_regist... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | replace_trivial_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7460 ns | +40 ns (+0.5%) | 7550 ns | +130 ns (+1.8%) | 11550 ns | +4130 ns (+55.7%) | 8830 ns | +1410 ns (+19.0%) |
| SM Throughput (%) | 50.06 % | 52.95 % | +2.89 pp | 52.98 % | +2.92 pp | 64.95 % | +14.89 pp | 54.20 % | +4.14 pp |
| Memory Throughput (%) | 24.92 % | 24.74 % | -0.18 pp | 24.48 % | -0.44 pp | 15.97 % | -8.95 pp | 20.90 % | -4.02 pp |
| DRAM Throughput (%) | 24.92 % | 24.74 % | -0.18 pp | 24.48 % | -0.44 pp | 15.97 % | -8.95 pp | 20.90 % | -4.02 pp |
| Achieved Occupancy (%) | 76.53 % | 75.44 % | -1.09 pp | 76.92 % | +0.39 pp | 82.91 % | +6.38 pp | 78.29 % | +1.76 pp |
| Registers/Thread | 18.00 register/thread | 23.00 register/thread | +5 | 26.00 register/thread | +8 | 18.00 register/thread | +0 | 26.00 register/thread | +8 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 5.57 | +2.79 | 2.78 | +0.00 |
| No Eligible (%) | 31.23 % | 26.53 % | -4.70 pp | 25.00 % | -6.23 pp | 21.12 % | -10.11 pp | 27.84 % | -3.39 pp |
| IPC Active | 2.77 inst/cycle | 2.90 inst/cycle | +0.13 | 2.87 inst/cycle | +0.10 | 3.13 inst/cycle | +0.36 | 2.83 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_to_hide_address_arithmetic_latency` (0.642×)

Increasing block size from 128 to 256 (doubling num_warps from 4 to 8) on a short-duration pointwise kernel (~7.5 µs) to address a 24% no-eligible-warp stall rate caused a 36% slowdown. Although achieved occupancy improved slightly (76.85% → 82.91%) and the no-eligible stall rate dropped modestly (24.32% → 21.12%), executed instructions nearly doubled (3.15M → 5.93M) and duration increased from 7.49 µs to 11.55 µs. The instruction explosion suggests the kernel body was restructured (eviction policies changed, FMA chains replaced with plain additions) alongside the block size change, and the combined effect of more waves (2.78 → 5.57) with more instructions per wave dominated over any latency-hiding benefit. For very short pointwise kernels where the no-eligible stall rate is moderate (~24%), the overhead of doubling the warp pool does not compensate for the increased total instruction count.

> **Applicable when**: Never apply this pattern to short-duration pointwise kernels (< 10 µs) where the no-eligible stall rate is moderate (20-30%) but not extreme. The latency-hiding benefit of more warps is insufficient to overcome the overhead when the kernel is already near its minimum duration.

> **Avoid when**: Do not increase block size (num_warps) on short-duration pointwise kernels (< 10 µs) when: (1) the no-eligible stall rate is only moderate (~24%), not extreme (>50%); (2) the kernel body is simultaneously restructured (changing eviction policies, FMA→plain add), making it impossible to isolate the block-size effect; (3) the autotune overhead adds calibration cost that dominates for kernels with < 3 waves. Also avoid when executed instruction count is already near 3M — doubling block size risks doubling instructions if the autotuner selects a suboptimal config.

---

## Cumulative Progress

```
Baseline:  7420 ns  (1.000×)

Final best: 7490 ns  (0.9907×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 752640
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x3 = xindex
-    x0 = (xindex % 384)
-    x1 = ((xindex // 384) % 196)
-    x2 = xindex // 75264
-    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
-    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 14))) % 7)) + 224*(((4 + (x1 // 14)) % 7)) + 1568*(x0 // 32) + 18816*((4 + ((x1 % 14))) // 7) + 56448*((4 + (x1 // 14)) // 7) + 169344*x2 + ((x0 % 32))), xmask)
-    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 14)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 14))) % 7))), xmask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp6 = tl.load(in_ptr3 + (x3), xmask)
-    tmp3 = tmp1 + tmp2
-    tmp4 = tmp0 + tmp3
-    tmp7 = tmp5 + tmp6
-    tmp8 = tmp4 + tmp7
-    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
-
-
+    pid = tl.program_id(0)
+    num_sms = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x3 = xindex
+        x0 = (xindex % 384)
+        x1 = ((xindex // 384) % 196)
+        x2 = xindex // 75264
+        x1_mod_14 = x1 % 14
+        x1_div_14 = x1 // 14
+        x0_div_32 = x0 // 32
+        x0_mod_32 = x0 % 32
+        x1_mod_14_s = 4 + x1_mod_14
+        x1_div_14_s = 4 + x1_div_14
+        tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
+        tmp1 = tl.load(in_ptr0 + (32*(x1_mod_14_s % 7) + 224*(x1_div_14_s % 7) + 1568*x0_div_32 + 18816*(x1_mod_14_s // 7) + 56448*(x1_div_14_s // 7) + 169344*x2 + x0_mod_32), xmask, eviction_policy='evict_last')
+        tmp2 = tl.load(in_ptr1 + (7*(x1_div_14_s % 7) + 49*x0_div_32 + (x1_mod_14_s % 7)), xmask, eviction_policy='evict_last')
+        tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+        tmp6 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_first')
+        # fma(1.0, a, b) = 1.0*a + b = a + b, emits PTX fma.rn.f32 (FFMA)
+        # Chain all 5 values: tmp0 + tmp1 + tmp2 + tmp5 + tmp6
+        acc = libdevice.fma(1.0, tmp1, tmp0)   # tmp0 + tmp1
+        acc = libdevice.fma(1.0, tmp2, acc)    # tmp0 + tmp1 + tmp2
+        acc = libdevice.fma(1.0, tmp5, acc)    # tmp0 + tmp1 + tmp2 + tmp5
+        acc = libdevice.fma(1.0, tmp6, acc)    # tmp0 + tmp1 + tmp2 + tmp5 + tmp6
+        tl.store(in_out_ptr0 + (x3), acc, xmask)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
