# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244130 ns — **Best latency**: 243070 ns (**1.0044×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_redundant_dual_store_single_write_path`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_2_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244130 ns
- **SM Throughput**: 32.52 % | **Memory**: 48.28 % | **DRAM**: 48.28 %
- **Achieved Occupancy**: 45.70 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 86.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | 0.998× | 1.000× | ✓ |  |
| B | vectorized_loads_with_larger_xblock | 1.000× | 1.000× | ✓ | **✓ best** |
| C | memory_coalescing_via_xblock_autotune | 0.755× | 0.756× | ✓ |  |
| D | increase_xblock_tile_size_with_autotune | 0.755× | 0.650× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244130 ns | 244190 ns | +60 ns (+0.0%) | 244160 ns | +30 ns (+0.0%) | 322940 ns | +78810 ns (+32.3%) | 375520 ns | +131390 ns (+53.8%) |
| SM Throughput (%) | 32.52 % | 32.53 % | +0.01 pp | 32.59 % | +0.07 pp | 24.77 % | -7.75 pp | 42.93 % | +10.41 pp |
| Memory Throughput (%) | 48.28 % | 48.20 % | -0.08 pp | 48.27 % | -0.01 pp | 36.43 % | -11.85 pp | 31.71 % | -16.57 pp |
| DRAM Throughput (%) | 48.28 % | 48.20 % | -0.08 pp | 48.27 % | -0.01 pp | 36.43 % | -11.85 pp | 31.32 % | -16.96 pp |
| Achieved Occupancy (%) | 45.70 % | 45.61 % | -0.09 pp | 45.82 % | +0.12 pp | 44.40 % | -1.30 pp | 44.27 % | -1.43 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 20.00 register/thread | +4 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 372.36 | +186.18 |
| No Eligible (%) | 86.02 % | 81.97 % | -4.05 pp | 86.09 % | +0.07 pp | 83.25 % | -2.77 pp | 62.16 % | -23.86 pp |
| IPC Active | 0.56 inst/cycle | 0.72 inst/cycle | +0.16 | 0.56 inst/cycle | +0.00 | 0.67 inst/cycle | +0.11 | 1.50 inst/cycle | +0.94 |


### Key Learnings

#### ⚠️ Plan D: `increase_xblock_tile_size_with_autotune` (0.650×)

Adding @triton.autotune to select larger XBLOCK values was intended to reduce grid size and increase per-CTA memory-level parallelism, but the kernel_function launch still used a hardcoded grid=(cdiv(xnumel, 128),) instead of the autotune-managed lambda grid. This caused the grid to remain at 393,216 CTAs while autotune selected a larger block size (256), doubling the total thread count from 50M to 100M and executing 4x more instructions (26.7M → 110.5M). The extra instructions inflated duration from 244µs to 375µs despite the stall-per-instruction metric improving (51 → 18 cycles/instruction), because the absolute work done ballooned.

> **Applicable when**: Pointwise kernels with high L1TEX stall cycles (>40 cycles/warp/instruction), low eligible warps per scheduler (<0.2), and small XBLOCK (≤128) causing insufficient memory-level parallelism per CTA.

> **Avoid when**: When the grid lambda is not updated to use meta['XBLOCK'] — a hardcoded grid with autotune will cause the selected block size to process more elements than exist per CTA, massively inflating instruction count. Also avoid when register pressure from larger XBLOCK reduces occupancy below the level needed to hide remaining latency.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 1.000× | 0.998× | ✓ |  |
| B | increase_num_warps_for_l1tex_latency_hiding | 1.001× | 1.000× | ✓ | **✓ best** |
| C | l2_cache_bypass_via_evict_first_load_and_stores | 1.000× | 0.998× | ✓ |  |
| D | fix_autotune_grid_lambda_mismatch | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | l2_cache_bypass_v... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244130 ns | 244670 ns | +540 ns (+0.2%) | 244130 ns | +0 ns (+0.0%) | 244510 ns | +380 ns (+0.2%) | 244740 ns | +610 ns (+0.2%) |
| SM Throughput (%) | 32.52 % | 32.51 % | -0.01 pp | 32.49 % | -0.03 pp | 32.46 % | -0.06 pp | 32.42 % | -0.10 pp |
| Memory Throughput (%) | 48.28 % | 48.15 % | -0.13 pp | 48.22 % | -0.06 pp | 48.13 % | -0.15 pp | 48.13 % | -0.15 pp |
| DRAM Throughput (%) | 48.28 % | 48.15 % | -0.13 pp | 48.22 % | -0.06 pp | 48.13 % | -0.15 pp | 48.13 % | -0.15 pp |
| Achieved Occupancy (%) | 45.70 % | 48.34 % | +2.64 pp | 45.63 % | -0.07 pp | 48.93 % | +3.23 pp | 48.57 % | +2.87 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 86.02 % | 80.36 % | -5.66 pp | 81.93 % | -4.09 pp | 76.21 % | -9.81 pp | 80.36 % | -5.66 pp |
| IPC Active | 0.56 inst/cycle | 0.79 inst/cycle | +0.23 | 0.72 inst/cycle | +0.16 | 0.95 inst/cycle | +0.39 | 0.78 inst/cycle | +0.22 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | 2d_tiled_channel_coalesced_access_pattern_rewrite | 0.514× | 0.525× | ✓ |  |
| B | differentiated_eviction_policies_load_evict_last_backward_store_evict_last | 1.000× | 0.998× | ✓ |  |
| C | store_evict_first_load_evict_last_dual_output_streaming | 0.999× | 0.999× | ✓ |  |
| D | evict_first_on_streaming_load_only | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | 2d_tiled_channel_... | Δ | differentiated_ev... | Δ | store_evict_first... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244130 ns | 465340 ns | +221210 ns (+90.6%) | 244640 ns | +510 ns (+0.2%) | 244380 ns | +250 ns (+0.1%) | 244640 ns | +510 ns (+0.2%) |
| SM Throughput (%) | 32.52 % | 52.56 % | +20.04 pp | 32.40 % | -0.12 pp | 32.56 % | +0.04 pp | 32.52 % | +0.00 pp |
| Memory Throughput (%) | 48.28 % | 45.84 % | -2.44 pp | 47.81 % | -0.47 pp | 48.14 % | -0.14 pp | 48.06 % | -0.22 pp |
| DRAM Throughput (%) | 48.28 % | 25.32 % | -22.96 pp | 47.81 % | -0.47 pp | 48.14 % | -0.14 pp | 48.06 % | -0.22 pp |
| Achieved Occupancy (%) | 45.70 % | 79.28 % | +33.58 pp | 48.82 % | +3.12 pp | 48.29 % | +2.59 pp | 48.52 % | +2.82 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 86.02 % | 47.17 % | -38.85 pp | 71.27 % | -14.75 pp | 71.35 % | -14.67 pp | 75.36 % | -10.66 pp |
| IPC Active | 0.56 inst/cycle | 2.13 inst/cycle | +1.57 | 1.15 inst/cycle | +0.59 | 1.15 inst/cycle | +0.59 | 0.98 inst/cycle | +0.42 |


### Key Learnings

#### ❌ Plan A: `2d_tiled_channel_coalesced_access_pattern_rewrite` (0.525×)

Restructuring a flat 1D pointwise kernel into a 2D tile with the channel dimension (C=6, non-power-of-2) as the inner axis and padding to C=8 (next power-of-2) caused a 5.5x instruction count explosion (34.6M → 192.2M) because the padded inner dimension (8 vs 6 real channels) wastes 25% of all memory operations on masked-out lanes, and the autotune grid was left as a static tuple instead of a lambda, causing the kernel to process each spatial position multiple times. Even though per-instruction stall cycles improved (39.7 → 23.7 cycles) and occupancy rose (45.6% → 79.3%), the total duration increased from 244µs to 465µs because the absolute instruction count ballooned. DRAM throughput dropped from 48% to 25%, confirming that the extra instructions were wasted work rather than useful memory traffic.

> **Applicable when**: Never apply this specific pattern. Avoid 2D channel-inner tiling when C is not a power-of-2 and padding is required, as masked lanes waste instruction budget. Also avoid when the autotune grid is not updated to a lambda.

> **Avoid when**: When the channel count C is not a power-of-2 (e.g., C=6): padding to the next power-of-2 (C=8) wastes 25% of all memory operations on masked lanes, causing instruction count to balloon. Also do not apply when the autotune grid is left as a static tuple — this causes the kernel to reprocess elements, multiplying instruction count. The combination of both flaws produced a 5.5x instruction explosion and 1.9x duration regression despite per-instruction stall cycles improving.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_increase_per_warp_memory_level_parallelism | 1.007× | 1.004× | ✓ | **✓ best** |
| B | alignment_hints_and_eliminate_redundant_dual_store | 0.999× | 0.999× | ✓ |  |
| C | software_pipelining_via_num_stages_to_hide_l1tex_latency | 1.000× | 0.999× | ✓ |  |
| D | revert_broken_2d_tiling_to_correct_flat_1d_with_autotune_grid_lambda | 1.006× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | alignment_hints_a... | Δ | software_pipelini... | Δ | revert_broken_2d_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244130 ns | 243230 ns | -900 ns (-0.4%) | 244350 ns | +220 ns (+0.1%) | 244480 ns | +350 ns (+0.1%) | 243520 ns | -610 ns (-0.2%) |
| SM Throughput (%) | 32.52 % | 16.34 % | -16.18 pp | 32.54 % | +0.02 pp | 32.57 % | +0.05 pp | 16.33 % | -16.19 pp |
| Memory Throughput (%) | 48.28 % | 48.36 % | +0.08 pp | 47.85 % | -0.43 pp | 48.10 % | -0.18 pp | 48.41 % | +0.13 pp |
| DRAM Throughput (%) | 48.28 % | 48.36 % | +0.08 pp | 47.85 % | -0.43 pp | 48.10 % | -0.18 pp | 48.41 % | +0.13 pp |
| Achieved Occupancy (%) | 45.70 % | 26.96 % | -18.74 pp | 48.71 % | +3.01 pp | 48.40 % | +2.70 pp | 27.57 % | -18.13 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 93.09 | -93.09 | 186.18 | +0.00 | 186.18 | +0.00 | 93.09 | -93.09 |
| No Eligible (%) | 86.02 % | 86.33 % | +0.31 pp | 71.30 % | -14.72 pp | 75.39 % | -10.63 pp | 89.22 % | +3.20 pp |
| IPC Active | 0.56 inst/cycle | 0.55 inst/cycle | -0.01 | 1.15 inst/cycle | +0.59 | 0.98 inst/cycle | +0.42 | 0.43 inst/cycle | -0.13 |

---

## Iteration 5

**Best so far**: 1.0037× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_and_add_num_stages_pipelining | 1.000× | 0.999× | ✓ |  |
| B | eliminate_redundant_dual_store_single_write_path | 1.007× | 1.004× | ✓ | **✓ best** |
| C | persistent_kernel_grid_stride_loop_to_reduce_launch_overhead_and_improve_sm_utilization | 0.999× | 0.999× | ✓ |  |
| D | fix_autotune_grid_lambda_to_enable_existing_autotune | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | eliminate_redunda... | Δ | persistent_kernel... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244130 ns | 244480 ns | +350 ns (+0.1%) | 243070 ns | -1060 ns (-0.4%) | 244420 ns | +290 ns (+0.1%) | 244740 ns | +610 ns (+0.2%) |
| SM Throughput (%) | 32.52 % | 32.52 % | +0.00 pp | 16.37 % | -16.15 pp | 38.99 % | +6.47 pp | 32.53 % | +0.01 pp |
| Memory Throughput (%) | 48.28 % | 48.08 % | -0.20 pp | 48.47 % | +0.19 pp | 47.26 % | -1.02 pp | 48.12 % | -0.16 pp |
| DRAM Throughput (%) | 48.28 % | 48.08 % | -0.20 pp | 48.47 % | +0.19 pp | 47.26 % | -1.02 pp | 48.12 % | -0.16 pp |
| Achieved Occupancy (%) | 45.70 % | 48.37 % | +2.67 pp | 27.92 % | -17.78 pp | 50.10 % | +4.40 pp | 48.23 % | +2.53 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 20.00 register/thread | +4 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 93.09 | -93.09 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 86.02 % | 71.28 % | -14.74 pp | 86.71 % | +0.69 pp | 60.83 % | -25.19 pp | 75.40 % | -10.62 pp |
| IPC Active | 0.56 inst/cycle | 1.15 inst/cycle | +0.59 | 0.53 inst/cycle | -0.03 | 1.58 inst/cycle | +1.02 | 0.98 inst/cycle | +0.42 |

---

## Cumulative Progress

```
Baseline:  244130 ns  (1.000×)
Iter 4     243230 ns  (1.004×)  — reduce_num_warps_to_increase_per_warp_memory_level_parallelism
Iter 5     243070 ns  (1.004×)  — eliminate_redundant_dual_store_single_write_path

Final best: 243070 ns  (1.0044×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 50331648
+def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None)
-    tmp1 = tl.full([1], 0, tl.int32)
-    tmp2 = tl.maximum(tmp1, tmp0)
-    tl.store(out_ptr0 + (x0), tmp2, None)
-    tl.store(out_ptr1 + (x0), tmp2, None)
-
-
+    xmask = xindex < xnumel
+    tmp0 = tl.load(in_ptr0 + xindex, xmask)
+    tmp2 = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp0)
+    tl.store(out_ptr0 + xindex, tmp2, xmask)
+    tl.store(out_ptr1 + xindex, tmp2, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -26,7 +30,7 @@

     arg_2 = inputs[2]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_relu_2[grid](arg_0, arg_1, arg_2, 50331648, XBLOCK=128)
+    triton_poi_fused_convolution_relu_2[grid](arg_0, arg_1, arg_2, 50331648)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
