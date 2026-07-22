# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_28` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8420 ns — **Best latency**: 7870 ns (**1.0699×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_xblock_to_512_to_eliminate_partial_wave_tail_and_enable_vectorized_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_28`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_28_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8820,1,1), Waves/SM: 4.18
- **Registers/thread**: 18
- **Duration**: 8420 ns
- **SM Throughput**: 45.74 % | **Memory**: 17.06 % | **DRAM**: 11.03 %
- **Achieved Occupancy**: 61.61 %
- **IPC Active**: 2.65 inst/cycle | **No Eligible**: 33.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_tile_size | 0.810× | 0.793× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.979× | 1.040× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.806× | 0.268× | ✓ |  |
| D | increase_xblock_with_autotune_to_reduce_l1tex_stalls_and_improve_occupancy | 0.825× | 0.795× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8420 ns | 10620 ns | +2200 ns (+26.1%) | 8100 ns | -320 ns (-3.8%) | 31420 ns | +23000 ns (+273.2%) | 10590 ns | +2170 ns (+25.8%) |
| SM Throughput (%) | 45.74 % | 65.91 % | +20.17 pp | 47.53 % | +1.79 pp | 80.81 % | +35.07 pp | 66.96 % | +21.22 pp |
| Memory Throughput (%) | 17.06 % | 18.45 % | +1.39 pp | 17.87 % | +0.81 pp | 16.35 % | -0.71 pp | 18.69 % | +1.63 pp |
| DRAM Throughput (%) | 11.03 % | 8.71 % | -2.32 pp | 11.42 % | +0.39 pp | 2.94 % | -8.09 pp | 8.73 % | -2.30 pp |
| Achieved Occupancy (%) | 61.61 % | 82.54 % | +20.93 pp | 61.30 % | -0.31 pp | 55.62 % | -5.99 pp | 82.61 % | +21.00 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 32.00 register/thread | +14 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 13.36 | +9.18 | 4.18 | +0.00 |
| No Eligible (%) | 33.25 % | 18.72 % | -14.53 pp | 33.76 % | +0.51 pp | 13.43 % | -19.82 pp | 17.86 % | -15.39 pp |
| IPC Active | 2.65 inst/cycle | 3.29 inst/cycle | +0.64 | 2.64 inst/cycle | -0.01 | 3.46 inst/cycle | +0.81 | 3.27 inst/cycle | +0.62 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_stalls | 1.002× | 0.689× | ✓ |  |
| B | replace_reciprocal_sqrt_with_rsqrt_to_enable_fma_fusion | 0.945× | 0.993× | ✓ |  |
| C | restructure_bn_arithmetic_to_enable_fma_fusion | 0.997× | 1.024× | ✓ |  |
| D | remove_autotune_to_restore_low_register_pressure_and_occupancy | 1.002× | 1.036× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_reciproca... | Δ | restructure_bn_ar... | Δ | remove_autotune_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8420 ns | 12220 ns | +3800 ns (+45.1%) | 8480 ns | +60 ns (+0.7%) | 8220 ns | -200 ns (-2.4%) | 8130 ns | -290 ns (-3.4%) |
| SM Throughput (%) | 45.74 % | 64.53 % | +18.79 pp | 38.08 % | -7.66 pp | 29.20 % | -16.54 pp | 47.53 % | +1.79 pp |
| Memory Throughput (%) | 17.06 % | 21.43 % | +4.37 pp | 25.36 % | +8.30 pp | 18.13 % | +1.07 pp | 17.81 % | +0.75 pp |
| DRAM Throughput (%) | 11.03 % | 7.56 % | -3.47 pp | 10.91 % | -0.12 pp | 11.25 % | +0.22 pp | 11.37 % | +0.34 pp |
| Achieved Occupancy (%) | 61.61 % | 79.50 % | +17.89 pp | 43.61 % | -18.00 pp | 47.33 % | -14.28 pp | 60.19 % | -1.42 pp |
| Registers/Thread | 18.00 register/thread | 21.00 register/thread | +3 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 8.35 | +4.17 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.25 % | 22.31 % | -10.94 pp | 42.46 % | +9.21 pp | 56.95 % | +23.70 pp | 34.22 % | +0.97 pp |
| IPC Active | 2.65 inst/cycle | 3.08 inst/cycle | +0.43 | 2.28 inst/cycle | -0.37 | 1.71 inst/cycle | -0.94 | 2.62 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_at_fixed_xblock_to_hide_l1tex_stalls` (0.689×)

Adding autotune configs with num_warps=8 at XBLOCK=128 (block_size=256) caused autotune to select a XBLOCK=256 config instead, which doubled executed instructions (3.03M→6.10M), increased registers/thread from 18 to 21, and reduced Block Limit Registers from 21 to 10 blocks/SM. Although eligible warps/scheduler improved (1.92→3.61) and achieved occupancy rose (61.3%→79.5%), the doubled instruction count and higher per-instruction latency (warp cycles/instruction: 14.48→16.18) caused duration to increase from 8.1 µs to 12.22 µs — a 45% regression. The autotune search space expansion inadvertently caused selection of a larger XBLOCK tile rather than the intended num_warps increase at the same XBLOCK, and the instruction count explosion dominated over any latency-hiding benefit.

> **Applicable when**: Never applicable as described for this kernel class. Expanding the autotune config space with larger XBLOCK or higher num_warps configs risks autotune selecting a configuration that doubles instruction count and increases register pressure, even when the intent was only to change num_warps at a fixed XBLOCK.

> **Avoid when**: Do not expand the autotune config space when: (1) the kernel is already at a near-optimal XBLOCK with low register count (≤20 registers/thread); (2) adding new configs includes any XBLOCK values larger than the current best, as autotune may select them; (3) the kernel's instruction count is sensitive to XBLOCK size (pointwise kernels where instructions scale linearly with tile size). Specifically, do not add XBLOCK=256+ configs alongside XBLOCK=128 configs when the baseline is already running at XBLOCK=128 with 18 registers/thread.

---

## Iteration 3

**Best so far**: 1.0357× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_fma_fusion_via_rsqrt_and_arithmetic_restructuring | 1.020× | 1.070× | ✓ | **✓ best** |
| B | precompute_bn_scale_bias_to_collapse_arithmetic_into_fma | 1.019× | 1.024× | ✓ |  |
| C | remove_broken_autotune_and_fix_grid_to_restore_low_register_baseline | 0.997× | 1.066× | ✓ |  |
| D | enable_fma_fusion_by_fixing_int32_reciprocal_to_float32_rsqrt | 1.018× | 1.049× | ✓ |  |
| E | add_triton_autotune | 0.980× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_fma_fusion... | Δ | precompute_bn_sca... | Δ | remove_broken_aut... | Δ | enable_fma_fusion... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8420 ns | 7870 ns | -550 ns (-6.5%) | 8220 ns | -200 ns (-2.4%) | 7900 ns | -520 ns (-6.2%) | 8030 ns | -390 ns (-4.6%) | 8160 ns | -260 ns (-3.1%) |
| SM Throughput (%) | 45.74 % | 28.87 % | -16.87 pp | 28.62 % | -17.12 pp | 28.57 % | -17.17 pp | 29.19 % | -16.55 pp | 47.57 % | +1.83 pp |
| Memory Throughput (%) | 17.06 % | 17.91 % | +0.85 pp | 18.02 % | +0.96 pp | 17.93 % | +0.87 pp | 18.10 % | +1.04 pp | 17.84 % | +0.78 pp |
| DRAM Throughput (%) | 11.03 % | 11.76 % | +0.73 pp | 11.27 % | +0.24 pp | 11.72 % | +0.69 pp | 11.53 % | +0.50 pp | 11.36 % | +0.33 pp |
| Achieved Occupancy (%) | 61.61 % | 46.75 % | -14.86 pp | 46.70 % | -14.91 pp | 46.93 % | -14.68 pp | 46.41 % | -15.20 pp | 61.08 % | -0.53 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.25 % | 57.96 % | +24.71 pp | 58.15 % | +24.90 pp | 58.77 % | +25.52 pp | 57.47 % | +24.22 pp | 33.55 % | +0.30 pp |
| IPC Active | 2.65 inst/cycle | 1.70 inst/cycle | -0.95 | 1.64 inst/cycle | -1.01 | 1.64 inst/cycle | -1.01 | 1.70 inst/cycle | -0.95 | 2.64 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `enable_fma_fusion_via_rsqrt_and_arithmetic_restructuring` (1.070×)

Replacing a multi-step BN normalization chain (tl.sqrt_rn + int32 division + redundant *1.0 multiply) with libdevice.rsqrt and restructuring the multiply-add as a single FMA-friendly expression reduced total executed instructions by ~40% (3.03M → 1.83M) and eliminated branch divergence (avg divergent branches: 66.82 → 0.0, avg active threads/warp: 28.37 → 32.0). The int32 division path and the redundant scalar multiplications prevented the compiler from detecting FMA patterns and introduced predicated-off threads; replacing with rsqrt and a direct fused multiply-add (x*weight + bias) allowed the compiler to emit FFMA instructions and eliminated the divergent branches, yielding a 7% wall-clock speedup despite lower achieved occupancy (60.2% → 46.8%) and higher per-instruction stall cycles (14.95 → 17.21 cycles/instruction).

> **Applicable when**: FPInstructions NCU rule fires with high non-fused fraction (>60% non-fused FP32 instructions); kernel contains BN normalization using tl.sqrt_rn followed by integer reciprocal division (int32(1)/sqrt) and/or redundant scalar multiplications (*1.0); branch divergence is present (Avg. Divergent Branches > 0, Avg. Active Threads Per Warp < 32) suggesting predicated-off threads from the int32 path.

> **Avoid when**: The kernel already uses libdevice.rsqrt or the FP32 instruction mix is already predominantly fused (non-fused fraction < 30%); when var+eps could be zero or negative (rsqrt is only safe for strictly positive inputs); when the arithmetic chain is not a BN-style normalize-scale-shift pattern and the restructuring would change numerical semantics.

---

## Iteration 4

**Best so far**: 1.0699× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_bn_channel_params_into_shared_memory_to_eliminate_l1tex_stalls | 0.607× | 0.546× | ✓ |  |
| B | add_evict_first_to_streaming_tensor_load_to_improve_channel_param_l1_residency | 1.019× | 1.052× | ✓ |  |
| C | streaming_evict_first_on_large_tensor_to_improve_channel_param_l1_hit_rate | 1.020× | 1.070× | ✓ | **✓ best** |
| D | reorder_loads_to_overlap_bn_param_prefetch_with_main_tensor_load | 1.000× | 1.070× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_bn_channe... | Δ | add_evict_first_t... | Δ | streaming_evict_f... | Δ | reorder_loads_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8420 ns | 15420 ns | +7000 ns (+83.1%) | 8000 ns | -420 ns (-5.0%) | 7870 ns | -550 ns (-6.5%) | 7870 ns | -550 ns (-6.5%) |
| SM Throughput (%) | 45.74 % | 38.49 % | -7.25 pp | 31.42 % | -14.32 pp | 30.62 % | -15.12 pp | 30.59 % | -15.15 pp |
| Memory Throughput (%) | 17.06 % | 75.21 % | +58.15 pp | 17.96 % | +0.90 pp | 17.69 % | +0.63 pp | 17.55 % | +0.49 pp |
| DRAM Throughput (%) | 11.03 % | 5.98 % | -5.05 pp | 11.57 % | +0.54 pp | 11.77 % | +0.74 pp | 11.73 % | +0.70 pp |
| Achieved Occupancy (%) | 61.61 % | 66.02 % | +4.41 pp | 48.08 % | -13.53 pp | 47.45 % | -14.16 pp | 47.45 % | -14.16 pp |
| Registers/Thread | 18.00 register/thread | 40.00 register/thread | +22 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 5.57 | +1.39 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.25 % | 73.70 % | +40.45 pp | 54.64 % | +21.39 pp | 54.29 % | +21.04 pp | 54.52 % | +21.27 pp |
| IPC Active | 2.65 inst/cycle | 1.05 inst/cycle | -1.60 | 1.83 inst/cycle | -0.82 | 1.83 inst/cycle | -0.82 | 1.80 inst/cycle | -0.85 |


### Key Learnings

#### ✅ Plan B: `add_evict_first_to_streaming_tensor_load_to_improve_channel_param_l1_residency` (1.052×)

When a kernel loads one large streaming tensor (accessed once per element, no reuse) alongside several small per-channel broadcast tensors marked evict_last, adding eviction_policy='evict_first' to the streaming load signals L1 to immediately free those cache lines after use. In this case the change also coincided with an arithmetic restructuring (FMA pre-computation of scale/offset) that increased executed instructions by ~8% (1.83M→1.98M) but improved IPC active (1.70→1.83) and reduced warp cycles per issued instruction (17.21→16.42), yielding a 5.25% wall-clock speedup despite the higher instruction count. The CPIStall 'No Eligible' fraction dropped from 57.96% to 54.64%, confirming reduced L1TEX stall exposure. Note: the profiling data shows the kernel duration actually increased from 7.87µs to 8.00µs in the NCU trace, so the measured speedup likely reflects end-to-end benchmark improvement rather than single-kernel NCU duration; the IPC and stall improvements are real but modest.

> **Applicable when**: Kernel loads one large streaming tensor (many MB, zero intra-CTA reuse) alongside several small broadcast/channel-parameter tensors already marked evict_last. CPIStall rule fires with L1TEX stall fraction >50%. L1 hit rate is moderate (70-85%) suggesting cache pressure from the large tensor evicting the small reused tensors. The streaming tensor is accessed in a linear, non-repeating pattern within each CTA.

> **Avoid when**: The streaming tensor has intra-CTA reuse (e.g., tiled GEMM where the same data is reused across the tile). Do not apply when there are no competing evict_last tensors in the same kernel — evict_first only helps when L1 is contested. Do not expect large speedups (>10%) when the kernel is already heavily L1TEX-stall-bound; the hint reduces cache pressure but does not eliminate the memory latency bottleneck. Note that NCU single-kernel duration may not reflect the end-to-end speedup due to measurement noise at short kernel durations (<10µs).

#### ✅ Plan D: `reorder_loads_to_overlap_bn_param_prefetch_with_main_tensor_load` (1.056×)

Applying `tl.multiple_of` to the full index vector `xindex = xoffset + tl.arange(0, XBLOCK)` is semantically incorrect because individual elements of xindex are NOT multiples of XBLOCK — only xoffset (the block base address) is. Moving the hint to `xoffset = tl.multiple_of(xoffset, XBLOCK)` immediately after computing xoffset gives the compiler a correct alignment signal, enabling it to emit vectorized (128-bit) load/store instructions. In this kernel the FPInstructions non-fused count dropped from 105,840 to 70,560 (a 33% reduction in non-fused FP32 instructions), consistent with the compiler emitting more efficient code paths when the alignment is correctly communicated. The NCU-reported single-kernel duration was essentially flat (7.87→7.97 µs, within measurement noise), but the end-to-end benchmark showed a 5.65% speedup, suggesting the benefit accumulates across many kernel invocations or the NCU profiling overhead masks the true improvement.

> **Applicable when**: Triton pointwise kernel uses `tl.multiple_of(xindex, XBLOCK)` where xindex = xoffset + tl.arange(0, XBLOCK) — i.e., the hint is applied to the full index vector rather than the scalar block offset. NCU FPInstructions rule fires with a notable non-fused fraction. The kernel is memory-bandwidth or L1TEX-stall-bound and would benefit from 128-bit vectorized loads.

> **Avoid when**: The kernel already applies `tl.multiple_of` correctly to xoffset (not xindex). Do not expect large speedups (>10%) when the kernel is deeply L1TEX-stall-bound for reasons unrelated to alignment (e.g., gather/scatter patterns, non-contiguous access). Do not apply when XBLOCK is not a power-of-2 or when the access pattern is not contiguous (the alignment hint only helps for sequential, aligned accesses).

---

## Iteration 5

**Best so far**: 1.0699× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_512_to_eliminate_partial_wave_tail_and_enable_vectorized_loads | 1.018× | 1.070× | ✓ |  |
| B | fix_multiple_of_hint_and_align_grid_to_eliminate_partial_wave | 1.020× | 1.056× | ✓ |  |
| C | align_grid_to_sm_count_to_eliminate_partial_wave_tail | 1.020× | 1.066× | ✓ |  |
| D | revert_gather_to_direct_modulo_loads_with_evict_policies_and_fma | 1.021× | 1.070× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | fix_multiple_of_h... | Δ | align_grid_to_sm_... | Δ | revert_gather_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8420 ns | 7870 ns | -550 ns (-6.5%) | 7970 ns | -450 ns (-5.3%) | 7900 ns | -520 ns (-6.2%) | 7870 ns | -550 ns (-6.5%) |
| SM Throughput (%) | 45.74 % | 31.44 % | -14.30 pp | 31.15 % | -14.59 pp | 31.26 % | -14.48 pp | 30.80 % | -14.94 pp |
| Memory Throughput (%) | 17.06 % | 18.18 % | +1.12 pp | 17.82 % | +0.76 pp | 17.93 % | +0.87 pp | 17.90 % | +0.84 pp |
| DRAM Throughput (%) | 11.03 % | 11.73 % | +0.70 pp | 11.58 % | +0.55 pp | 11.72 % | +0.69 pp | 11.74 % | +0.71 pp |
| Achieved Occupancy (%) | 61.61 % | 47.38 % | -14.23 pp | 48.08 % | -13.53 pp | 47.96 % | -13.65 pp | 47.64 % | -13.97 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.25 % | 55.08 % | +21.83 pp | 55.75 % | +22.50 pp | 53.98 % | +20.73 pp | 55.97 % | +22.72 pp |
| IPC Active | 2.65 inst/cycle | 1.82 inst/cycle | -0.83 | 1.83 inst/cycle | -0.82 | 1.82 inst/cycle | -0.83 | 1.79 inst/cycle | -0.86 |

---

## Cumulative Progress

```
Baseline:  8420 ns  (1.000×)
Iter 2     8130 ns  (1.036×)  — remove_autotune_to_restore_low_register_pressure_and_occupancy
Iter 3     7870 ns  (1.070×)  — enable_fma_fusion_via_rsqrt_and_arithmetic_restructuring

Final best: 7870 ns  (1.0699×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -13,29 +14,18 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 576)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
+    tmp0 = tl.load(in_out_ptr0 + x2, mask=xmask, eviction_policy='evict_first')
+    tmp_mean   = tl.load(in_ptr0 + x0, mask=xmask, eviction_policy='evict_last')
+    tmp_var    = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')
+    tmp_weight = tl.load(in_ptr2 + x0, mask=xmask, eviction_policy='evict_last')
+    tmp_bias   = tl.load(in_ptr3 + x0, mask=xmask, eviction_policy='evict_last')
+    tmp_rvar   = libdevice.rsqrt(tmp_var + 1e-5)
+    tmp_scale  = tmp_weight * tmp_rvar
+    tmp_offset = tmp_bias - tmp_mean * tmp_scale
+    tmp_norm   = tmp0 * tmp_scale + tmp_offset
+    tmp_norm   = tl.maximum(tmp_norm, tl.full([1], 0.0, tl.float32))
+    tmp_norm   = tl.minimum(tmp_norm, tl.full([1], 6.0, tl.float32))
+    tl.store(in_out_ptr0 + x2, tmp_norm, mask=xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
