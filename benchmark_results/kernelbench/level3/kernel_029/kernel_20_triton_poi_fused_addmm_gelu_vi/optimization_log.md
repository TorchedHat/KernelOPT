# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_gelu_view_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31330 ns — **Best latency**: 31330 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_gelu_view_20`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_20_triton_poi_fused_addmm_gelu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (47040,1,1), Waves/SM: 22.27
- **Registers/thread**: 16
- **Duration**: 31330 ns
- **SM Throughput**: 44.69 % | **Memory**: 19.39 % | **DRAM**: 16.20 %
- **Achieved Occupancy**: 43.63 %
- **IPC Active**: 1.95 inst/cycle | **No Eligible**: 51.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_latency_hiding | 0.663× | 0.657× | ✓ |  |
| B | increase_xblock_with_autotune_for_pointwise_gelu | 0.662× | 0.657× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_pointwise_gelu | 0.372× | 0.353× | ✓ |  |
| D | vectorized_loads_stores_via_larger_xblock | 0.663× | 0.653× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31330 ns | 47680 ns | +16350 ns (+52.2%) | 47650 ns | +16320 ns (+52.1%) | 88700 ns | +57370 ns (+183.1%) | 48000 ns | +16670 ns (+53.2%) |
| SM Throughput (%) | 44.69 % | 81.69 % | +37.00 pp | 81.94 % | +37.25 pp | 82.91 % | +38.22 pp | 81.91 % | +37.22 pp |
| Memory Throughput (%) | 19.39 % | 12.63 % | -6.76 pp | 12.62 % | -6.77 pp | 9.44 % | -9.95 pp | 12.53 % | -6.86 pp |
| DRAM Throughput (%) | 16.20 % | 10.62 % | -5.58 pp | 10.62 % | -5.58 pp | 5.71 % | -10.49 pp | 10.55 % | -5.65 pp |
| Achieved Occupancy (%) | 43.63 % | 86.90 % | +43.27 pp | 85.55 % | +41.92 pp | 91.80 % | +48.17 pp | 86.60 % | +42.97 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 32.00 register/thread | +16 | 28.00 register/thread | +12 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 |
| No Eligible (%) | 51.18 % | 14.92 % | -36.26 pp | 14.16 % | -37.02 pp | 14.70 % | -36.48 pp | 14.70 % | -36.48 pp |
| IPC Active | 1.95 inst/cycle | 3.43 inst/cycle | +1.48 | 3.40 inst/cycle | +1.45 | 3.40 inst/cycle | +1.45 | 3.42 inst/cycle | +1.47 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size_for_latency_hiding` (0.657×)

Adding @triton.autotune with larger XBLOCK values (512–4096) to a kernel that was originally XBLOCK=128 caused registers_per_thread to jump from 16 to 28, which halved the block limit from registers (32→16 blocks/SM). Although occupancy and eligible warps improved dramatically (43.6%→86.9%, 0.9→7.74 eligible warps/scheduler), the total executed instruction count nearly tripled (11M→30M), indicating the autotuner selected a configuration that dramatically increased per-CTA work or introduced additional overhead (e.g., masking with xmask instead of tl.full True). The net effect was a 52% slowdown despite better scheduling efficiency, because the instruction count explosion outweighed the latency-hiding benefit.

> **Applicable when**: Never apply this specific pattern as described — it caused regression. The autotune approach for XBLOCK increase is theoretically valid when: kernel is memory-latency bound (CPIStall on L1TEX >50%), eligible warps/scheduler <1.0, and the kernel body is simple enough that larger tiles don't inflate instruction count.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK values when: (1) the original kernel uses xmask=tl.full([XBLOCK], True) (no masking needed) and switching to xmask=xindex<xnumel adds unnecessary predication overhead; (2) the kernel has very low register usage (≤16 regs/thread) and larger tiles cause register pressure to jump (16→28 regs), halving block-level occupancy limits; (3) the instruction count is expected to grow proportionally with XBLOCK without a corresponding reduction in kernel launches. Also avoid when the grid is hardcoded and not updated to use a lambda — the diff shows grid=(cdiv(xnumel,128),) was NOT updated to use meta['XBLOCK'], meaning the autotuner selected larger XBLOCK but the grid remained fixed at XBLOCK=128 granularity, causing each CTA to do more work than intended.

#### ❌ Plan B: `increase_xblock_with_autotune_for_pointwise_gelu` (0.657×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) to a pointwise kernel that was stall-bound due to low occupancy improved occupancy (43% → 85%) and scheduler eligibility, but caused a 52% slowdown. The root cause was that the autotune decorator caused registers_per_thread to jump from 16 to 28, and executed instructions ballooned from 11M to 30M — a 2.77x increase — indicating the selected autotune config introduced significant instruction overhead (likely from the xmask guard added for non-power-of-2 tile boundaries or from the autotune overhead itself). The net effect was a longer kernel duration (31 µs → 47 µs) despite better occupancy, because the instruction count increase outweighed the latency-hiding benefit.

> **Applicable when**: Never — this specific combination of autotune + xmask introduction + static grid caused regression

> **Avoid when**: Do not add xmask guards when xnumel is already exactly divisible by XBLOCK (xmask=True is correct and cheaper). Do not add @triton.autotune to a kernel while keeping a static grid that does not adapt to meta['XBLOCK'] — the grid must use a lambda: `grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Do not increase XBLOCK if the resulting register pressure (registers_per_thread) increases significantly, as this can reduce theoretical occupancy and increase total instruction count beyond the latency-hiding benefit.

#### ❌ Plan D: `vectorized_loads_stores_via_larger_xblock` (0.653×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that had XBLOCK=128 hardcoded caused a 2.77x increase in executed instructions (11.1M → 30.8M) and a 53% increase in duration (31.33 µs → 48.0 µs), despite improving achieved occupancy from 43.6% to 86.6% and compute SM throughput from 44.7% to 81.9%. The regression occurred because the autotune selected a config that dramatically increased register pressure (16 → 28 registers/thread), which combined with the larger tile size caused each warp to execute far more instructions per element — likely due to the xmask guard being evaluated per-element across a much wider XBLOCK, or the autotune selecting a suboptimal config that unrolled more aggressively. When the original kernel was latency-bound (CPIStall, 62.7% of cycles on L1TEX), simply increasing XBLOCK without controlling register spill and instruction count can make things worse even if occupancy improves.

> **Applicable when**: Never — this specific strategy (autotune larger XBLOCK for pointwise kernels with small XBLOCK) caused regression here. The approach may work in other contexts but failed here.

> **Avoid when**: Do not apply autotune with larger XBLOCK configs when: (1) the original kernel uses few registers (≤16/thread) and adding larger tiles causes register count to jump significantly (e.g., 16→28), as this indicates the compiler is doing more work per tile; (2) the instruction count grows super-linearly with XBLOCK (more than proportionally); (3) the kernel is a simple pointwise op where the bottleneck is latency rather than throughput — increasing XBLOCK increases occupancy but also increases per-CTA work in a way that doesn't amortize the latency. Also avoid when the grid is already fixed and the autotune does not update the grid lambda to use meta['XBLOCK'] (the diff shows the grid was NOT updated to use meta['XBLOCK'], remaining hardcoded at cdiv(xnumel, 128), meaning all autotune configs still launched 47040 CTAs regardless of XBLOCK — this is likely the primary bug causing the regression).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.999× | 0.995× | ✓ |  |
| B | persistent_tile_scheduling_with_cached_bias | 1.005× | 1.000× | ✓ | **✓ best** |
| C | persistent_tile_scheduling | 0.998× | 1.000× | ✓ |  |
| D | persistent_tile_scheduling | 1.005× | 1.000× | ✓ |  |
| E | add_triton_autotune | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31330 ns | 31490 ns | +160 ns (+0.5%) | 31330 ns | +0 ns (+0.0%) | 31330 ns | +0 ns (+0.0%) | 31330 ns | +0 ns (+0.0%) | 31330 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.69 % | 52.55 % | +7.86 pp | 53.28 % | +8.59 pp | 57.03 % | +12.34 pp | 53.23 % | +8.54 pp | 47.91 % | +3.22 pp |
| Memory Throughput (%) | 19.39 % | 19.26 % | -0.13 pp | 19.39 % | +0.00 pp | 19.37 % | -0.02 pp | 19.38 % | -0.01 pp | 19.38 % | -0.01 pp |
| DRAM Throughput (%) | 16.20 % | 16.09 % | -0.11 pp | 16.15 % | -0.05 pp | 16.21 % | +0.01 pp | 16.16 % | -0.04 pp | 16.16 % | -0.04 pp |
| Achieved Occupancy (%) | 43.63 % | 47.08 % | +3.45 pp | 47.13 % | +3.50 pp | 49.63 % | +6.00 pp | 48.43 % | +4.80 pp | 44.99 % | +1.36 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 |
| No Eligible (%) | 51.18 % | 43.12 % | -8.06 pp | 42.21 % | -8.97 pp | 38.06 % | -13.12 pp | 42.20 % | -8.98 pp | 47.79 % | -3.39 pp |
| IPC Active | 1.95 inst/cycle | 2.28 inst/cycle | +0.33 | 2.32 inst/cycle | +0.37 | 2.45 inst/cycle | +0.50 | 2.31 inst/cycle | +0.36 | 2.08 inst/cycle | +0.13 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_arithmetic_to_fma_in_gelu_body | 0.999× | 1.000× | ✓ |  |
| B | increase_num_warps_fixed_xblock_for_l1tex_latency_hiding | 0.998× | 0.999× | ✓ |  |
| C | fuse_fp32_multiply_add_instructions_in_gelu_body | 1.000× | 1.000× | ✓ |  |
| D | preload_bias_into_registers_to_eliminate_l1tex_stalls | 0.989× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_arithme... | Δ | increase_num_warp... | Δ | fuse_fp32_multipl... | Δ | preload_bias_into... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31330 ns | 31330 ns | +0 ns (+0.0%) | 31360 ns | +30 ns (+0.1%) | 31330 ns | +0 ns (+0.0%) | 31840 ns | +510 ns (+1.6%) |
| SM Throughput (%) | 44.69 % | 52.52 % | +7.83 pp | 57.07 % | +12.38 pp | 53.22 % | +8.53 pp | 53.53 % | +8.84 pp |
| Memory Throughput (%) | 19.39 % | 19.39 % | +0.00 pp | 19.35 % | -0.04 pp | 19.38 % | -0.01 pp | 60.64 % | +41.25 pp |
| DRAM Throughput (%) | 16.20 % | 16.16 % | -0.04 pp | 16.19 % | -0.01 pp | 16.17 % | -0.03 pp | 15.94 % | -0.26 pp |
| Achieved Occupancy (%) | 43.63 % | 46.92 % | +3.29 pp | 48.50 % | +4.87 pp | 47.90 % | +4.27 pp | 52.82 % | +9.19 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 17.00 register/thread | +1 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 |
| No Eligible (%) | 51.18 % | 43.15 % | -8.03 pp | 38.06 % | -13.12 pp | 42.21 % | -8.97 pp | 41.94 % | -9.24 pp |
| IPC Active | 1.95 inst/cycle | 2.27 inst/cycle | +0.32 | 2.47 inst/cycle | +0.52 | 2.31 inst/cycle | +0.36 | 2.31 inst/cycle | +0.36 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_main_tensor_load | 1.000× | 0.991× | ✓ |  |
| B | stage_bias_in_shared_memory_to_eliminate_l1tex_stalls | 0.986× | 0.982× | ✓ |  |
| C | eliminate_modulo_bias_index_via_contiguous_slice_load | 1.005× | 0.999× | ✓ |  |
| D | eliminate_tl_gather_and_fuse_gelu_fma_instructions | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | stage_bias_in_sha... | Δ | eliminate_modulo_... | Δ | eliminate_tl_gath... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31330 ns | 31620 ns | +290 ns (+0.9%) | 31900 ns | +570 ns (+1.8%) | 31360 ns | +30 ns (+0.1%) | 31330 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.69 % | 56.35 % | +11.66 pp | 60.27 % | +15.58 pp | 56.20 % | +11.51 pp | 44.91 % | +0.22 pp |
| Memory Throughput (%) | 19.39 % | 19.21 % | -0.18 pp | 62.12 % | +42.73 pp | 19.37 % | -0.02 pp | 19.40 % | +0.01 pp |
| DRAM Throughput (%) | 16.20 % | 16.00 % | -0.20 pp | 15.83 % | -0.37 pp | 16.13 % | -0.07 pp | 16.13 % | -0.07 pp |
| Achieved Occupancy (%) | 43.63 % | 47.41 % | +3.78 pp | 56.88 % | +13.25 pp | 48.61 % | +4.98 pp | 43.75 % | +0.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 19.00 register/thread | +3 | 16.00 register/thread | +0 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 |
| No Eligible (%) | 51.18 % | 38.93 % | -12.25 pp | 34.68 % | -16.50 pp | 38.96 % | -12.22 pp | 51.25 % | +0.07 pp |
| IPC Active | 1.95 inst/cycle | 2.44 inst/cycle | +0.49 | 2.59 inst/cycle | +0.64 | 2.44 inst/cycle | +0.49 | 1.95 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  31330 ns  (1.000×)

Final best: 31330 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 6021120
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 768)
-    x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp2 = tmp0 + tmp1
-    tmp3 = tl.full([1], 0.5, tl.float32)
-    tmp4 = tmp2 * tmp3
-    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
-    tmp6 = tmp2 * tmp5
-    tmp7 = libdevice.erf(tmp6)
-    tmp8 = tl.full([1], 1.0, tl.float32)
-    tmp9 = tmp7 + tmp8
-    tmp10 = tmp4 * tmp9
-    tl.store(in_out_ptr0 + (x2), tmp10, None)
-
-
+def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    num_sms = tl.num_programs(0)
+    sm_id = tl.program_id(0)   # 0..num_sms-1, one CTA per SM
+    tile_id = sm_id             # first tile assigned to this SM
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        # xnumel=6021120 exactly divisible by XBLOCK=128 -- no mask needed
+        x0 = xindex % 768       # bias index, stays hot in L1 across tiles
+        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
+        tmp1 = tl.load(in_out_ptr0 + xindex)
+        tmp2 = tmp0 + tmp1
+        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
+        # Use FMA: half*(tmp7+1) = half*tmp7 + half = fma(half, tmp7, half)
+        tmp6 = tmp2 * 0.7071067811865476
+        tmp7 = libdevice.erf(tmp6)
+        half = tmp2 * 0.5
+        tmp10 = tl.math.fma(half, tmp7, half)  # half*tmp7 + half = half*(tmp7+1)
+        tl.store(in_out_ptr0 + xindex, tmp10)
+        tile_id += num_sms      # stride by SM count
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
