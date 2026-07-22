# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_unfold_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3940 ns — **Best latency**: 3680 ns (**1.0707×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`persistent_tile_scheduling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_unfold_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_0_triton_poi_fused_clone_unfold_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2352,1,1), Waves/SM: 1.11
- **Registers/thread**: 16
- **Duration**: 3940 ns
- **SM Throughput**: 13.26 % | **Memory**: 11.09 % | **DRAM**: 6.27 %
- **Achieved Occupancy**: 61.95 %
- **IPC Active**: 1.31 inst/cycle | **No Eligible**: 65.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.044× | 1.071× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.061× | 1.071× | ✓ | **✓ best** |
| C | increase_xblock_with_autotune_to_eliminate_partial_wave | 1.022× | 0.816× | ✓ |  |
| D | add_triton_autotune | 1.015× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_e... | Δ | increase_xblock_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3680 ns | -260 ns (-6.6%) | 3680 ns | -260 ns (-6.6%) | 4830 ns | +890 ns (+22.6%) | 4000 ns | +60 ns (+1.5%) |
| SM Throughput (%) | 13.26 % | 10.09 % | -3.17 pp | 13.84 % | +0.58 pp | 41.52 % | +28.26 pp | 13.58 % | +0.32 pp |
| Memory Throughput (%) | 11.09 % | 11.90 % | +0.81 pp | 11.03 % | -0.06 pp | 11.59 % | +0.50 pp | 10.99 % | -0.10 pp |
| DRAM Throughput (%) | 6.27 % | 6.72 % | +0.45 pp | 6.77 % | +0.50 pp | 5.12 % | -1.15 pp | 6.21 % | -0.06 pp |
| Achieved Occupancy (%) | 61.95 % | 55.22 % | -6.73 pp | 59.01 % | -2.94 pp | 70.91 % | +8.96 pp | 58.84 % | -3.11 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.11 | 1.11 | +0.00 | 1.11 | +0.00 | 4.45 | +3.34 | 1.11 | +0.00 |
| No Eligible (%) | 65.56 % | 70.03 % | +4.47 pp | 58.89 % | -6.67 pp | 34.04 % | -31.52 pp | 65.55 % | -0.01 pp |
| IPC Active | 1.31 inst/cycle | 1.07 inst/cycle | -0.24 | 1.65 inst/cycle | +0.34 | 2.60 inst/cycle | +1.29 | 1.26 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `persistent_tile_scheduling` (1.071×)

Converting a fixed one-CTA-per-tile launch (grid=N_tiles) to a persistent kernel (grid=NUM_SMS) with an inner while-loop that strides over tiles reduces total executed instructions by ~34% (413K→272K) and elapsed cycles by ~6.6% (5868→5481), yielding a 1.07x speedup. The LaunchConfiguration NCU rule fired due to a partial wave (waves_per_sm=1.11), but the primary gain came from instruction reduction rather than eliminating the partial wave — the waves_per_sm metric remained 1.11 after the change, indicating the partial wave was not fully eliminated. The autotune over larger XBLOCK values (256–1024) contributed to the instruction reduction by amortizing per-tile overhead across fewer, larger tiles.

> **Applicable when**: Triton pointwise kernel with waves_per_sm slightly above 1.0 (1.05–1.5), LaunchConfiguration NCU rule fired, low memory and compute throughput (<20% of peak), high instruction count relative to work done. Especially effective when XBLOCK can be increased significantly (e.g. 128→512+) to amortize per-tile overhead.

> **Avoid when**: Kernel already has waves_per_sm close to an integer (no partial wave waste). Kernel is already memory-bandwidth bound (>60% DRAM throughput) — the persistent loop adds control-flow overhead without benefit. Kernel has data dependencies between tiles that prevent independent tile processing.

#### ✅ Plan B: `autotune_xblock_eliminate_partial_wave` (1.071×)

When NCU's LaunchConfiguration rule flags a partial wave (waves_per_sm near an integer + small fraction, e.g. 1.11), replacing a hardcoded small XBLOCK with @triton.autotune over larger tile sizes allows the runtime to select a configuration that reduces the partial-wave tail. The primary observable effect is improved IPC active (1.31→1.65) and issue slot utilization (34.3%→42.6%), with elapsed cycles dropping ~7.5%. Note: in this case waves_per_sm did not change (autotune selected a config that still produced ~1.11 waves), suggesting the speedup came primarily from the autotuner selecting a better num_warps/XBLOCK combination that improved scheduler efficiency rather than eliminating the partial wave entirely.

> **Applicable when**: NCU LaunchConfiguration rule fires with waves_per_sm near a small integer (1.0–2.0); kernel uses a hardcoded XBLOCK tile size; issue slot utilization is below 50%; the kernel is a simple pointwise or strided-gather pattern with low arithmetic intensity; CPIStall rule fires for L1TEX scoreboard stalls.

> **Avoid when**: Kernel already uses autotune or has many waves_per_sm (>4); XBLOCK is constrained by shared memory usage that scales with tile size; the kernel has complex inter-thread dependencies that break with larger tiles; autotuning overhead is unacceptable (e.g., single-use kernels in latency-critical paths).

#### ❌ Plan C: `increase_xblock_with_autotune_to_eliminate_partial_wave` (0.816×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) on a short-duration, memory-bound pointwise kernel caused a 22% regression despite reducing scheduler idle slots (No Eligible: 65.6%→34.0%) and improving occupancy (61.9%→70.9%). The autotune overhead and the 4x increase in total executed instructions (413K→1585K) outweighed any latency-hiding benefit, and the kernel duration increased from 3.94µs to 4.83µs. For very short kernels (<5µs) with small total work, autotune compilation overhead and the extra instruction count from larger tiles can dominate, making this strategy counterproductive even when NCU's LaunchConfiguration rule predicts a 50% gain.

> **Applicable when**: Kernel has many waves per SM (>4) and a large partial-wave tail; kernel duration is long enough (>20µs) that autotune overhead is amortized; total work is large enough that larger tiles reduce grid size meaningfully without inflating instruction count.

> **Avoid when**: Kernel duration is very short (<5µs); total executed instructions are already low (<500K); the NCU LaunchConfiguration rule fires but the kernel is latency-bound rather than occupancy-bound; autotune compilation cost is not amortized across many invocations.

---

## Iteration 2

**Best so far**: 1.0707× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | improve_load_coalescing_via_smaller_xblock_configs | 1.025× | 0.592× | ✓ |  |
| B | streaming_load_eviction_policy_to_reduce_l2_imbalance | 1.047× | 1.054× | ✓ |  |
| C | streaming_cache_eviction_policy_with_increased_warps | 0.990× | 0.940× | ✓ |  |
| D | l2_swizzle_program_id_reorder_for_l2_balance | 0.816× | 0.150× | ✓ |  |


### Metric Comparison

| Metric | Baseline | improve_load_coal... | Δ | streaming_load_ev... | Δ | streaming_cache_e... | Δ | l2_swizzle_progra... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 6660 ns | +2720 ns (+69.0%) | 3740 ns | -200 ns (-5.1%) | 4190 ns | +250 ns (+6.3%) | 26340 ns | +22400 ns (+568.5%) |
| SM Throughput (%) | 13.26 % | 27.52 % | +14.26 pp | 15.58 % | +2.32 pp | 14.24 % | +0.98 pp | 9.06 % | -4.20 pp |
| Memory Throughput (%) | 11.09 % | 7.97 % | -3.12 pp | 10.77 % | -0.32 pp | 10.30 % | -0.79 pp | 12.91 % | +1.82 pp |
| DRAM Throughput (%) | 6.27 % | 3.71 % | -2.56 pp | 6.62 % | +0.35 pp | 5.91 % | -0.36 pp | 0.94 % | -5.33 pp |
| Achieved Occupancy (%) | 61.95 % | 49.34 % | -12.61 pp | 62.01 % | +0.06 pp | 66.82 % | +4.87 pp | 72.04 % | +10.09 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.11 | 0.56 | -0.55 | 1.11 | +0.00 | 1.11 | +0.00 | 4.45 | +3.34 |
| No Eligible (%) | 65.56 % | 56.57 % | -8.99 pp | 48.71 % | -16.85 pp | 64.15 % | -1.41 pp | 88.87 % | +23.31 pp |
| IPC Active | 1.31 inst/cycle | 1.65 inst/cycle | +0.34 | 1.86 inst/cycle | +0.55 | 1.18 inst/cycle | -0.13 | 0.44 inst/cycle | -0.87 |


### Key Learnings

#### ❌ Plan A: `improve_load_coalescing_via_smaller_xblock_configs` (0.592×)

Adding smaller XBLOCK configs (16, 32, 64) to the autotune search space, intended to align tiles with the stride-1 dimension and improve L1TEX coalescing, caused a 1.69x regression (3.68µs → 6.66µs). The autotuner selected XBLOCK=64 (block_size=64), which increased total executed instructions by 5.3x (272K → 1434K) and reduced waves_per_sm from 1.11 to 0.56, leaving the GPU severely underutilized. Although L1 hit rate improved from 0% to 33.3% and warp cycles per issued instruction dropped from 30.7 to 18.2, the massive instruction count inflation from smaller tiles completely dominated, making the kernel far slower. For persistent kernels with an inner tile loop, smaller XBLOCK values multiply the per-tile loop overhead across many more iterations, causing instruction count to explode even when coalescing improves.

> **Applicable when**: Never: this strategy (adding very small XBLOCK configs to autotune) is an anti-pattern for persistent kernels with inner tile loops. The autotuner will select small tiles that inflate instruction count.

> **Avoid when**: Persistent kernels with inner while-loops over tiles: smaller XBLOCK multiplies loop iterations, causing instruction count to explode (5x+ increase observed). Even when L1TEX coalescing improves (0%→33% L1 hit rate), the instruction overhead dominates. Also avoid when waves_per_sm is already below 1.0 — smaller XBLOCK reduces grid size further, worsening GPU utilization (1.11→0.56 waves).

#### ✅ Plan B: `streaming_load_eviction_policy_to_reduce_l2_imbalance` (1.054×)

Adding eviction_policy='evict_first' to a tl.load with a strided gather pattern (0% L1 hit rate, pure streaming access) reduced warp cycles per issued instruction from 30.7 to 20.64 and increased eligible warps per scheduler from 1.73 to 3.04, yielding a 1.05x speedup despite a 61% increase in executed instructions (272K→439K) caused by a simultaneous autotune config change. The evict_first hint (PTX ld.global.cs) signals the cache to treat loaded lines as low-reuse, freeing L2 ways sooner for other in-flight warp requests and reducing L1TEX scoreboard stalls. This strategy is most effective when L1 hit rate is 0%, the CPIStall rule fires for L1TEX scoreboard stalls, and the access pattern is a pure streaming gather with no temporal reuse.

> **Applicable when**: L1/TEX hit rate is 0% (pure streaming gather, no reuse); CPIStall rule fires for L1TEX scoreboard stalls; No Eligible scheduler cycles >50%; kernel is a strided gather or scatter with large non-unit strides; L2 WorkloadImbalance rule fires (one slice significantly busier than average); kernel duration is short (<10µs) and memory-bound.

> **Avoid when**: L1 hit rate is non-zero (data is reused within a tile — evict_first would hurt cache efficiency); kernel is compute-bound rather than memory-latency-bound; access pattern is sequential/coalesced with high L2 hit rate that benefits from caching; the kernel already uses evict_last or other explicit cache policies.

#### ❌ Plan D: `l2_swizzle_program_id_reorder_for_l2_balance` (0.150×)

Replacing a flat linear program_id mapping with a GROUP_SIZE_M swizzle to improve L2 slice balance caused a 6.7x regression (3.68µs → 26.34µs, elapsed cycles 5481 → 39442). Although L1 hit rate improved dramatically (0% → 85%) and L2 hit rate rose (59.8% → 93.8%), the actual memory throughput collapsed from 328 GB/s to 46 GB/s, and warp cycles per issued instruction exploded from 30.7 to 105.3 cycles. The root cause was that the swizzle was applied on top of a persistent kernel that had been replaced with a standard one-CTA-per-tile launch, but the autotune simultaneously selected a much larger XBLOCK (512 vs 128), inflating total executed instructions by 6.8x (272K → 1844K) and creating 4.45 waves per SM instead of 1.11. The CPIStall shifted from L1TEX scoreboard stalls to EXIT drain stalls (39.2% of 105.3 cycles), indicating the larger tiles caused massive store-drain latency. The swizzle arithmetic itself added no measurable benefit while the combined effect of larger XBLOCK and more waves dominated runtime.

> **Applicable when**: Never demonstrated to help for short pointwise kernels (<5µs); the WorkloadImbalance NCU rule for L2 slices does not reliably predict that a PID swizzle will improve performance for 1D pointwise kernels.

> **Avoid when**: Short-duration pointwise kernels (<5µs, <10K elapsed cycles); kernels where the swizzle is combined with autotune that may simultaneously select a much larger XBLOCK, inflating instruction count; kernels where the primary stall is already L1TEX scoreboard (CPIStall) — swizzle does not address this and may worsen EXIT drain stalls by increasing per-block work; 1D kernels where the swizzle identity (pid_m * GROUP_SIZE_M + pid_n == pid) produces no actual reordering.

---

## Iteration 3

**Best so far**: 1.0707× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_to_autotune_configs | 1.019× | 1.079× | ✓ |  |
| B | autotune_num_stages_software_pipeline_l1tex_latency | 1.020× | 0.970× | ✓ |  |
| C | fix_broken_autotune_grid_and_increase_num_warps_for_latency_hiding | 0.995× | 1.026× | ✓ |  |
| D | remove_swizzle_overhead_and_reduce_xblock_to_restore_baseline | 0.999× | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | autotune_num_stag... | Δ | fix_broken_autotu... | Δ | remove_swizzle_ov... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3650 ns | -290 ns (-7.4%) | 4060 ns | +120 ns (+3.0%) | 3840 ns | -100 ns (-2.5%) | 3970 ns | +30 ns (+0.8%) |
| SM Throughput (%) | 13.26 % | 15.55 % | +2.29 pp | 14.46 % | +1.20 pp | 13.57 % | +0.31 pp | 13.91 % | +0.65 pp |
| Memory Throughput (%) | 11.09 % | 11.04 % | -0.05 pp | 10.98 % | -0.11 pp | 6.46 % | -4.63 pp | 11.14 % | +0.05 pp |
| DRAM Throughput (%) | 6.27 % | 6.76 % | +0.49 pp | 6.13 % | -0.14 pp | 3.25 % | -3.02 pp | 6.27 % | +0.00 pp |
| Achieved Occupancy (%) | 61.95 % | 60.63 % | -1.32 pp | 63.40 % | +1.45 pp | 60.59 % | -1.36 pp | 59.16 % | -2.79 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1.11 | 1.11 | +0.00 | 1.11 | +0.00 | 1.11 | +0.00 | 1.11 | +0.00 |
| No Eligible (%) | 65.56 % | 49.98 % | -15.58 pp | 63.68 % | -1.88 pp | 65.43 % | -0.13 pp | 62.98 % | -2.58 pp |
| IPC Active | 1.31 inst/cycle | 1.79 inst/cycle | +0.48 | 1.37 inst/cycle | +0.06 | 1.34 inst/cycle | +0.03 | 1.28 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan A: `add_num_stages_software_pipelining_to_autotune_configs` (1.079×)

Adding num_stages=2 variants alongside num_stages=1 configs in a Triton @triton.autotune decorator for a short pointwise/gather kernel yielded a modest 1.08x speedup (3.74µs → 3.65µs, elapsed cycles 5567 → 5449). The improvement is small and the per-metric changes are marginal — No Eligible scheduler cycles barely changed (48.71% → 49.98%), eligible warps per scheduler slightly decreased (3.04 → 2.97), and warp cycles per issued instruction increased slightly (20.64 → 22.88). The speedup likely comes from the autotuner selecting a slightly better XBLOCK/num_warps combination rather than from software pipelining itself, as num_stages has limited effect on simple load-store kernels without explicit loops that the compiler can pipeline. The strategy is low-risk (correctness-safe) but yields only marginal gains for short pointwise kernels.

> **Applicable when**: Triton kernel with IssueSlotUtilization rule firing (No Eligible >40%); kernel already uses @triton.autotune but configs do not include num_stages variants; kernel has low register pressure (<=16 registers/thread) leaving headroom for pipelining state; kernel is memory-latency bound with 0% L1 hit rate; adding num_stages variants is low-risk since the autotuner will select the best option.

> **Avoid when**: Kernel is a simple single-load-store pointwise with no explicit inner loop — num_stages has minimal effect since the compiler cannot pipeline across iterations that don't exist; when scheduler stall metrics (No Eligible, eligible warps) do not improve after the change, indicating the compiler ignored the hint; when register pressure is already high (>32 registers/thread) and num_stages=2 would spill registers.

---

## Cumulative Progress

```
Baseline:  3940 ns  (1.000×)
Iter 1     3680 ns  (1.071×)  — autotune_xblock_eliminate_partial_wave

Final best: 3680 ns  (1.0707×)
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
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_unfold_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 301056
@@ -16,17 +29,16 @@

     x2 = ((xindex // 256) % 14)
     x3 = xindex // 3584
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 224*x1 + 3584*x3), xmask)
+    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 224*x1 + 3584*x3), xmask,
+                   eviction_policy='evict_first')  # streaming gather, no L1 reuse
     tl.store(out_ptr0 + (x4), tmp0, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 301056
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_unfold_0[grid](arg_0, arg_1, 301056, XBLOCK=128)
+    triton_poi_fused_clone_unfold_0[grid](arg_0, arg_1, 301056)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
