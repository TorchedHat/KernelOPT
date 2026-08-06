# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 2850 ns (**1.0877×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`rightsize_tile_blocks_to_fill_gpu`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_6_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 28
- **Duration**: 3100 ns
- **SM Throughput**: 15.60 % | **Memory**: 14.70 % | **DRAM**: 1.97 %
- **Achieved Occupancy**: 36.90 %
- **IPC Active**: 1.70 inst/cycle | **No Eligible**: 54.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rightsize_tile_blocks_to_fill_gpu | — | 1.088× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_small_xnumel | — | 1.211× | ✓ |  |
| C | fix_tile_dimensions_for_small_xnumel | — | 1.010× | ✓ |  |
| D | autotune_tile_sizes_for_small_xnumel_transpose | — | 1.140× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rightsize_tile_bl... | Δ | autotune_tile_siz... | Δ | fix_tile_dimensio... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2850 ns | -250 ns (-8.1%) | 2560 ns | -540 ns (-17.4%) | 3070 ns | -30 ns (-1.0%) | 2720 ns | -380 ns (-12.3%) |
| SM Throughput (%) | 15.60 % | 11.16 % | -4.44 pp | 6.22 % | -9.38 pp | 13.76 % | -1.84 pp | 6.60 % | -9.00 pp |
| Memory Throughput (%) | 14.70 % | 10.35 % | -4.35 pp | 2.40 % | -12.30 pp | 6.85 % | -7.85 pp | 5.87 % | -8.83 pp |
| DRAM Throughput (%) | 1.97 % | 2.16 % | +0.19 pp | 2.40 % | +0.43 pp | 2.00 % | +0.03 pp | 2.26 % | +0.29 pp |
| Achieved Occupancy (%) | 36.90 % | 16.22 % | -20.68 pp | 25.01 % | -11.89 pp | 34.31 % | -2.59 pp | 16.35 % | -20.55 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 16.00 register/thread | -12 | 28.00 register/thread | +0 | 21.00 register/thread | -7 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.48 | +0.00 | 0.48 | +0.00 | 0.24 | -0.24 |
| No Eligible (%) | 54.69 % | 61.05 % | +6.36 pp | 76.41 % | +21.72 pp | 42.21 % | -12.48 pp | 75.69 % | +21.00 pp |
| IPC Active | 1.70 inst/cycle | 1.20 inst/cycle | -0.50 | 0.84 inst/cycle | -0.86 | 1.78 inst/cycle | +0.08 | 0.68 inst/cycle | -1.02 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_small_xnumel` (1.183×)

When a 2D Triton kernel has a very small x-dimension (xnumel=9) but all autotune configs use XBLOCK >= 128, the vast majority of x-threads are masked out, causing extreme register pressure (92 registers/thread) that caps Block Limit Registers at 5 blocks/SM and theoretical occupancy at 31.25%. Replacing all configs with XBLOCK in {1, 16} and YBLOCK in {64–512} reduces registers/thread from 92 to 16, raises Block Limit Registers from 5 to 64, eliminates dynamic shared memory (8KB → 0), and cuts total executed instructions by ~50x (2.73M → 54K), yielding a 1.18x speedup (7.68 µs → 2.62 µs). The causal chain is: oversized XBLOCK → masked threads inflate register pressure → register pressure limits occupancy and blocks/SM → fewer eligible warps → high scheduler idle rate → long kernel duration.

> **Applicable when**: 2D Triton pointwise/permutation kernel where xnumel is much smaller than XBLOCK (e.g., xnumel <= 16 but XBLOCK >= 128); registers_per_thread is high (>40) due to large tile waste; Block Limit Registers is the binding occupancy constraint; IssueSlotUtilization No-Eligible > 40%; TheoreticalOccupancy < 50%; LaunchConfiguration rule triggered with partial-wave tail effect; MemoryCacheAccessPattern rule triggered with low bytes-utilized-per-sector

> **Avoid when**: xnumel is large and close to XBLOCK (tile is well-utilized); kernel is already register-pressure-limited for other reasons unrelated to tile size; grid is already very small (waves_per_sm < 0.1) and adding more CTAs would not help; kernel has complex inter-thread dependencies requiring specific XBLOCK values

---

## Iteration 2

**Best so far**: 1.0877× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | — | 0.012× | ✓ |  |
| B | persistent_tile_scheduling | — | 0.004× | ✓ |  |
| C | persistent_tile_scheduling_via_grid_stride_loop | — | 0.017× | ✓ |  |
| D | persistent_tile_scheduling_for_small_grid | — | 0.012× | ✓ |  |
| E | add_triton_autotune | — | 0.404× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 265150 ns | +262050 ns (+8453.2%) | 750660 ns | +747560 ns (+24114.8%) | 187780 ns | +184680 ns (+5957.4%) | 265410 ns | +262310 ns (+8461.6%) | 7680 ns | +4580 ns (+147.7%) |
| SM Throughput (%) | 15.60 % | 62.11 % | +46.51 pp | 59.57 % | +43.97 pp | 30.12 % | +14.52 pp | 29.55 % | +13.95 pp | 44.92 % | +29.32 pp |
| Memory Throughput (%) | 14.70 % | 22.71 % | +8.01 pp | 70.71 % | +56.01 pp | 43.66 % | +28.96 pp | 48.02 % | +33.32 pp | 57.25 % | +42.55 pp |
| DRAM Throughput (%) | 1.97 % | 0.03 % | -1.94 pp | 0.01 % | -1.96 pp | 0.04 % | -1.93 pp | 0.03 % | -1.94 pp | 0.82 % | -1.15 pp |
| Achieved Occupancy (%) | 36.90 % | 39.23 % | +2.33 pp | 48.04 % | +11.14 pp | 24.24 % | -12.66 pp | 48.17 % | +11.27 pp | 25.52 % | -11.38 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 31.00 register/thread | +3 | 16.00 register/thread | -12 | 92.00 register/thread | +64 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.24 | -0.24 | 0.48 | +0.00 | 1.55 | +1.07 |
| No Eligible (%) | 54.69 % | 36.85 % | -17.84 pp | 39.12 % | -15.57 pp | 67.87 % | +13.18 pp | 71.69 % | +17.00 pp | 40.19 % | -14.50 pp |
| IPC Active | 1.70 inst/cycle | 2.51 inst/cycle | +0.81 | 2.48 inst/cycle | +0.78 | 1.29 inst/cycle | -0.41 | 1.15 inst/cycle | -0.55 | 2.33 inst/cycle | +0.63 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.012×)

Converting a one-CTA-per-tile dispatch to a persistent kernel with a while loop over tiles caused a ~85x slowdown (2.56 µs → 265 µs) despite improving SM compute throughput (6% → 62%) and occupancy (25% → 39%). The causal mechanism is that the persistent loop's per-iteration integer divmod overhead (tile_id // num_x_tiles, tile_id % num_x_tiles) and loop control instructions inflated total executed instructions by ~1063x (115K → 122.7M), completely overwhelming any tail-wave savings. For extremely short kernels (duration < ~10 µs, elapsed cycles < ~10K), the persistent loop's instruction overhead per tile far exceeds the cost of the original tail-wave inefficiency, making this approach strictly harmful.

> **Applicable when**: Never beneficial for kernels with duration < ~10 µs or elapsed cycles < ~10K cycles; the persistent loop overhead always dominates at this scale.

> **Avoid when**: Kernel duration is very short (< ~10 µs); total work per CTA is tiny (few tiles per CTA); the per-tile loop overhead (divmod, branch, index arithmetic) exceeds the tail-wave savings. Do NOT apply persistent scheduling when waves_per_sm < 1 but the kernel is already completing in microseconds — the tail-wave cost is negligible compared to loop instruction overhead.

#### ❌ Plan B: `persistent_tile_scheduling` (0.012×)

Converting a one-CTA-per-tile dispatch to a persistent kernel with a while loop over tiles caused a ~85x slowdown (2.56 µs → 265 µs) despite improving SM compute throughput (6% → 62%) and occupancy (25% → 39%). The causal mechanism is that the persistent loop's per-iteration integer divmod overhead (tile_id // num_x_tiles, tile_id % num_x_tiles) and loop control instructions inflated total executed instructions by ~1063x (115K → 122.7M), completely overwhelming any tail-wave savings. For extremely short kernels (duration < ~10 µs, elapsed cycles < ~10K), the persistent loop's instruction overhead per tile far exceeds the cost of the original tail-wave inefficiency, making this approach strictly harmful.

> **Applicable when**: Never beneficial for kernels with duration < ~10 µs or elapsed cycles < ~10K cycles; the persistent loop overhead always dominates at this scale.

> **Avoid when**: Kernel duration is very short (< ~10 µs); total work per CTA is tiny (few tiles per CTA); the per-tile loop overhead (divmod, branch, index arithmetic) exceeds the tail-wave savings. Do NOT apply persistent scheduling when waves_per_sm < 1 but the kernel is already completing in microseconds — the tail-wave cost is negligible compared to loop instruction overhead.

#### ❌ Plan C: `persistent_tile_scheduling_via_grid_stride_loop` (0.017×)

Converting a small-grid 2D Triton kernel to a persistent 1D grid-stride loop caused a catastrophic 60x regression (2.56 µs → 187.78 µs). The root cause is that the grid-stride while loop with dynamic tile decomposition (integer div/mod inside the loop body) generated ~396x more executed instructions (115K → 45.7M), because the autotuner selected a config with very small tiles (e.g., XBLOCK=1, YBLOCK=128) causing each CTA to iterate over hundreds of tiles in the while loop, each iteration paying the full div/mod overhead. Additionally, registers per thread nearly doubled (16 → 31) due to loop-carried variables and the div/mod temporaries, and dynamic shared memory was introduced (0 → 2KB/block), further reducing occupancy. The CPIStall rule triggered with 57.4% of cycles stalled on L1TEX scoreboard dependencies, indicating the loop body's repeated memory accesses created long dependency chains that the scheduler could not hide.

> **Applicable when**: Never: this pattern is an anti-pattern for small-work kernels where the autotuner may select small tile configs, causing the grid-stride loop to iterate hundreds of times per CTA with expensive div/mod decomposition overhead.

> **Avoid when**: Any kernel where: (1) total_tiles can greatly exceed the capped grid size under some autotuner configs, causing many loop iterations per CTA; (2) the kernel body is lightweight (simple load/store permutation) so the loop overhead dominates; (3) registers per thread are already low and the loop-carried variables would double register pressure; (4) the kernel is already short-lived (<10 µs) where the instruction explosion from many loop iterations far outweighs any SM-fill benefit.

#### ❌ Plan D: `persistent_tile_scheduling_for_small_grid` (0.012×)

Converting a small 2D tiled Triton kernel to a persistent 1D flat-index loop with an internal while-loop caused a catastrophic 85x slowdown (2.56 µs → 265.41 µs). The persistent loop's while-loop body introduced a modulo and integer division by a non-power-of-2 constant (9) inside the hot loop, which Triton/PTX cannot optimize to a multiply-shift sequence, resulting in executed instructions exploding from 115K to 59M (512x increase). The CPIStall rule appeared post-change (76.9% of cycles stalled on L1TEX scoreboard), and Warp Cycles Per Issued Instruction worsened from 16.89 to 27.21, indicating the persistent loop's repeated modulo/div by 9 created severe instruction-level serialization that overwhelmed any wave-fragmentation benefit.

> **Applicable when**: Never: this strategy caused an 85x regression. Do not apply persistent while-loop scheduling to small Triton kernels that require non-power-of-2 modulo/division inside the loop body.

> **Avoid when**: Any kernel where the flat-index decomposition requires modulo or integer division by a non-power-of-2 constant (e.g., xnumel=9) inside the persistent loop body. The cost of repeated non-power-of-2 integer division in PTX far exceeds any wave-fragmentation savings for small kernels. Also avoid when the kernel duration is already in the low-microsecond range (< 10 µs) where launch overhead is negligible relative to computation.

---

## Iteration 3

**Best so far**: 1.0877× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_configs_to_match_tiny_xnumel_and_reduce_register_pressure | — | 1.183× | ✓ |  |
| B | revert_persistent_grid_stride_loop_to_clean_2d_grid | — | 1.065× | ✓ |  |
| D | revert_persistent_loop_and_fix_uncoalesced_access_via_xblock_rightsizing | — | 0.951× | ✓ |  |
| E | add_triton_autotune | — | 1.183× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_conf... | Δ | revert_persistent... | Δ | revert_persistent... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2620 ns | -480 ns (-15.5%) | 2910 ns | -190 ns (-6.1%) | 3260 ns | +160 ns (+5.2%) | 2620 ns | -480 ns (-15.5%) |
| SM Throughput (%) | 15.60 % | 3.08 % | -12.52 pp | 10.84 % | -4.76 pp | 16.70 % | +1.10 pp | 6.39 % | -9.21 pp |
| Memory Throughput (%) | 14.70 % | 2.34 % | -12.36 pp | 10.05 % | -4.65 pp | 15.69 % | +0.99 pp | 2.36 % | -12.34 pp |
| DRAM Throughput (%) | 1.97 % | 2.34 % | +0.37 pp | 2.12 % | +0.15 pp | 1.89 % | -0.08 pp | 2.36 % | +0.39 pp |
| Achieved Occupancy (%) | 36.90 % | 11.32 % | -25.58 pp | 16.39 % | -20.51 pp | 34.31 % | -2.59 pp | 25.29 % | -11.61 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 28.00 register/thread | +0 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.24 | -0.24 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 54.69 % | 84.22 % | +29.53 pp | 63.64 % | +8.95 pp | 56.64 % | +1.95 pp | 80.03 % | +25.34 pp |
| IPC Active | 1.70 inst/cycle | 0.34 inst/cycle | -1.36 | 1.29 inst/cycle | -0.41 | 1.54 inst/cycle | -0.16 | 0.83 inst/cycle | -0.87 |


### Key Learnings

#### ✅ Plan B: `revert_persistent_grid_stride_loop_to_clean_2d_grid` (1.065×)

When a Triton kernel has been incorrectly converted to a persistent grid-stride while-loop, reverting to a clean 2D program_id-indexed dispatch eliminates the per-iteration div/mod overhead and loop-carried register pressure. In this case, executed instructions dropped from 2.73M to 238K (11.5x), registers_per_thread fell from 92 to 30, and dynamic shared memory per block dropped from 8KB to 2KB, reducing kernel duration from 7.68 µs to 2.91 µs (2.6x). The theoretical occupancy improved from 31% to 100% because the register bottleneck was removed, even though achieved occupancy fell due to fewer waves (1.55 → 0.24) — the net effect was still a 1.07x speedup because the instruction count reduction dominated.

> **Applicable when**: A Triton kernel has been converted to a persistent while-loop and shows: registers_per_thread >> 30 (e.g., 60-92), dynamic_smem_per_block >> expected, executed instructions >> expected for the workload size, TheoreticalOccupancy limited by registers (Block Limit Registers << Block Limit Warps), and the kernel body is a lightweight operation (load/store permutation, elementwise) where loop overhead dominates.

> **Avoid when**: The kernel is genuinely compute-heavy per tile (matmul, attention) where loop overhead is negligible relative to tile work; or when the persistent loop was correctly implemented with a fixed NUM_SMS grid and pre-computed tile counts passed as kernel args (not computed via tl.cdiv inside the kernel body).

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)
Iter 1     2850 ns  (1.088×)  — rightsize_tile_blocks_to_fill_gpu

Final best: 2850 ns  (1.0877×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=4),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 16},  num_warps=8),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 8},  num_warps=8),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
@@ -22,14 +29,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 8192
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6[grid](arg_0, arg_1, 8192, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6[grid](arg_0, arg_1, 8192, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
