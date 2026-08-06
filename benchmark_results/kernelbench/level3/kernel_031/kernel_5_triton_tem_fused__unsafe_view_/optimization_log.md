# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 662980 ns — **Best latency**: 662980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_5_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 99.30
- **Registers/thread**: 95
- **Duration**: 662980 ns
- **SM Throughput**: 67.43 % | **Memory**: 88.49 % | **DRAM**: 32.69 %
- **Achieved Occupancy**: 30.23 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 32.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_to_reduce_l1_pressure_and_improve_reuse | 0.234× | 0.235× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 662980 ns | 2820000 ns | +2157020 ns (+325.4%) |
| SM Throughput (%) | 67.43 % | 57.88 % | -9.55 pp |
| Memory Throughput (%) | 88.49 % | 53.61 % | -34.88 pp |
| DRAM Throughput (%) | 32.69 % | 8.21 % | -24.48 pp |
| Achieved Occupancy (%) | 30.23 % | 12.37 % | -17.86 pp |
| Registers/Thread | 95.00 register/thread | 255.00 register/thread | +160 |
| Waves/SM | 99.30 | 248.24 | +148.94 |
| No Eligible (%) | 32.25 % | 41.72 % | +9.47 pp |
| IPC Active | 2.71 inst/cycle | 2.33 inst/cycle | -0.38 |


### Key Learnings

#### ❌ Plan A: `autotune_tile_sizes_to_reduce_l1_pressure_and_improve_reuse` (0.235×)

Increasing BLOCK_M and BLOCK_N from 64 to 128 (while keeping the grid fixed at the original 65536 CTAs instead of reducing it) caused registers per thread to jump from 95 to 255, collapsing theoretical occupancy from 31.25% to 12.5% and triggering massive register spilling (MemoryCacheAccessPattern rule: only 1 of 32 bytes per sector utilized for local loads/stores, 51.93% potential speedup lost). The executed instruction count grew 3.7x (348M to 1284M) and duration increased 4.25x (663 us to 2.82 ms). The critical implementation flaw was that the grid was NOT updated to match the larger tile sizes — the grid remained at 65536 CTAs, so each CTA still processed the same output region but with a larger register footprint, causing spill. Even if the grid had been corrected, adding @triton.autotune with only num_warps/num_stages variation (no BLOCK_M/BLOCK_N in configs) while hardcoding larger tiles is an anti-pattern: the autotuner cannot explore tile sizes and the larger tiles simply increase register pressure without reducing CTA count.

> **Applicable when**: Never apply this specific pattern (larger tiles with unchanged grid). The general direction of tile-size autotuning is applicable when L1/TEX throughput > 80%, registers_per_thread < 128, and theoretical occupancy is register-limited.

> **Avoid when**: Do not increase BLOCK_M/BLOCK_N without simultaneously updating the grid to cdiv(M, BLOCK_M)*cdiv(N, BLOCK_N). Do not use @triton.autotune with only num_warps/num_stages variation when the tile sizes are hardcoded to larger values — this gives the autotuner no tile-size freedom while still incurring the register cost. Avoid when registers_per_thread is already near 128+ as larger tiles will push into spill territory (255 regs = max, causes spill).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_size_and_autotune_block_tiles_for_bmm | 0.999× | 1.009× | ✓ |  |
| B | fix_grid_size_and_autotune_block_mn_to_eliminate_register_spill | 0.994× | 1.013× | ✓ |  |
| C | fix_grid_tile_mismatch_and_add_vectorized_stores | 0.235× | 0.236× | ✓ |  |
| D | fix_grid_to_match_hardcoded_128x128_tile_sizes_eliminating_register_spill | 0.234× | 0.237× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_size_and... | Δ | fix_grid_size_and... | Δ | fix_grid_tile_mis... | Δ | fix_grid_to_match... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 662980 ns | 657280 ns | -5700 ns (-0.9%) | 654400 ns | -8580 ns (-1.3%) | 2810000 ns | +2147020 ns (+323.8%) | 2800000 ns | +2137020 ns (+322.3%) |
| SM Throughput (%) | 67.43 % | 67.42 % | -0.01 pp | 67.43 % | +0.00 pp | 58.04 % | -9.39 pp | 57.90 % | -9.53 pp |
| Memory Throughput (%) | 88.49 % | 88.48 % | -0.01 pp | 88.49 % | +0.00 pp | 53.98 % | -34.51 pp | 53.60 % | -34.89 pp |
| DRAM Throughput (%) | 32.69 % | 32.98 % | +0.29 pp | 33.13 % | +0.44 pp | 8.24 % | -24.45 pp | 8.26 % | -24.43 pp |
| Achieved Occupancy (%) | 30.23 % | 30.24 % | +0.01 pp | 30.24 % | +0.01 pp | 12.38 % | -17.85 pp | 12.37 % | -17.86 pp |
| Registers/Thread | 95.00 register/thread | 95.00 register/thread | +0 | 95.00 register/thread | +0 | 255.00 register/thread | +160 | 255.00 register/thread | +160 |
| Waves/SM | 99.30 | 99.30 | +0.00 | 99.30 | +0.00 | 248.24 | +148.94 | 248.24 | +148.94 |
| No Eligible (%) | 32.25 % | 32.26 % | +0.01 pp | 32.25 % | +0.00 pp | 41.51 % | +9.26 pp | 41.61 % | +9.36 pp |
| IPC Active | 2.71 inst/cycle | 2.71 inst/cycle | +0.00 | 2.71 inst/cycle | +0.00 | 2.34 inst/cycle | -0.37 | 2.33 inst/cycle | -0.38 |


### Key Learnings

#### ❌ Plan C: `fix_grid_tile_mismatch_and_add_vectorized_stores` (0.236×)

Adding tl.max_contiguous and tl.multiple_of hints to the store index produced no measurable improvement (speedup 0.2359x vs 0.2351x baseline — essentially identical). All key metrics remained unchanged: registers_per_thread stayed at 255, theoretical occupancy at 12.5%, duration at ~2.81-2.82 ms, and the MemoryCacheAccessPattern rule still fired with 1/32 bytes per sector for local loads/stores. The root cause — register spill from an oversized 128x128 accumulator tile with a grid that was not corrected to match — was not addressed by the store hint change alone. When a kernel is in a severe register-spill regime (255 regs/thread, Block Limit Registers=2), cosmetic epilogue changes like contiguity hints have no effect because the bottleneck is the spill traffic from the accumulator, not the final output store pattern.

> **Applicable when**: Never — this specific combination (store hint without fixing the grid) is confirmed ineffective. The general store-hint technique may be applicable when the kernel is NOT in a register-spill regime and the output store is the actual bottleneck.

> **Avoid when**: Do not apply tl.max_contiguous/tl.multiple_of store hints as a standalone fix when the kernel has registers_per_thread=255 and MemoryCacheAccessPattern fires for local loads (not global stores). The local load/store pattern in NCU indicates register spill to local memory — this is fundamentally different from uncoalesced global memory stores and cannot be fixed with store contiguity hints. The grid mismatch (65536 CTAs for 128x128 tiles over a 16384x16384 output) must be fixed first.

#### ❌ Plan D: `fix_grid_to_match_hardcoded_128x128_tile_sizes_eliminating_register_spill` (0.237×)

The plan to fix the grid tuple from (65536,1,1) to (16384,8,1) to match 128x128 tile sizes produced no measurable change: the diff was empty and all profiling metrics remained identical (255 registers/thread, 12.5% theoretical occupancy, 2.8 ms duration, 1/32 bytes/sector local load utilization). The code change was not actually applied — the diff shows no modifications — meaning the grid fix was either not implemented or had no effect on the compiled kernel. This confirms that the underlying register spill problem (registers_per_thread=255, MemoryCacheAccessPattern local load/store utilization=1/32 bytes/sector) persists and cannot be resolved without actually modifying the kernel tile sizes or register pressure, not just the launch grid.

> **Applicable when**: Never: this specific plan (grid-only fix) was not applied and produced no effect. The general grid-correction approach is theoretically valid but requires the code change to actually land.

> **Avoid when**: Do not attempt grid-only fixes when the diff is empty — verify the change was actually applied before attributing any metric movement. Do not expect grid changes alone to reduce registers_per_thread; register count is determined at compile time by tile sizes and kernel body, not by the launch grid. When registers_per_thread=255 and MemoryCacheAccessPattern shows 1/32 bytes/sector for local loads, the fix must reduce the accumulator tile size (BLOCK_M, BLOCK_N) inside the kernel body, not just adjust the grid.

---

## Cumulative Progress

```
Baseline:  662980 ns  (1.000×)

Final best: 662980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=16, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 64
+    BLOCK_M : tl.constexpr = 128
+    BLOCK_N : tl.constexpr = 128
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -86,8 +99,6 @@

     xindex = idx_n + 16384*idx_m + 268435456*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
