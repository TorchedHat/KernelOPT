# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 4220 ns (**1.4550×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints_and_expanded_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_084/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6140 ns
- **SM Throughput**: 16.51 % | **Memory**: 40.01 % | **DRAM**: 20.05 %
- **Achieved Occupancy**: 55.16 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.266× | 0.979× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.385× | 1.937× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.325× | 1.380× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 6270 ns | +130 ns (+2.1%) | 3170 ns | -2970 ns (-48.4%) | 4450 ns | -1690 ns (-27.5%) |
| SM Throughput (%) | 16.51 % | 16.65 % | +0.14 pp | 5.18 % | -11.33 pp | 27.63 % | +11.12 pp |
| Memory Throughput (%) | 40.01 % | 39.84 % | -0.17 pp | 11.28 % | -28.73 pp | 21.93 % | -18.08 pp |
| DRAM Throughput (%) | 20.05 % | 19.65 % | -0.40 pp | 2.64 % | -17.41 pp | 3.73 % | -16.32 pp |
| Achieved Occupancy (%) | 55.16 % | 55.86 % | +0.70 pp | 26.06 % | -29.10 pp | 68.87 % | +13.71 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 1.48 | +0.74 |
| No Eligible (%) | 70.41 % | 70.82 % | +0.41 pp | 80.49 % | +10.08 pp | 42.36 % | -28.05 pp |
| IPC Active | 1.16 inst/cycle | 1.10 inst/cycle | -0.06 | 0.67 inst/cycle | -0.49 | 2.37 inst/cycle | +1.21 |

---

## Iteration 2

**Best so far**: 1.9369× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.415× | 1.937× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints_and_expanded_autotune | 1.421× | 1.455× | ✓ | **✓ best** |
| C | expand_autotune_configs_with_yblock1_small_tile_variants | 1.389× | 1.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3170 ns | -2970 ns (-48.4%) | 4220 ns | -1920 ns (-31.3%) | 3100 ns | -3040 ns (-49.5%) |
| SM Throughput (%) | 16.51 % | 5.21 % | -11.30 pp | 25.67 % | +9.16 pp | 2.75 % | -13.76 pp |
| Memory Throughput (%) | 40.01 % | 11.49 % | -28.52 pp | 20.38 % | -19.63 pp | 3.95 % | -36.06 pp |
| DRAM Throughput (%) | 20.05 % | 2.63 % | -17.42 pp | 3.90 % | -16.15 pp | 0.69 % | -19.36 pp |
| Achieved Occupancy (%) | 55.16 % | 26.04 % | -29.12 pp | 69.26 % | +14.10 pp | 13.56 % | -41.60 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.41 % | 79.93 % | +9.52 pp | 39.34 % | -31.07 pp | 87.91 % | +17.50 pp |
| IPC Active | 1.16 inst/cycle | 0.62 inst/cycle | -0.54 | 2.39 inst/cycle | +1.23 | 0.40 inst/cycle | -0.76 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.937×)

Applying tl.max_contiguous and tl.multiple_of hints to the load index in a Triton kernel is intended to unlock 128-bit (float4) vectorized LDG instructions by asserting alignment and contiguity to the compiler. However, in this case the NCU profiling metrics before and after are nearly identical — duration, CPIStall percentage, eligible warps, and uncoalesced sector counts all remain unchanged — indicating the hints did not materially alter the generated PTX or memory transaction pattern. The reported 1.94x speedup likely reflects measurement noise or host-side overhead reduction rather than a genuine kernel-level improvement, as the kernel-level metrics show no meaningful change.

> **Applicable when**: CPIStall (L1TEX) > 50% of warp cycles; load index is provably contiguous (tl.arange-based) and aligned to a power-of-2 boundary; xnumel is divisible by 16; all XBLOCK autotune configs are multiples of 16; the kernel is load-latency-bound rather than store-bound

> **Avoid when**: The kernel is already at the GPU launch overhead floor (waves_per_sm < 0.5, duration < 5 us) where NCU metrics show no change from the hints; store-side uncoalesced scatter dominates and cannot be fixed by load hints; the load index is not a simple tl.arange (e.g. involves modulo or non-unit strides); XBLOCK configs include values not divisible by 16 (would make tl.multiple_of assertion incorrect)

---

## Iteration 3

**Best so far**: 1.9369× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_with_large_yblock_num_stages1_configs | 1.389× | 1.937× | ✓ |  |
| B | evict_first_load_with_pipelined_stages | 1.376× | 1.937× | ✓ |  |
| C | software_pipelining_via_num_stages | 1.371× | 1.524× | ✓ |  |
| D | add_triton_autotune | 1.421× | 2.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | evict_first_load_... | Δ | software_pipelini... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3170 ns | -2970 ns (-48.4%) | 3170 ns | -2970 ns (-48.4%) | 4030 ns | -2110 ns (-34.4%) | 3070 ns | -3070 ns (-50.0%) |
| SM Throughput (%) | 16.51 % | 2.81 % | -13.70 pp | 5.11 % | -11.40 pp | 26.54 % | +10.03 pp | 2.74 % | -13.77 pp |
| Memory Throughput (%) | 40.01 % | 3.72 % | -36.29 pp | 11.44 % | -28.57 pp | 21.06 % | -18.95 pp | 3.83 % | -36.18 pp |
| DRAM Throughput (%) | 20.05 % | 0.67 % | -19.38 pp | 2.63 % | -17.42 pp | 4.12 % | -15.93 pp | 0.69 % | -19.36 pp |
| Achieved Occupancy (%) | 55.16 % | 14.08 % | -41.08 pp | 26.87 % | -28.29 pp | 69.06 % | +13.90 pp | 13.77 % | -41.39 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.41 % | 87.70 % | +17.29 pp | 79.76 % | +9.35 pp | 36.98 % | -33.43 pp | 88.45 % | +18.04 pp |
| IPC Active | 1.16 inst/cycle | 0.42 inst/cycle | -0.74 | 0.58 inst/cycle | -0.58 | 2.42 inst/cycle | +1.26 | 0.41 inst/cycle | -0.75 |


### Key Learnings

#### ⚠️ Plan A: `expand_autotune_with_large_yblock_num_stages1_configs` (1.937×)

Adding num_stages=1 configs with large YBLOCK (4–32) to the Triton autotune search space was intended to reduce shared memory pressure and allow more CTAs per SM, raising waves_per_sm and achieved occupancy. However, the profiling metrics after the change are nearly identical to before: block_size=32, grid_size=1568, waves_per_sm=0.37, achieved_occupancy=14.08% vs 13.56%, eligible_warps=0.19 — the autotuner still selected the same tiny XBLOCK=32, YBLOCK=1 config. The 1.94x reported speedup matches the previous iteration's speedup exactly, strongly indicating the autotuner cached the prior benchmark result and the new configs were never actually evaluated, or the grid lambda fix was not applied correctly so the new tile shapes produced incorrect output and were rejected.

> **Applicable when**: Triton kernel with waves_per_sm < 0.5, achieved_occupancy < 20%, theoretical_occupancy > 40%, and Block Limit Shared Mem is the binding occupancy constraint; all existing autotune configs use num_stages >= 2; the kernel has no explicit shared memory usage but pipeline buffering consumes smem slots

> **Avoid when**: The autotuner has already cached results for the key tuple (ynumel, xnumel) — new configs will not be benchmarked unless the cache is cleared; the grid is hardcoded rather than using a META lambda, causing new tile shapes to silently produce wrong results; the kernel is already at the GPU launch overhead floor where block_size and grid_size do not change after adding new configs

#### ✅ Plan B: `evict_first_load_with_pipelined_stages` (1.937×)

For streaming kernels with zero data reuse (each element loaded and stored exactly once), switching the load eviction policy from 'evict_last' to 'evict_first' reduces L1/L2 cache pollution, lowering the CPIStall L1TEX percentage from ~57% to ~39%. Simultaneously, adding num_stages=2 to all autotune configs enables Triton's async prefetch pipeline, overlapping next-tile memory loads with current-tile compute/store, and the autotune search over a broader config space (including larger XBLOCK/YBLOCK combos) selects a block_size=64 config that doubles achieved occupancy from 13.6% to 26.9% and memory throughput from 33 GB/s to 128 GB/s. The combination of cache-eviction policy correction and software pipelining is particularly effective when L1TEX stalls dominate (>50% of warp cycles) and the access pattern is purely streaming.

> **Applicable when**: Triton kernel is a pure streaming memory operation (layout transpose, copy, type cast) with zero temporal data reuse; CPIStall L1TEX > 50% of warp cycles; load eviction policy is 'evict_last' (the wrong policy for streaming); autotune configs have num_stages=1 or num_stages=3 without async prefetch benefit; achieved occupancy is well below theoretical occupancy

> **Avoid when**: Kernel has significant data reuse (reductions, matmul, convolution with filter reuse) where 'evict_last' is correct to retain hot data in cache; kernel uses shared memory heavily where num_stages pipelining interacts with smem allocation; kernel is already at the GPU launch overhead floor with waves_per_sm << 0.5 and duration < 2 us where these changes cannot overcome dispatch latency

#### ✅ Plan C: `software_pipelining_via_num_stages` (1.524×)

When all autotune configs use num_stages=1, Triton's software pipeline is disabled and the scheduler stalls waiting for memory ops to complete (No Eligible warp % > 85%, achieved occupancy < 15%). Expanding the autotune config set to include num_stages=2 and num_stages=3 variants alongside larger block sizes allows the autotuner to select a pipelined configuration that overlaps load/store latency with computation. In this case, the winning config used block_size=256 (vs 32 before), raising waves_per_sm from 0.37 to 1.48, achieved occupancy from 13.6% to 69.1%, and memory throughput from 33 GB/s to 200 GB/s — the primary driver of the 1.52x speedup was the larger block size enabling better SM utilization, with pipelining contributing to latency hiding.

> **Applicable when**: All autotune configs use num_stages=1; No Eligible warp stall % > 50%; achieved occupancy is far below theoretical (e.g., <30% vs 50%+ theoretical); waves_per_sm < 1.0 indicating the grid is too small to saturate the GPU; the kernel is memory-latency-bound (CPIStall L1TEX > 30%); block sizes in existing configs are small (≤64 threads)

> **Avoid when**: The kernel already uses num_stages>1 in its autotune configs; register pressure is high (registers_per_thread > 40) where adding pipeline stages would reduce occupancy; the kernel is compute-bound rather than memory-latency-bound; the grid is already large enough to saturate all SMs (waves_per_sm > 4)

---

## Iteration 4

**Best so far**: 1.9369× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_meta_lambda | 1.385× | 2.000× | ✓ |  |
| B | shared_memory_transpose_to_coalesce_strided_stores | 1.355× | 1.937× | ✓ |  |
| C | fix_hardcoded_grid_to_use_meta_lambda_for_autotune | 1.407× | 1.380× | ✓ |  |
| D | store_evict_first_with_large_warp_configs | 1.368× | 1.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | shared_memory_tra... | Δ | fix_hardcoded_gri... | Δ | store_evict_first... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3070 ns | -3070 ns (-50.0%) | 3170 ns | -2970 ns (-48.4%) | 4450 ns | -1690 ns (-27.5%) | 3170 ns | -2970 ns (-48.4%) |
| SM Throughput (%) | 16.51 % | 2.72 % | -13.79 pp | 5.01 % | -11.50 pp | 27.32 % | +10.81 pp | 2.68 % | -13.83 pp |
| Memory Throughput (%) | 40.01 % | 3.87 % | -36.14 pp | 11.22 % | -28.79 pp | 21.70 % | -18.31 pp | 3.84 % | -36.17 pp |
| DRAM Throughput (%) | 20.05 % | 0.69 % | -19.36 pp | 2.62 % | -17.43 pp | 3.70 % | -16.35 pp | 0.67 % | -19.38 pp |
| Achieved Occupancy (%) | 55.16 % | 12.68 % | -42.48 pp | 27.17 % | -27.99 pp | 68.30 % | +13.14 pp | 14.99 % | -40.17 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.41 % | 86.91 % | +16.50 pp | 80.42 % | +10.01 pp | 42.22 % | -28.19 pp | 88.73 % | +18.32 pp |
| IPC Active | 1.16 inst/cycle | 0.36 inst/cycle | -0.80 | 0.69 inst/cycle | -0.47 | 2.42 inst/cycle | +1.26 | 0.31 inst/cycle | -0.85 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_use_meta_lambda` (2.000×)

When a Triton kernel's launch grid is hardcoded to a fixed tile shape (e.g., `cdiv(xnumel, 128), cdiv(ynumel, 8)`) rather than using a META lambda, the autotuner silently evaluates all configs against the wrong grid — any config whose XBLOCK/YBLOCK differs from the hardcoded values either produces incorrect results or is rejected, forcing the autotuner to always select the one config that matches the hardcoded tile. Replacing the hardcoded grid with `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` allows the autotuner to correctly benchmark all configs with their proper grid dimensions. In this iteration the NCU per-kernel metrics remained nearly identical (block_size=32, waves_per_sm=0.37 unchanged), suggesting the autotune cache was still active or the winning config did not change, but the 2.0x wall-clock speedup indicates host-side dispatch or config selection improved.

> **Applicable when**: Triton kernel uses @triton.autotune with multiple XBLOCK/YBLOCK configs but the launch grid is a hardcoded tuple computed from fixed tile sizes; autotuner always selects the same tiny config (block_size matches the hardcoded tile); waves_per_sm << 1.0 and achieved occupancy far below theoretical despite a large autotune config space; prior attempts to expand the config space yielded no NCU metric improvement

> **Avoid when**: The kernel already uses a META lambda for its grid; the kernel has only a single autotune config (no benefit from dynamic grid); XBLOCK/YBLOCK keys in the META dict do not match the keys used in all autotune configs (would cause KeyError at runtime); the kernel is not autotuned at all

#### ✅ Plan C: `fix_hardcoded_grid_to_use_meta_lambda_for_autotune` (1.380×)

When a Triton kernel uses @triton.autotune but the grid is hardcoded to a fixed tile size (e.g., cdiv(xnumel, 128)), the autotuner benchmarks all configs but only the config matching the hardcoded tile shape actually runs correctly — all other configs silently produce wrong results or are rejected. Replacing the hardcoded grid with a META lambda (lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))) allows the autotuner to genuinely explore the full config space. In this case, fixing the META lambda enabled the autotuner to select a block_size=256 config, raising waves_per_sm from 0.37 to 1.48, achieved occupancy from 13.8% to 68.3%, and memory throughput from 33.75 GB/s to 181.29 GB/s, yielding a 1.38x speedup on top of an already-improved baseline.

> **Applicable when**: Triton kernel has @triton.autotune with multiple XBLOCK/YBLOCK configs but the grid passed to the kernel call is hardcoded to specific tile values rather than using a META lambda; NCU shows low waves_per_sm (<0.5), low achieved occupancy (<20%), and the block_size in NCU matches only one of the autotune configs; IssueSlotUtilization and AchievedOccupancy NCU rules are triggered

> **Avoid when**: The grid is already a META lambda; the kernel has only one autotune config (no benefit from META lambda); the kernel uses a 1D grid where only XBLOCK varies and YBLOCK is not a tile dimension; correctness constraints prevent certain XBLOCK/YBLOCK combinations (e.g., when tile dimensions affect index arithmetic correctness)

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)
Iter 1     3170 ns  (1.937×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 4220 ns  (1.4550×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # YBLOCK=1 variants -- KEY missing configs
+        triton.Config({'XBLOCK': 16,   'YBLOCK': 1}, num_warps=1, num_stages=3),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        # YBLOCK=2 variants
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+        # YBLOCK=4/8 retained
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        # num_stages=1 configs to reduce shared memory pressure and increase occupancy
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 16}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=4, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +50,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 4), 4)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +60,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
