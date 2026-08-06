# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 3870 ns (**1.5556×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_dead_autotune_activate_dynamic_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_065/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.56 % | **Memory**: 40.97 % | **DRAM**: 20.44 %
- **Achieved Occupancy**: 54.89 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tile_size_tuning | 1.262× | 1.012× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.467× | 1.593× | ✓ | **✓ best** |
| D | autotune_tile_sizes_increase_xblock_waves | 1.312× | 1.393× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 5950 ns | -70 ns (-1.2%) | 3780 ns | -2240 ns (-37.2%) | 4320 ns | -1700 ns (-28.2%) |
| SM Throughput (%) | 16.56 % | 24.31 % | +7.75 pp | 25.36 % | +8.80 pp | 27.14 % | +10.58 pp |
| Memory Throughput (%) | 40.97 % | 35.80 % | -5.17 pp | 18.09 % | -22.88 pp | 21.52 % | -19.45 pp |
| DRAM Throughput (%) | 20.44 % | 11.04 % | -9.40 pp | 4.35 % | -16.09 pp | 3.84 % | -16.60 pp |
| Achieved Occupancy (%) | 54.89 % | 50.67 % | -4.22 pp | 65.95 % | +11.06 pp | 68.82 % | +13.93 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.62 % | 54.67 % | -15.95 pp | 48.41 % | -22.21 pp | 37.68 % | -32.94 pp |
| IPC Active | 1.15 inst/cycle | 1.77 inst/cycle | +0.62 | 2.22 inst/cycle | +1.07 | 2.29 inst/cycle | +1.14 |


### Key Learnings

#### ✅ Plan D: `autotune_tile_sizes_increase_xblock_waves` (1.437×)

When a Triton kernel uses @triton.autotune but the launch grid is a hardcoded tuple (e.g., `(cdiv(xnumel, 128), cdiv(ynumel, 8))`), the autotuner benchmarks all configs but the winning tile is never applied — the kernel always runs with the hardcoded tile. Replacing the grid with a lambda `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` and removing explicit XBLOCK/YBLOCK kwargs from the call site activates the autotuner's tile selection. Adding larger XBLOCK configs (e.g., 1024) and num_stages=2 variants further expands the search space, allowing the autotuner to select tiles that increase L1/TEX cache throughput (29% → 46%) and eligible warps per scheduler (3.47 → 4.04), yielding a 1.44x speedup.

> **Applicable when**: Triton kernel uses @triton.autotune but launch grid is a hardcoded tuple (not a lambda), causing the autotuner's winning tile to be silently ignored. LaunchConfiguration rule triggered with partial-wave tail. Eligible warps per scheduler below 4.0 indicating latency-hiding headroom. L2 hit rate ~75-80% suggesting moderate reuse that benefits from larger tiles.

> **Avoid when**: Grid is already a lambda (autotune already active). Kernel uses shared memory sized by XBLOCK/YBLOCK where changing tile size would cause OOM. Kernel is already achieving > 2 waves/SM and occupancy > 80%. Very short kernels where autotune overhead dominates.

---

## Iteration 2

**Best so far**: 1.5926× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.470× | 1.556× | ✓ | **✓ best** |
| B | vectorized_loads_with_yblock1_store_coalescing | 1.429× | 1.447× | ✓ |  |
| C | autotune_yblock1_to_fix_strided_store_coalescing | 1.465× | 1.528× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | autotune_yblock1_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3870 ns | -2150 ns (-35.7%) | 4160 ns | -1860 ns (-30.9%) | 3940 ns | -2080 ns (-34.6%) |
| SM Throughput (%) | 16.56 % | 25.49 % | +8.93 pp | 27.55 % | +10.99 pp | 26.23 % | +9.67 pp |
| Memory Throughput (%) | 40.97 % | 17.63 % | -23.34 pp | 21.85 % | -19.12 pp | 17.31 % | -23.66 pp |
| DRAM Throughput (%) | 20.44 % | 4.30 % | -16.14 pp | 3.99 % | -16.45 pp | 4.21 % | -16.23 pp |
| Achieved Occupancy (%) | 54.89 % | 66.48 % | +11.59 pp | 69.58 % | +14.69 pp | 65.86 % | +10.97 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.62 % | 42.28 % | -28.34 pp | 43.05 % | -27.57 pp | 42.32 % | -28.30 pp |
| IPC Active | 1.15 inst/cycle | 2.24 inst/cycle | +1.09 | 2.27 inst/cycle | +1.12 | 2.20 inst/cycle | +1.05 |


### Key Learnings

#### ✅ Plan C: `autotune_yblock1_to_fix_strided_store_coalescing` (1.593×)

When a 2D Triton kernel performs stores with a stride in the Y dimension (e.g., layout permutation like NCHW→NHWC where the store index contains `y0 + stride*x`), using YBLOCK>1 causes threads within a block to write to non-contiguous addresses, triggering UncoalescedGlobalAccess. Adding a `@triton.autotune` decorator with configurations that include YBLOCK=1 allows the autotuner to select a tile shape where each CTA handles a single Y-row, making X-dimension stores contiguous and eliminating intra-block store stride interleaving. This also reduces register pressure (26→18 registers/thread) and increases waves/SM (0.74→1.48), raising achieved occupancy from 54.9% to 66% and cutting warp stall cycles per instruction from 29.9 to 18.2.

> **Applicable when**: Triton 2D kernel with UncoalescedGlobalAccess rule triggered on stores; store index has a stride in the Y dimension (e.g., `y0 + stride*x` pattern from layout permutation); waves/SM < 1.0 indicating undersized grid; CPIStall on L1TEX > 40%; eligible warps per scheduler < 1.5; YBLOCK fixed at > 1 in baseline.

> **Avoid when**: Store pattern is already coalesced (no stride in Y dimension); kernel is compute-bound rather than memory-latency-bound; ynumel is large (e.g., > 1024) making YBLOCK=1 create too many CTAs and excessive grid overhead; kernel already uses autotune.

---

## Iteration 3

**Best so far**: 1.5926× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_dead_autotune_activate_dynamic_grid | 0.717× | 1.556× | ✓ |  |
| B | evict_first_store_with_software_pipelining | 1.470× | 1.593× | ✓ |  |
| C | fix_autotune_grid_lambda_with_software_pipelining | 1.412× | 1.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_dead_autotune... | Δ | evict_first_store... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3870 ns | -2150 ns (-35.7%) | 3780 ns | -2240 ns (-37.2%) | 4220 ns | -1800 ns (-29.9%) |
| SM Throughput (%) | 16.56 % | 26.78 % | +10.22 pp | 26.84 % | +10.28 pp | 27.45 % | +10.89 pp |
| Memory Throughput (%) | 40.97 % | 17.70 % | -23.27 pp | 18.13 % | -22.84 pp | 21.81 % | -19.16 pp |
| DRAM Throughput (%) | 20.44 % | 4.30 % | -16.14 pp | 4.39 % | -16.05 pp | 3.91 % | -16.53 pp |
| Achieved Occupancy (%) | 54.89 % | 66.05 % | +11.16 pp | 67.29 % | +12.40 pp | 70.24 % | +15.35 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.62 % | 40.77 % | -29.85 pp | 39.62 % | -31.00 pp | 36.98 % | -33.64 pp |
| IPC Active | 1.15 inst/cycle | 2.21 inst/cycle | +1.06 | 2.35 inst/cycle | +1.20 | 2.27 inst/cycle | +1.12 |

---

## Iteration 4

**Best so far**: 1.5926× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_activate_autotune_tile_selection | 1.438× | 1.636× | ✓ |  |
| B | activate_lambda_grid_with_num_stages2_pipelining | 1.470× | 1.437× | ✓ |  |
| C | eliminate_partial_wave_with_lambda_grid_and_wave_aligned_configs | 1.414× | 1.362× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | activate_lambda_g... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3680 ns | -2340 ns (-38.9%) | 4190 ns | -1830 ns (-30.4%) | 4420 ns | -1600 ns (-26.6%) |
| SM Throughput (%) | 16.56 % | 26.48 % | +9.92 pp | 28.99 % | +12.43 pp | 25.57 % | +9.01 pp |
| Memory Throughput (%) | 40.97 % | 18.40 % | -22.57 pp | 21.43 % | -19.54 pp | 20.16 % | -20.81 pp |
| DRAM Throughput (%) | 20.44 % | 4.47 % | -15.97 pp | 3.95 % | -16.49 pp | 3.75 % | -16.69 pp |
| Achieved Occupancy (%) | 54.89 % | 66.98 % | +12.09 pp | 69.30 % | +14.41 pp | 68.30 % | +13.41 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.62 % | 39.48 % | -31.14 pp | 38.93 % | -31.69 pp | 36.63 % | -33.99 pp |
| IPC Active | 1.15 inst/cycle | 2.35 inst/cycle | +1.20 | 2.46 inst/cycle | +1.31 | 2.36 inst/cycle | +1.21 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_activate_autotune_tile_selection` (1.636×)

When a Triton kernel has a `@triton.autotune` decorator but the launch site uses a hardcoded grid tuple (e.g., `grid = (cdiv(xnumel, 128), cdiv(ynumel, 8))`), the autotuner benchmarks all configs but the winning tile dimensions are never applied at launch. Replacing the fixed grid with `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` activates the autotuner's tile selection. In this iteration, the profiling metrics barely changed (duration 3.78→3.68 µs, grid size unchanged at 1568, same waves/SM), suggesting the autotuner selected a config very close to the hardcoded baseline, yielding only a modest ~2.7% wall-clock improvement in NCU but a 1.64x speedup in end-to-end timing — likely due to reduced autotuner overhead or warm-path selection.

> **Applicable when**: Triton kernel has @triton.autotune decorator but launch site uses a hardcoded grid tuple; LaunchConfiguration rule triggered (partial wave tail effect); UncoalescedGlobalAccess present on stores; AchievedOccupancy below theoretical; kernel is memory-latency-bound (SM throughput < 30%).

> **Avoid when**: Grid is already a lambda META (autotune already functional); kernel is compute-bound (SM throughput > 60%); shared memory arrays are sized by XBLOCK/YBLOCK constexprs where changing tile size would break correctness or cause OOM; ynumel is very large making YBLOCK=1 create excessive CTAs.

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3780 ns  (1.593×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3870 ns  (1.5556×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original 9 configs with explicit num_stages=1
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 4}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=1),
+        # 7 new configs with num_stages=2
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4,  num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +37,17 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting
+    xarange = tl.arange(0, XBLOCK)
+    xarange = tl.max_contiguous(tl.multiple_of(xarange, XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +55,7 @@

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
