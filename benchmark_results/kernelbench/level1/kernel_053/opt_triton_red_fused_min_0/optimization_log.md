# Kernel Optimization Log

## Summary

- **Kernel**: `min_dim1_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15390000 ns — **Best latency**: 2440000 ns (**6.3074×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_uncoalesced_access_via_xblock_vectorized_tiling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `min_dim1_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_053/synthesis/kernel_0_triton_red_fused_min_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524160,1,1), Waves/SM: 496.36
- **Registers/thread**: 64
- **Duration**: 15390000 ns
- **SM Throughput**: 9.41 % | **Memory**: 95.28 % | **DRAM**: 11.40 %
- **Achieved Occupancy**: 47.40 %
- **IPC Active**: 0.38 inst/cycle | **No Eligible**: 90.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_uncoalesced_access_via_transposed_tiling | 0.713× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15390000 ns | 15500000 ns | +110000 ns (+0.7%) |
| SM Throughput (%) | 9.41 % | 8.23 % | -1.18 pp |
| Memory Throughput (%) | 95.28 % | 95.31 % | +0.03 pp |
| DRAM Throughput (%) | 11.40 % | 11.32 % | -0.08 pp |
| Achieved Occupancy (%) | 47.40 % | 59.74 % | +12.34 pp |
| Registers/Thread | 64.00 register/thread | 45.00 register/thread | -19 |
| Waves/SM | 496.36 | 397.09 | -99.27 |
| No Eligible (%) | 90.59 % | 91.74 % | +1.15 pp |
| IPC Active | 0.38 inst/cycle | 0.33 inst/cycle | -0.05 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints_and_access_reorder | 0.606× | 2.100× | ✓ |  |
| B | fix_stride_access_pattern_for_coalesced_loads | 1.552× | 6.439× | ✓ |  |
| C | fix_uncoalesced_access_via_contiguous_r0_innermost_tiling | 0.568× | 2.283× | ✓ |  |
| D | fix_uncoalesced_access_via_xblock_vectorized_tiling | 1.609× | 6.307× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_stride_access... | Δ | fix_uncoalesced_a... | Δ | fix_uncoalesced_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15390000 ns | 7330000 ns | -8060000 ns (-52.4%) | 2390000 ns | -13000000 ns (-84.5%) | 6740000 ns | -8650000 ns (-56.2%) | 2440000 ns | -12950000 ns (-84.1%) |
| SM Throughput (%) | 9.41 % | 37.38 % | +27.97 pp | 31.71 % | +22.30 pp | 40.98 % | +31.57 pp | 22.43 % | +13.02 pp |
| Memory Throughput (%) | 95.28 % | 76.00 % | -19.28 pp | 78.32 % | -16.96 pp | 85.21 % | -10.07 pp | 72.50 % | -22.78 pp |
| DRAM Throughput (%) | 11.40 % | 23.88 % | +12.48 pp | 78.32 % | +66.92 pp | 26.07 % | +14.67 pp | 72.50 % | +61.10 pp |
| Achieved Occupancy (%) | 47.40 % | 12.46 % | -34.94 pp | 48.00 % | +0.60 pp | 18.63 % | -28.77 pp | 95.80 % | +48.40 pp |
| Registers/Thread | 64.00 register/thread | 244.00 register/thread | +180 | 32.00 register/thread | -32 | 168.00 register/thread | +104 | 32.00 register/thread | -32 |
| Waves/SM | 496.36 | 31.02 | -465.34 | 0.48 | -495.88 | 41.36 | -455.00 | 1.94 | -494.42 |
| No Eligible (%) | 90.59 % | 77.23 % | -13.36 pp | 67.55 % | -23.04 pp | 72.87 % | -17.72 pp | 76.94 % | -13.65 pp |
| IPC Active | 0.38 inst/cycle | 0.91 inst/cycle | +0.53 | 1.33 inst/cycle | +0.95 | 1.08 inst/cycle | +0.70 | 0.93 inst/cycle | +0.55 |


### Key Learnings

#### ✅ Plan D: `fix_uncoalesced_access_via_xblock_vectorized_tiling` (6.307×)

When a Triton reduction kernel iterates over a non-contiguous dimension (high stride) while the contiguous dimension is mapped to a small XBLOCK, warp lanes span high-stride addresses, causing severe uncoalesced access (NCU: UncoalescedGlobalAccess, bytes utilized per sector as low as 4/32). Increasing XBLOCK (e.g. from 1–8 to 64–1024) so that warp lanes cover contiguous memory addresses converts scattered loads into coalesced 128-byte transactions, dropping excessive sectors from 87% to 18% and raising DRAM throughput from 11% to 72%. This also reduces register pressure (45→32 registers/thread) and raises theoretical occupancy from 62.5% to 100%, yielding a 6.3x end-to-end speedup.

> **Applicable when**: NCU reports UncoalescedGlobalAccess with >50% excessive sectors; bytes utilized per sector is low (≤8/32); the kernel has a 2D tile [XBLOCK, R0_BLOCK] where XBLOCK is small (1–8) and R0_BLOCK is large (512–4096); the reduction axis has high stride in memory while the non-reduction axis is contiguous (stride=1); DRAM throughput is low (<20%) while L2 throughput is high (>80%).

> **Avoid when**: The reduction axis is already the contiguous (stride-1) dimension in memory; XBLOCK is already large and the bottleneck is compute or shared memory; the tensor is small enough that a single block covers the full reduction (no tiling needed); increasing XBLOCK would exceed shared memory or register limits causing occupancy collapse.

---

## Iteration 3

**Best so far**: 6.3074× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguity_hints_and_software_pipelining | 1.532× | 6.359× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints_to_reduce_register_pressure | 0.893× | 3.220× | ✓ |  |
| D | reduce_register_pressure_via_small_r0block_and_large_xblock_configs | 0.858× | 3.474× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15390000 ns | 2420000 ns | -12970000 ns (-84.3%) | 4780000 ns | -10610000 ns (-68.9%) | 4430000 ns | -10960000 ns (-71.2%) |
| SM Throughput (%) | 9.41 % | 23.19 % | +13.78 pp | 34.84 % | +25.43 pp | 32.16 % | +22.75 pp |
| Memory Throughput (%) | 95.28 % | 78.41 % | -16.87 pp | 86.75 % | -8.53 pp | 78.63 % | -16.65 pp |
| DRAM Throughput (%) | 11.40 % | 78.41 % | +67.01 pp | 36.81 % | +25.41 pp | 40.49 % | +29.09 pp |
| Achieved Occupancy (%) | 47.40 % | 48.04 % | +0.64 pp | 18.36 % | -29.04 pp | 12.12 % | -35.28 pp |
| Registers/Thread | 64.00 register/thread | 32.00 register/thread | -32 | 156.00 register/thread | +92 | 255.00 register/thread | +191 |
| Waves/SM | 496.36 | 0.48 | -495.88 | 5.17 | -491.19 | 1.94 | -494.42 |
| No Eligible (%) | 90.59 % | 77.28 % | -13.31 pp | 69.49 % | -21.10 pp | 69.60 % | -20.99 pp |
| IPC Active | 0.38 inst/cycle | 0.91 inst/cycle | +0.53 | 1.22 inst/cycle | +0.84 | 1.22 inst/cycle | +0.84 |

---

## Iteration 4

**Best so far**: 6.3074× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_waves_per_sm_via_smaller_xblock_autotune_configs | 1.575× | 6.605× | ✓ |  |
| B | cap_r0block_to_restore_occupancy_from_register_explosion | 1.465× | 5.919× | ✓ |  |
| C | remove_evict_last_to_restore_occupancy_from_register_pressure | 0.912× | 3.338× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_waves_pe... | Δ | cap_r0block_to_re... | Δ | remove_evict_last... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15390000 ns | 2330000 ns | -13060000 ns (-84.9%) | 2600000 ns | -12790000 ns (-83.1%) | 4610000 ns | -10780000 ns (-70.0%) |
| SM Throughput (%) | 9.41 % | 52.32 % | +42.91 pp | 44.16 % | +34.75 pp | 37.05 % | +27.64 pp |
| Memory Throughput (%) | 95.28 % | 79.50 % | -15.78 pp | 85.15 % | -10.13 pp | 90.72 % | -4.56 pp |
| DRAM Throughput (%) | 11.40 % | 79.50 % | +68.10 pp | 69.20 % | +57.80 pp | 38.24 % | +26.84 pp |
| Achieved Occupancy (%) | 47.40 % | 95.63 % | +48.23 pp | 51.54 % | +4.14 pp | 18.08 % | -29.32 pp |
| Registers/Thread | 64.00 register/thread | 24.00 register/thread | -40 | 48.00 register/thread | -16 | 156.00 register/thread | +92 |
| Waves/SM | 496.36 | 1.94 | -494.42 | 1.55 | -494.81 | 5.17 | -491.19 |
| No Eligible (%) | 90.59 % | 55.73 % | -34.86 pp | 53.13 % | -37.46 pp | 74.39 % | -16.20 pp |
| IPC Active | 0.38 inst/cycle | 1.81 inst/cycle | +1.43 | 1.87 inst/cycle | +1.49 | 1.02 inst/cycle | +0.64 |


### Key Learnings

#### ✅ Plan A: `increase_waves_per_sm_via_smaller_xblock_autotune_configs` (6.605×)

When a Triton reduction kernel's autotune search space only includes large XBLOCK values (256–2048), the resulting grid is too small to fill the GPU (e.g., 0.48 waves/SM), leaving the scheduler starved of eligible warps (77% no-eligible-warp cycles) and unable to hide L1TEX scoreboard stalls (~31 cycles, 92% of all stall cycles). Adding smaller XBLOCK configs (64, 128) to the autotune space increases the grid size (e.g., 8190 blocks = 1.94 waves/SM), raising achieved occupancy from 48% to 96%, active warps/scheduler from 7.7 to 15.1, and eligible warps/scheduler from 0.27 to 0.80, which converts idle scheduler cycles into useful compute work. The warp stall duration per instruction remains similar (memory-bound), but the higher occupancy allows more warps to overlap latency, yielding a 6.6x speedup.

> **Applicable when**: Triton reduction kernel where: (1) all autotune configs have large XBLOCK (>=256), producing a grid with <1.0 waves/SM; (2) achieved occupancy is far below theoretical (e.g., 48% vs 100%) despite no register-pressure constraint (registers/thread is moderate, e.g., 32); (3) scheduler reports >70% no-eligible-warp cycles with eligible warps/scheduler <0.3; (4) CPIStall rule fires citing L1TEX scoreboard stalls as >90% of stall cycles; (5) the kernel is DRAM-bandwidth-bound (DRAM throughput >70%) indicating the bottleneck is latency hiding, not compute.

> **Avoid when**: Register pressure is already high (>64 registers/thread) — smaller XBLOCK with more blocks may not help if each block still consumes too many registers. Also avoid when the grid is already producing >2 waves/SM and occupancy is already >80%; in that case the bottleneck is elsewhere (e.g., uncoalesced access or compute). Do not apply when XBLOCK < 32 would break warp-level coalescing for the stride-1 access dimension.

#### ✅ Plan C: `remove_evict_last_to_restore_occupancy_from_register_pressure` (3.338×)

When a Triton reduction kernel uses eviction_policy='evict_last' on a tl.load inside a loop, the compiler treats the loaded tile as long-lived, inflating register count (e.g., 156 registers/thread) and collapsing theoretical occupancy to ~18% (Block Limit Registers=6 vs Block Limit Warps=32). Removing the eviction_policy hint allows the compiler to use default eviction semantics, reducing live register count and restoring occupancy. This is most impactful when combined with autotune configs that include smaller R0_BLOCK values (16–32) and lower num_warps (2–4), which the autotuner can now select without register-induced occupancy collapse. The speedup comes from the autotuner being able to select a config that achieves higher effective GPU utilization (more waves/SM) rather than from a direct change in the profiling metrics of the selected config.

> **Applicable when**: Triton reduction kernel where: (1) registers per thread is very high (>64) and Block Limit Registers is the binding occupancy constraint (Block Limit Registers << Block Limit Warps), (2) NCU TheoreticalOccupancy rule fires citing register count, (3) a tl.load inside a loop body carries eviction_policy='evict_last', (4) theoretical occupancy is <25% while Block Limit Warps would allow much higher occupancy.

> **Avoid when**: Register pressure is already low (<64 registers/thread) and occupancy is not register-limited; when evict_last was intentionally added to improve L1 hit rate for a tile that is genuinely reused across loop iterations and the kernel is already occupancy-limited by other factors (shared memory, warps); when removing evict_last causes correctness issues due to cache eviction of data needed for correctness.

---

## Cumulative Progress

```
Baseline:  15390000 ns  (1.000×)
Iter 2     2440000 ns  (6.307×)  — fix_uncoalesced_access_via_xblock_vectorized_tiling

Final best: 2440000 ns  (6.3074×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,21 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=4, num_stages=2),
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=4, num_stages=3),
-        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 1},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 1},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 1},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 1},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 4},  num_warps=4,  num_stages=5),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 4},  num_warps=8,  num_stages=5),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 4},  num_warps=16, num_stages=5),
+        triton.Config({'XBLOCK': 256,  'R0_BLOCK': 8},  num_warps=4,  num_stages=5),
+        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 8},  num_warps=8,  num_stages=5),
+        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 8},  num_warps=16, num_stages=6),
+        triton.Config({'XBLOCK': 128,  'R0_BLOCK': 16}, num_warps=4,  num_stages=6),
+        triton.Config({'XBLOCK': 256,  'R0_BLOCK': 16}, num_warps=8,  num_stages=6),
     ],
     key=['xnumel', 'r0_numel'],
 )
@@ -28,38 +33,37 @@

     # x dimension: batch * dim2 = 128 * 4095 = 524160
     # r0 dimension: dim1 = 4096 (reduction dimension)
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
+    xindex = xoffset + tl.arange(0, XBLOCK)   # [XBLOCK] 1D
     xmask = xindex < xnumel
 
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
+    # x0 = column index (dim2), x1 = batch index
+    x0 = xindex % 4095   # [XBLOCK]
+    x1 = xindex // 4095  # [XBLOCK]
 
-    # x0 = column index (dim2), x1 = batch index
-    x0 = xindex % 4095  # dim2 index
-    x1 = xindex // 4095  # batch index
-
-    # Initialize accumulator with +inf
-    _min_val = tl.full([XBLOCK, R0_BLOCK], float("inf"), tl.float32)
+    # Initialize accumulator with +inf, shape [XBLOCK]
+    _min_val = tl.full([XBLOCK], float("inf"), tl.float32)
 
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
+        r0_index = r0_offset + tl.arange(0, R0_BLOCK)  # [R0_BLOCK] contiguous
         r0_mask = r0_index < r0_numel
 
+        # Apply contiguity hints to x0 to enable vectorized LDG.128 loads
+        x0_vec = tl.max_contiguous(tl.multiple_of(x0, 16), 16)  # assert stride-1, 16-aligned
+
+        # 2D pointer array: [XBLOCK, R0_BLOCK]
+        # x0_vec varies across XBLOCK (stride-1 in memory) -> COALESCED + VECTORIZED
         # Input layout: [128, 4096, 4095] with strides [16773120, 4095, 1]
         # index = x1 * 16773120 + r0_index * 4095 + x0
-        tmp0 = tl.load(
-            in_ptr0 + (x0 + 4095 * r0_index + 16773120 * x1),
-            r0_mask & xmask,
-            eviction_policy='evict_first',
-            other=float("inf"),
-        )
-        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-        # Element-wise minimum update
-        _min_val = tl.where(r0_mask & xmask, tl.minimum(_min_val, tmp1), _min_val)
+        ptrs = (in_ptr0
+                + x0_vec[:, None]            # [XBLOCK, 1] stride-1 (fast dim, vectorized!)
+                + 4095 * r0_index[None, :]   # [1, R0_BLOCK] stride-4095 (slow dim, narrow)
+                + 16773120 * x1[:, None])    # [XBLOCK, 1] batch offset
+        mask2d = xmask[:, None] & r0_mask[None, :]
+        tmp0 = tl.load(ptrs, mask=mask2d, other=float("inf"))  # [XBLOCK, R0_BLOCK]
+        row_min = tl.min(tmp0, axis=1)   # reduce R0 immediately -> [XBLOCK]
+        _min_val = tl.minimum(_min_val, row_min)
 
-    # Reduce across R0_BLOCK dimension
-    result = tl.min(_min_val, axis=1)[:, None]
-    tl.store(out_ptr0 + xindex, result, xmask)
-
+    tl.store(out_ptr0 + xindex, _min_val, mask=xmask)
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
