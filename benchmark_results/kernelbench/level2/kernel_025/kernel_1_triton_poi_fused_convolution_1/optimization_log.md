# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285120 ns — **Best latency**: 196700 ns (**1.4495×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`mio_pressure_reduction_via_vectorized_load_pointer_hints_and_evict_first_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_025/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285120 ns
- **SM Throughput**: 19.80 % | **Memory**: 79.06 % | **DRAM**: 72.38 %
- **Achieved Occupancy**: 94.06 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_fix_strided_load | 2.125× | 1.096× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285120 ns | 260060 ns | -25060 ns (-8.8%) |
| SM Throughput (%) | 19.80 % | 52.70 % | +32.90 pp |
| Memory Throughput (%) | 79.06 % | 91.49 % | +12.43 pp |
| DRAM Throughput (%) | 72.38 % | 40.62 % | -31.76 pp |
| Achieved Occupancy (%) | 94.06 % | 57.89 % | -36.17 pp |
| Registers/Thread | 25.00 register/thread | 48.00 register/thread | +23 |
| Waves/SM | 59.88 | 95.81 | +35.93 |
| No Eligible (%) | 80.18 % | 46.66 % | -33.52 pp |
| IPC Active | 0.79 inst/cycle | 2.13 inst/cycle | +1.34 |

---

## Iteration 2

**Best so far**: 1.0964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_multiple_of_hints | 2.337× | 1.094× | ✓ | **✓ best** |
| B | smem_tiled_transpose_coalesced_store | 1.419× | 1.519× | ✓ |  |
| C | coalesced_store_via_shared_memory_transpose | 2.070× | 1.104× | ✓ |  |
| D | fix_hardcoded_grid_to_enable_autotune | 2.119× | 1.082× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | smem_tiled_transp... | Δ | coalesced_store_v... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285120 ns | 260700 ns | -24420 ns (-8.6%) | 187740 ns | -97380 ns (-34.2%) | 258240 ns | -26880 ns (-9.4%) | 263550 ns | -21570 ns (-7.6%) |
| SM Throughput (%) | 19.80 % | 52.59 % | +32.79 pp | 38.16 % | +18.36 pp | 54.31 % | +34.51 pp | 52.68 % | +32.88 pp |
| Memory Throughput (%) | 79.06 % | 91.30 % | +12.24 pp | 56.89 % | -22.17 pp | 93.84 % | +14.78 pp | 91.45 % | +12.39 pp |
| DRAM Throughput (%) | 72.38 % | 40.53 % | -31.85 pp | 56.89 % | -15.49 pp | 40.90 % | -31.48 pp | 40.06 % | -32.32 pp |
| Achieved Occupancy (%) | 94.06 % | 57.90 % | -36.16 pp | 91.86 % | -2.20 pp | 69.89 % | -24.17 pp | 57.91 % | -36.15 pp |
| Registers/Thread | 25.00 register/thread | 48.00 register/thread | +23 | 25.00 register/thread | +0 | 40.00 register/thread | +15 | 48.00 register/thread | +23 |
| Waves/SM | 59.88 | 95.81 | +35.93 | 59.88 | +0.00 | 79.84 | +19.96 | 95.81 | +35.93 |
| No Eligible (%) | 80.18 % | 46.68 % | -33.50 pp | 62.79 % | -17.39 pp | 45.00 % | -35.18 pp | 46.67 % | -33.51 pp |
| IPC Active | 0.79 inst/cycle | 2.13 inst/cycle | +1.34 | 1.46 inst/cycle | +0.67 | 2.20 inst/cycle | +1.41 | 2.13 inst/cycle | +1.34 |

---

## Iteration 3

**Best so far**: 1.0964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_fixed_autotune_grid | 2.335× | 1.501× | ✓ |  |
| B | register_reduction_via_smaller_tiles_for_occupancy | 1.575× | 1.495× | ✓ |  |
| C | vectorized_loads_with_register_transpose_for_coalesced_store | 2.332× | 1.103× | ✓ |  |
| D | vectorized_loads_with_small_tile_register_reduction | 1.421× | 1.482× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | register_reductio... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285120 ns | 189920 ns | -95200 ns (-33.4%) | 190720 ns | -94400 ns (-33.1%) | 258620 ns | -26500 ns (-9.3%) | 192350 ns | -92770 ns (-32.5%) |
| SM Throughput (%) | 19.80 % | 38.23 % | +18.43 pp | 37.37 % | +17.57 pp | 54.27 % | +34.47 pp | 39.08 % | +19.28 pp |
| Memory Throughput (%) | 79.06 % | 56.26 % | -22.80 pp | 56.01 % | -23.05 pp | 93.80 % | +14.74 pp | 55.55 % | -23.51 pp |
| DRAM Throughput (%) | 72.38 % | 56.26 % | -16.12 pp | 56.01 % | -16.37 pp | 40.85 % | -31.53 pp | 55.55 % | -16.83 pp |
| Achieved Occupancy (%) | 94.06 % | 91.60 % | -2.46 pp | 91.93 % | -2.13 pp | 69.91 % | -24.15 pp | 92.00 % | -2.06 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 40.00 register/thread | +15 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 79.84 | +19.96 | 59.88 | +0.00 |
| No Eligible (%) | 80.18 % | 63.02 % | -17.16 pp | 63.60 % | -16.58 pp | 44.99 % | -35.19 pp | 63.79 % | -16.39 pp |
| IPC Active | 0.79 inst/cycle | 1.45 inst/cycle | +0.66 | 1.47 inst/cycle | +0.68 | 2.20 inst/cycle | +1.41 | 1.42 inst/cycle | +0.63 |


### Key Learnings

#### ✅ Plan B: `register_reduction_via_smaller_tiles_for_occupancy` (1.519×)

When a Triton kernel is register-limited (Block Limit Registers < Block Limit Warps), reducing XBLOCK/YBLOCK tile sizes lowers registers-per-thread, unlocking higher theoretical and achieved occupancy. Here, shrinking XBLOCK from 128 to 32-64 dropped registers/thread from 48 to 25, raising theoretical occupancy from 62.5% to 100% and achieved occupancy from 57.9% to 91.9%, cutting kernel duration from 260 µs to 188 µs (1.52x speedup). The tradeoff is increased warp stall cycles per instruction (17→41) due to more waves competing for memory bandwidth, but the net effect is strongly positive when the register bottleneck is the primary limiter. This pattern applies to any Triton pointwise or transpose kernel where TheoreticalOccupancy rule fires and Block Limit Registers is the binding constraint.

> **Applicable when**: TheoreticalOccupancy rule fires; Block Limit Registers is the binding occupancy constraint (lower than Block Limit Warps and Block Limit Shared Mem); registers/thread >40; theoretical occupancy <75%; kernel is a pointwise or layout-transpose op where tile size directly controls register count

> **Avoid when**: Kernel is already register-efficient (registers/thread <32) or occupancy is limited by shared memory rather than registers; reducing tile size would cause excessive grid overhead (too many small CTAs); kernel is compute-bound and larger tiles are needed for instruction-level parallelism

---

## Iteration 4

**Best so far**: 1.0964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_store_via_tl_trans_register_transpose | 2.333× | 1.498× | ✓ |  |
| B | vectorized_loads_with_correct_autotune_grid | 1.575× | 1.507× | ✓ |  |
| C | coalesced_store_via_tl_trans_register_transpose | 1.420× | 1.468× | ✓ |  |
| D | mio_pressure_reduction_via_vectorized_load_pointer_hints_and_evict_first_store | 2.355× | 1.450× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_store_v... | Δ | vectorized_loads_... | Δ | coalesced_store_v... | Δ | mio_pressure_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285120 ns | 190370 ns | -94750 ns (-33.2%) | 189150 ns | -95970 ns (-33.7%) | 194270 ns | -90850 ns (-31.9%) | 196700 ns | -88420 ns (-31.0%) |
| SM Throughput (%) | 19.80 % | 37.84 % | +18.04 pp | 39.70 % | +19.90 pp | 38.46 % | +18.66 pp | 28.52 % | +8.72 pp |
| Memory Throughput (%) | 79.06 % | 56.04 % | -23.02 pp | 56.43 % | -22.63 pp | 55.05 % | -24.01 pp | 54.51 % | -24.55 pp |
| DRAM Throughput (%) | 72.38 % | 56.04 % | -16.34 pp | 56.43 % | -15.95 pp | 55.05 % | -17.33 pp | 54.51 % | -17.87 pp |
| Achieved Occupancy (%) | 94.06 % | 91.83 % | -2.23 pp | 92.40 % | -1.66 pp | 92.00 % | -2.06 pp | 89.32 % | -4.74 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 32.00 register/thread | +7 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 29.94 | -29.94 |
| No Eligible (%) | 80.18 % | 62.80 % | -17.38 pp | 63.30 % | -16.88 pp | 62.10 % | -18.08 pp | 70.84 % | -9.34 pp |
| IPC Active | 0.79 inst/cycle | 1.46 inst/cycle | +0.67 | 1.47 inst/cycle | +0.68 | 1.44 inst/cycle | +0.65 | 1.16 inst/cycle | +0.37 |


### Key Learnings

#### ✅ Plan A: `coalesced_store_via_tl_trans_register_transpose` (1.468×)

When a Triton kernel loads a [YBLOCK, XBLOCK] tile with coalesced loads but stores it with a large-stride fast-thread dimension (e.g., stride-31500 in the fast XBLOCK direction), applying tl.trans() to transpose the register tile to [XBLOCK, YBLOCK] and re-indexing the store so the fast thread dimension maps to the stride-1 output axis converts 32 store transactions per warp into 1. Despite the theoretical benefit (MemoryCacheAccessPattern: 28.4/32 bytes/sector on stores, 1M excessive sectors), the NCU-measured per-kernel duration did not improve (189.92us → 194.27us), yet wall-clock speedup was 1.47x, indicating the gain is system-level (autotuner config selection, launch overhead, or cross-kernel scheduling) rather than per-kernel store coalescing. The tl.trans() call is a zero-cost register reorder on Hopper/Ampere and does not add runtime instructions.

> **Applicable when**: MemoryCacheAccessPattern rule triggered with <30 bytes/sector on stores; UncoalescedGlobalAccess rule triggered with >5% excessive sectors; kernel performs a layout transpose where the load is already coalesced but the store fast-thread dimension maps to a large-stride output axis; XBLOCK and YBLOCK are both powers-of-two (required for tl.trans); kernel is on Ampere/Hopper where tl.trans is a zero-cost register reorder

> **Avoid when**: Both load and store are already coalesced; XBLOCK or YBLOCK are not powers-of-two (tl.trans requires square or power-of-two tiles); kernel is compute-bound (Compute SM Throughput >70%); NCU per-kernel metrics show no store coalescing issue (bytes/sector close to 32); the store stride is already 1 in the fast thread dimension

#### ✅ Plan C: `coalesced_store_via_tl_trans_register_transpose` (1.468×)

When a Triton kernel loads a [YBLOCK, XBLOCK] tile with coalesced loads but stores it with a large-stride fast-thread dimension (e.g., stride-31500 in the fast XBLOCK direction), applying tl.trans() to transpose the register tile to [XBLOCK, YBLOCK] and re-indexing the store so the fast thread dimension maps to the stride-1 output axis converts 32 store transactions per warp into 1. Despite the theoretical benefit (MemoryCacheAccessPattern: 28.4/32 bytes/sector on stores, 1M excessive sectors), the NCU-measured per-kernel duration did not improve (189.92us → 194.27us), yet wall-clock speedup was 1.47x, indicating the gain is system-level (autotuner config selection, launch overhead, or cross-kernel scheduling) rather than per-kernel store coalescing. The tl.trans() call is a zero-cost register reorder on Hopper/Ampere and does not add runtime instructions.

> **Applicable when**: MemoryCacheAccessPattern rule triggered with <30 bytes/sector on stores; UncoalescedGlobalAccess rule triggered with >5% excessive sectors; kernel performs a layout transpose where the load is already coalesced but the store fast-thread dimension maps to a large-stride output axis; XBLOCK and YBLOCK are both powers-of-two (required for tl.trans); kernel is on Ampere/Hopper where tl.trans is a zero-cost register reorder

> **Avoid when**: Both load and store are already coalesced; XBLOCK or YBLOCK are not powers-of-two (tl.trans requires square or power-of-two tiles); kernel is compute-bound (Compute SM Throughput >70%); NCU per-kernel metrics show no store coalescing issue (bytes/sector close to 32); the store stride is already 1 in the fast thread dimension

---

## Iteration 5

**Best so far**: 1.4495× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_load_and_store_for_streaming_transpose_with_vectorized_load_hint_fix | 1.368× | 1.480× | ✓ |  |
| B | evict_first_load_to_reduce_l1tex_stall_and_cache_pollution | 2.132× | 1.475× | ✓ |  |
| C | increase_pipeline_depth_via_num_stages_to_hide_l1tex_latency | 1.422× | 1.473× | ✓ |  |
| D | eliminate_tl_trans_mio_pressure_via_direct_index_remap_with_fixed_autotune_grid | 1.455× | 1.512× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_load_... | Δ | evict_first_load_... | Δ | increase_pipeline... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285120 ns | 192640 ns | -92480 ns (-32.4%) | 193280 ns | -91840 ns (-32.2%) | 193600 ns | -91520 ns (-32.1%) | 188610 ns | -96510 ns (-33.8%) |
| SM Throughput (%) | 19.80 % | 39.39 % | +19.59 pp | 38.37 % | +18.57 pp | 39.19 % | +19.39 pp | 39.66 % | +19.86 pp |
| Memory Throughput (%) | 79.06 % | 55.19 % | -23.87 pp | 55.06 % | -24.00 pp | 55.20 % | -23.86 pp | 56.22 % | -22.84 pp |
| DRAM Throughput (%) | 72.38 % | 55.19 % | -17.19 pp | 55.06 % | -17.32 pp | 55.20 % | -17.18 pp | 56.22 % | -16.16 pp |
| Achieved Occupancy (%) | 94.06 % | 92.08 % | -1.98 pp | 92.18 % | -1.88 pp | 91.94 % | -2.12 pp | 92.10 % | -1.96 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.18 % | 61.19 % | -18.99 pp | 62.56 % | -17.62 pp | 64.14 % | -16.04 pp | 60.65 % | -19.53 pp |
| IPC Active | 0.79 inst/cycle | 1.50 inst/cycle | +0.71 | 1.51 inst/cycle | +0.72 | 1.48 inst/cycle | +0.69 | 1.55 inst/cycle | +0.76 |


### Key Learnings

#### ✅ Plan B: `evict_first_load_to_reduce_l1tex_stall_and_cache_pollution` (1.475×)

Changing tl.load eviction_policy from 'evict_last' to 'evict_first' and adding eviction_policy='evict_first' to tl.store in a read-once/write-once layout transpose kernel produced a 1.48x wall-clock speedup despite NCU-measured kernel duration being essentially unchanged (189.15us → 193.28us). All per-kernel NCU metrics remained nearly flat: DRAM throughput (56.43% → 55.06%), L2 throughput (74.34% → 72.58%), warp cycles per instruction (39.63 → 38.51), No Eligible % (63.3% → 62.56%), and achieved occupancy (92.4% → 92.18%). The CPIStall bottleneck (L1TEX scoreboard stall ~16 cycles, ~41% of stall time) was not measurably reduced, confirming that evict_first on loads does not reduce per-kernel L1TEX latency when the kernel is already streaming with near-zero L1 hit rate (0.14%). The wall-clock gain is system-level, consistent with the pattern seen in iterations 3 and 4 where autotune config changes drove wall-clock improvements invisible to single-kernel NCU profiling.

> **Applicable when**: Triton streaming kernel (read-once loads, write-once stores) with near-zero L1/TEX hit rate (<0.2%) and high wave count (waves/SM >50); CPIStall rule triggered with L1TEX scoreboard stall dominant; kernel uses evict_last on loads of data that is never reused; wall-clock time is the primary optimization target

> **Avoid when**: Kernel has meaningful L1/TEX hit rate (>5%) indicating data reuse — evict_first will evict reused data and cause regressions; kernel is compute-bound (Compute SM Throughput >70%); per-kernel NCU duration is the primary target (this change does not improve NCU-measured duration); the load eviction policy is already evict_first or evict_normal

---

## Cumulative Progress

```
Baseline:  285120 ns  (1.000×)
Iter 1     260060 ns  (1.096×)  — memory_coalescing_fix_strided_load
Iter 4     196700 ns  (1.450×)  — mio_pressure_reduction_via_vectorized_load_pointer_hints_and_evict_first_store

Final best: 196700 ns  (1.4495×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,51 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
+    # program_id(0) covers ynumel (channel+batch) — FAST dim for coalescing (stride-1 in input)
+    # program_id(1) covers xnumel (spatial) — SLOW dim (stride-128 in input)
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # spatial (stride-128 in input)
+    ymask = yindex < xnumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # channel+batch (stride-1 in input)
+    xmask = xindex < ynumel
+    x0 = (xindex % 128)   # channel within group, stride-1 in input — COALESCED!
+    x1 = xindex // 128    # batch index
+    y2 = yindex           # spatial index, stride-128 in input
+    # Load: consecutive threads (xindex) access x0 with stride-1 -> COALESCED
+    # tmp0 shape: [YBLOCK, XBLOCK]
+    _off = tl.max_contiguous(tl.multiple_of(x0 + 128*y2 + 4032000*x1, [1, 16]), [1, XBLOCK])
+    tmp0 = tl.load(in_ptr0 + _off, xmask & ymask, eviction_policy='evict_last')
+    # Transpose in registers: tmp_t shape [XBLOCK, YBLOCK]
+    # Fast dim is now YBLOCK (spatial, stride-1 in output) -> COALESCED store
+    tmp_t = tl.trans(tmp0)
+    # Re-index for coalesced store
+    xindex_t = xoffset + tl.arange(0, XBLOCK)[:, None]   # channel+batch (slow, stride-31500 in output)
+    yindex_t = yoffset + tl.arange(0, YBLOCK)[None, :]   # spatial (fast, stride-1 in output)
+    xmask_t  = xindex_t < ynumel
+    ymask_t  = yindex_t < xnumel
+    # Store: consecutive threads differ in yindex_t, stride-1 in output -> COALESCED
+    tl.store(out_ptr0 + (yindex_t + 31500*xindex_t), tmp_t, xmask_t & ymask_t)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +57,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
