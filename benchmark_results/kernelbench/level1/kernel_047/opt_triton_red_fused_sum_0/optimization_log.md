# Kernel Optimization Log

## Summary

- **Kernel**: `sum_dim1_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15260000 ns — **Best latency**: 15350000 ns (**0.9941×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `sum_dim1_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_047/synthesis/kernel_0_triton_red_fused_sum_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524160,1,1), Waves/SM: 248.18
- **Registers/thread**: 32
- **Duration**: 15260000 ns
- **SM Throughput**: 6.53 % | **Memory**: 97.27 % | **DRAM**: 11.60 %
- **Achieved Occupancy**: 97.83 %
- **IPC Active**: 0.26 inst/cycle | **No Eligible**: 93.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_evict_first_for_streaming_reduction_loads | 0.822× | 0.973× | ✓ |  |
| B | vectorized_loads_via_contiguous_hints | 1.044× | 0.994× | ✓ |  |
| C | memory_coalescing_reorder_tile_dimensions | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_e... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 15260000 ns | 15690000 ns | +430000 ns (+2.8%) | 15350000 ns | +90000 ns (+0.6%) |
| SM Throughput (%) | 6.53 % | 7.93 % | +1.40 pp | 6.49 % | -0.04 pp |
| Memory Throughput (%) | 97.27 % | 95.02 % | -2.25 pp | 96.72 % | -0.55 pp |
| DRAM Throughput (%) | 11.60 % | 12.07 % | +0.47 pp | 11.54 % | -0.06 pp |
| Achieved Occupancy (%) | 97.83 % | 97.84 % | +0.01 pp | 97.84 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 248.18 | 248.18 | +0.00 | 248.18 | +0.00 |
| No Eligible (%) | 93.47 % | 92.06 % | -1.41 pp | 93.46 % | -0.01 pp |
| IPC Active | 0.26 inst/cycle | 0.32 inst/cycle | +0.06 | 0.26 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 2.426× | 0.994× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15260000 ns | 15350000 ns | +90000 ns (+0.6%) |
| SM Throughput (%) | 6.53 % | 6.49 % | -0.04 pp |
| Memory Throughput (%) | 97.27 % | 96.73 % | -0.54 pp |
| DRAM Throughput (%) | 11.60 % | 11.53 % | -0.07 pp |
| Achieved Occupancy (%) | 97.83 % | 97.84 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 248.18 | 248.18 | +0.00 |
| No Eligible (%) | 93.47 % | 93.48 % | +0.01 pp |
| IPC Active | 0.26 inst/cycle | 0.26 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_cache_policy_for_streaming_reduction_input | 2.277× | 0.974× | ✓ |  |
| C | fix_strided_load_via_tile_axis_transposition | 0.688× | 1.137× | ✓ |  |
| D | fix_load_tile_axis_order_for_coalesced_col_access | 0.844× | 2.481× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_cache... | Δ | fix_strided_load_... | Δ | fix_load_tile_axi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15260000 ns | 15670000 ns | +410000 ns (+2.7%) | 13420000 ns | -1840000 ns (-12.1%) | 6150000 ns | -9110000 ns (-59.7%) |
| SM Throughput (%) | 6.53 % | 7.90 % | +1.37 pp | 16.58 % | +10.05 pp | 44.83 % | +38.30 pp |
| Memory Throughput (%) | 97.27 % | 95.12 % | -2.15 pp | 95.75 % | -1.52 pp | 92.20 % | -5.07 pp |
| DRAM Throughput (%) | 11.60 % | 11.98 % | +0.38 pp | 13.08 % | +1.48 pp | 28.49 % | +16.89 pp |
| Achieved Occupancy (%) | 97.83 % | 97.88 % | +0.05 pp | 99.35 % | +1.52 pp | 18.71 % | -79.12 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 23.00 register/thread | -9 | 165.00 register/thread | +133 |
| Waves/SM | 248.18 | 248.18 | +0.00 | 248.18 | +0.00 | 41.36 | -206.82 |
| No Eligible (%) | 93.47 % | 92.11 % | -1.36 pp | 86.18 % | -7.29 pp | 68.31 % | -25.16 pp |
| IPC Active | 0.26 inst/cycle | 0.32 inst/cycle | +0.06 | 0.55 inst/cycle | +0.29 | 1.27 inst/cycle | +1.01 |


### Key Learnings

#### ✅ Plan C: `fix_strided_load_via_tile_axis_transposition` (1.137×)

When a kernel loads a 2D tile [BLOCK_M, BLOCK_R] where the reduction axis (BLOCK_R) has a large stride in memory (e.g., last_dim elements apart), warp threads access non-contiguous sectors, wasting 87%+ of L2 bandwidth. Transposing the tile to [BLOCK_R, BLOCK_M] so that the contiguous dimension (stride=1) varies along axis=1 (the warp-contiguous axis) eliminates sector waste and reduces warp stall cycles per instruction from ~241 to ~115. This is a minimal 3-line change: flip the broadcast dimensions of the index components and change tl.sum axis from 1 to 0, with no grid or output indexing changes required.

> **Applicable when**: UncoalescedGlobalAccess rule fires with 80%+ excessive sectors; MemoryCacheAccessPattern shows only 4/32 bytes utilized per sector; L2 throughput >90% saturated while compute <15%; the strided dimension is the reduction axis (not the output axis); a single CTA owns all reduction iterations for its output elements (no atomics needed).

> **Avoid when**: The reduction axis is already the contiguous (stride=1) dimension in memory; when BLOCK_M=1 (no 2D tile to transpose); when the kernel uses atomics for partial reductions across CTAs (transposing alone won't fix correctness); when the output axis has a large stride instead of the reduction axis.

#### ✅ Plan D: `fix_load_tile_axis_order_for_coalesced_col_access` (2.481×)

When reducing over a middle dimension of a 3D row-major tensor (strides: [D1*D2, D2, 1]), a [BLOCK_M, BLOCK_R] load tile that places the reduction index (stride=D2) on the warp-contiguous axis (axis=1) causes each warp's 32 threads to access memory D2 elements apart, wasting ~87% of L2 sectors (only 4/32 bytes utilized per sector) and saturating L2 throughput above 96%. The fix is to transpose the tile to [BLOCK_R, BLOCK_M] so the output index (stride=1) is the warp-contiguous axis (axis=1), then reduce with tl.sum(vals, axis=0). This collapses L2 throughput from 96% to 51%, raises L1 hit rate from 0.02% to 84.75%, drops warp cycles per instruction from 241 to 9.5, and improves compute throughput from 6.5% to 44.8%.

> **Applicable when**: Triton kernel reduces over a middle dimension of a 3D row-major tensor (strides [D1*D2, D2, 1]). The load tile is 2D with the reduction index (stride=D2, large) placed on the warp-contiguous axis (axis=1 / last tile dimension). NCU reports UncoalescedGlobalAccess with >80% excessive sectors, MemoryCacheAccessPattern with only 4/32 bytes utilized per sector, L2 throughput >80%, and compute throughput <15%.

> **Avoid when**: The reduction dimension has stride=1 (already coalesced). The tile is already [BLOCK_R, BLOCK_M] with the output index on axis=1. Occupancy is already register-limited and adding BLOCK_M as a large tile dimension would further reduce theoretical occupancy below an acceptable threshold (after the fix, registers per thread rose to 165, limiting theoretical occupancy to 18.75%).

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_cache_policy_for_streaming_reduction_tile_loads | 0.807× | 2.377× | ✓ |  |
| B | evict_first_for_streaming_input_and_output | 0.505× | 1.026× | ✓ |  |
| C | evict_first_for_streaming_output_store | 2.278× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_cache... | Δ | evict_first_for_s... | Δ | evict_first_for_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15260000 ns | 6420000 ns | -8840000 ns (-57.9%) | 14870000 ns | -390000 ns (-2.6%) | 15670000 ns | +410000 ns (+2.7%) |
| SM Throughput (%) | 6.53 % | 42.93 % | +36.40 pp | 14.90 % | +8.37 pp | 7.90 % | +1.37 pp |
| Memory Throughput (%) | 97.27 % | 90.61 % | -6.66 pp | 95.20 % | -2.07 pp | 95.02 % | -2.25 pp |
| DRAM Throughput (%) | 11.60 % | 27.30 % | +15.70 pp | 11.80 % | +0.20 pp | 12.00 % | +0.40 pp |
| Achieved Occupancy (%) | 97.83 % | 18.67 % | -79.16 pp | 99.28 % | +1.45 pp | 97.88 % | +0.05 pp |
| Registers/Thread | 32.00 register/thread | 165.00 register/thread | +133 | 24.00 register/thread | -8 | 32.00 register/thread | +0 |
| Waves/SM | 248.18 | 41.36 | -206.82 | 248.18 | +0.00 | 248.18 | +0.00 |
| No Eligible (%) | 93.47 % | 66.40 % | -27.07 pp | 85.44 % | -8.03 pp | 92.07 % | -1.40 pp |
| IPC Active | 0.26 inst/cycle | 1.34 inst/cycle | +1.08 | 0.58 inst/cycle | +0.32 | 0.32 inst/cycle | +0.06 |


### Key Learnings

#### ✅ Plan A: `evict_first_cache_policy_for_streaming_reduction_tile_loads` (2.377×)

Adding eviction_policy='evict_first' to streaming reduction loads is expected to reduce L1/L2 cache pollution from non-reused data, but in this case the profiling data shows the kernel actually became ~4% slower (6.15ms → 6.42ms) despite the reported 2.38x speedup label. The L1 hit rate dropped from 84.75% to 73.07% and L2 hit rate increased from 42.27% to 55.12%, suggesting evict_first caused more L2 pressure rather than relieving it. The kernel remained bottlenecked on L1/TEX throughput (~91-93%) with no meaningful improvement in scheduler eligibility or compute throughput, indicating that evict_first alone cannot overcome the fundamental register-pressure-limited occupancy (18.75%) and residual uncoalesced access patterns.

> **Applicable when**: Streaming reduction where input data far exceeds L2 capacity, L1/TEX throughput is saturated >90%, L2 hit rate is low (<50%), and the kernel is memory-bandwidth bound with high reuse of accumulator data. Most effective when occupancy is not register-limited and there is genuine cache pollution from non-reused streaming data.

> **Avoid when**: The kernel is already register-limited (e.g., 165 registers/thread, 18.75% theoretical occupancy) and L1 hit rate is already high (>80%) due to prior coalescing fixes. In this regime, evict_first drops L1 hit rate (84.75% → 73.07%) and increases L2 pressure, causing a net regression. Also avoid when uncoalesced access patterns remain the dominant bottleneck — fix coalescing first.

---

## Cumulative Progress

```
Baseline:  15260000 ns  (1.000×)

Final best: 15350000 ns  (0.9941×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,18 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 2048}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 4, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 4, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_R': 128}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 64},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 128}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 64},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 64},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 128}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_M': 512, 'BLOCK_R': 64},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 512, 'BLOCK_R': 128}, num_warps=8, num_stages=3),
     ],
     key=['batch', 'reduce_dim', 'last_dim'],
 )
@@ -40,6 +42,8 @@

     col_offsets = out_start + tl.arange(0, BLOCK_M)
     batch_idx = col_offsets // last_dim
     col_idx = col_offsets % last_dim
+    # Hint: col_idx is contiguous and aligned to BLOCK_M
+    col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_M), BLOCK_M)
 
     mask_out = (batch_idx < batch) & (col_idx < last_dim)
 
@@ -48,26 +52,27 @@

 
     for r_start in range(0, reduce_dim, BLOCK_R):
         r_offsets = r_start + tl.arange(0, BLOCK_R)
+        # Hint: r_offsets are contiguous and aligned to BLOCK_R
+        r_offsets = tl.max_contiguous(tl.multiple_of(r_offsets, BLOCK_R), BLOCK_R)
         r_mask = r_offsets < reduce_dim
 
         # Load: in_ptr[batch_idx, r_offsets, col_idx]
-        # Shape: [BLOCK_M, BLOCK_R]
+        # Shape: [BLOCK_R, BLOCK_M]  (coalesced: col_idx is fast axis=1)
         # in_ptr stride: [reduce_dim * last_dim, last_dim, 1]
         # Linear index: batch_idx * reduce_dim * last_dim + r * last_dim + col_idx
 
-        b_idx = batch_idx[:, None]  # [BLOCK_M, 1]
-        r_idx = r_offsets[None, :]  # [1, BLOCK_R]
-        c_idx = col_idx[:, None]    # [BLOCK_M, 1]
+        b_idx = batch_idx[None, :]  # [1, BLOCK_M]
+        r_idx = r_offsets[:, None]  # [BLOCK_R, 1]
+        c_idx = col_idx[None, :]    # [1, BLOCK_M]
 
         linear_idx = b_idx * (reduce_dim * last_dim) + r_idx * last_dim + c_idx
 
-        combined_mask = mask_out[:, None] & r_mask[None, :]
+        combined_mask = mask_out[None, :] & r_mask[:, None]  # [BLOCK_R, BLOCK_M]
 
         vals = tl.load(in_ptr + linear_idx, mask=combined_mask, other=0.0)
-        acc += tl.sum(vals, axis=1)
+        acc += tl.sum(vals, axis=0)  # reduce r (axis=0) -> [BLOCK_M]
 
     tl.store(out_ptr + col_offsets, acc, mask=mask_out)
-
 
 def kernel_function(x):
     # x: [128, 4096, 4095], reduce over dim=1, keepdim=True -> [128, 1, 4095]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
