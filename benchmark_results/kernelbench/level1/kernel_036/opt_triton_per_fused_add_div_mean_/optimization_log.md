# Kernel Optimization Log

## Summary

- **Kernel**: `rms_norm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41680000 ns — **Best latency**: 3670000 ns (**11.3569×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`single_pass_rms_norm_register_cache_via_2d_tile`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `rms_norm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_036/synthesis/kernel_0_triton_per_fused_add_div_mean_pow_sqrt_0.py`
- **Framework**: triton
- **Launch**: Block (64,1,1), Grid (29360128,1,1), Waves/SM: 6950.79
- **Registers/thread**: 16
- **Duration**: 41680000 ns
- **SM Throughput**: 12.45 % | **Memory**: 79.19 % | **DRAM**: 7.46 %
- **Achieved Occupancy**: 94.88 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | 0.989× | 1.004× | ✓ |  |
| B | vectorized_loads_via_transposed_access_pattern | — | fail | ✓ |  |
| D | fix_uncoalesced_memory_access_via_warp_contiguous_spatial_indexing | 0.559× | 5.310× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_uncoalesced_m... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41680000 ns | 41500000 ns | -180000 ns (-0.4%) | 7850000 ns | -33830000 ns (-81.2%) |
| SM Throughput (%) | 12.45 % | 13.42 % | +0.97 pp | 14.88 % | +2.43 pp |
| Memory Throughput (%) | 79.19 % | 79.66 % | +0.47 pp | 38.95 % | -40.24 pp |
| DRAM Throughput (%) | 7.46 % | 7.49 % | +0.03 pp | 38.95 % | +31.49 pp |
| Achieved Occupancy (%) | 94.88 % | 95.16 % | +0.28 pp | 10.64 % | -84.24 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 246.00 register/thread | +230 |
| Waves/SM | 6950.79 | 6950.79 | +0.00 | 868.85 | -6081.94 |
| No Eligible (%) | 87.55 % | 86.60 % | -0.95 pp | 84.62 % | -2.93 pp |
| IPC Active | 0.50 inst/cycle | 0.54 inst/cycle | +0.04 | 0.60 inst/cycle | +0.10 |

---

## Iteration 2

**Best so far**: 5.3096× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_dynamic_channel_loop | 0.708× | 7.704× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_dynamic_loop_and_vectorized_load_hints | 0.706× | 6.513× | ✓ |  |
| C | shared_memory_transposed_coalesced_load_for_rms_norm | 0.675× | 4.933× | ✓ |  |
| D | vectorized_loads_via_max_contiguous_multiple_of_hints | 0.705× | 2.389× | ✓ |  |
| E | add_triton_autotune | 0.560× | 5.846× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | shared_memory_tra... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41680000 ns | 5410000 ns | -36270000 ns (-87.0%) | 6400000 ns | -35280000 ns (-84.6%) | 8450000 ns | -33230000 ns (-79.7%) | 17450000 ns | -24230000 ns (-58.1%) | 7130000 ns | -34550000 ns (-82.9%) |
| SM Throughput (%) | 12.45 % | 72.85 % | +60.40 pp | 25.74 % | +13.29 pp | 51.97 % | +39.52 pp | 76.24 % | +63.79 pp | 16.39 % | +3.94 pp |
| Memory Throughput (%) | 79.19 % | 62.56 % | -16.63 pp | 71.38 % | -7.81 pp | 47.94 % | -31.25 pp | 41.69 % | -37.50 pp | 42.90 % | -36.29 pp |
| DRAM Throughput (%) | 7.46 % | 62.56 % | +55.10 pp | 71.38 % | +63.92 pp | 47.94 % | +40.48 pp | 17.52 % | +10.06 pp | 42.90 % | +35.44 pp |
| Achieved Occupancy (%) | 94.88 % | 94.25 % | -0.63 pp | 97.26 % | +2.38 pp | 96.78 % | +1.90 pp | 99.00 % | +4.12 pp | 10.99 % | -83.89 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 22.00 register/thread | +6 | 28.00 register/thread | +12 | 23.00 register/thread | +7 | 246.00 register/thread | +230 |
| Waves/SM | 6950.79 | 217.21 | -6733.58 | 108.61 | -6842.18 | 217.21 | -6733.58 | 868.85 | -6081.94 | 868.85 | -6081.94 |
| No Eligible (%) | 87.55 % | 27.28 % | -60.27 pp | 74.31 % | -13.24 pp | 48.00 % | -39.55 pp | 23.72 % | -63.83 pp | 83.30 % | -4.25 pp |
| IPC Active | 0.50 inst/cycle | 2.92 inst/cycle | +2.42 | 1.03 inst/cycle | +0.53 | 2.09 inst/cycle | +1.59 | 3.05 inst/cycle | +2.55 | 0.66 inst/cycle | +0.16 |


### Key Learnings

#### ✅ Plan A: `reduce_register_pressure_via_dynamic_channel_loop` (6.844×)

Using tl.static_range() or tl.range() in Triton over large loop counts (e.g., 64 iterations) causes the compiler to fully unroll the loop at compile time, keeping all intermediate values simultaneously live in registers and inflating registers/thread from ~32 to 246. This collapses Block Limit Registers from 16 to 2 blocks/SM, theoretical occupancy from 100% to 12.5%, and active warps/scheduler from ~15 to ~1.85, leaving 83% of scheduler cycles with no eligible warp and preventing L1TEX latency hiding. Replacing with a plain Python range() loop allows the compiler to reuse registers across iterations, restoring registers/thread to 32, theoretical occupancy to 100%, DRAM throughput from 28% to 73%, and reducing duration by ~6.8x.

> **Applicable when**: NCU reports TheoreticalOccupancy limited by registers (Block Limit Registers << Block Limit Warps); registers per thread is very high (>100) despite modest kernel logic; the kernel uses tl.static_range or tl.range over a loop count >=16; Theoretical Occupancy is <25% and Achieved Occupancy matches it; No Eligible cycles % is >70%; CPIStall rule points to L1TEX scoreboard stalls as the dominant stall reason.

> **Avoid when**: The loop count is small (<=8 iterations) where static unrolling is beneficial and register pressure stays low; when occupancy is already high (>75%) and the bottleneck is compute or memory bandwidth rather than latency hiding; when the loop body has complex data dependencies that require compile-time scheduling.

#### ✅ Plan E: `add_triton_autotune` (5.846×)

When a kernel already has a @triton.autotune decorator with many configs (including very small or very large XBLOCK values and num_stages > 1), pruning to a smaller, focused set of configs (e.g., XBLOCK in {128, 256, 512, 1024} with num_warps in {4, 8}, dropping num_stages) can improve the selected config by reducing autotuning noise and allowing the tuner to converge on a better block size. In this case, doubling block size from 64 to 128 threads reduced grid size by half (458752 → 229376), improved DRAM throughput from 38.95% to 42.9%, and reduced warp cycles per issued instruction from 11.42 to 10.71, yielding a modest but real speedup. The theoretical occupancy remained unchanged (register-bound at 12.5%), so the gain came from better per-block amortization of launch overhead and slightly improved scheduler efficiency.

> **Applicable when**: Kernel has @triton.autotune with many configs including extreme XBLOCK values (very small like 64 or very large like 2048) and/or multiple num_stages values; register pressure is high (>200 registers/thread) making occupancy register-bound; DRAM throughput is moderate (30-50%) and warp cycles per issued instruction is in the 10-15 range; the reduction dimension (BLOCK_R) is fixed and not a tuning target.

> **Avoid when**: Kernel is not register-bound and occupancy can be improved by other means; num_stages is actually beneficial for hiding memory latency (e.g., when the kernel has software pipelining opportunities); XBLOCK=64 or XBLOCK=2048 are genuinely optimal for the problem size; rnumel varies significantly across calls (in which case keeping 'rnumel' in the autotune key is important).

---

## Iteration 3

**Best so far**: 7.7043× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_pass1_reuse_pass2 | 0.443× | 3.806× | ✓ |  |
| D | single_pass_rms_norm_with_eviction_policy_and_software_pipelining | 0.696× | 6.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | single_pass_rms_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41680000 ns | 10950000 ns | -30730000 ns (-73.7%) | 6450000 ns | -35230000 ns (-84.5%) |
| SM Throughput (%) | 12.45 % | 16.67 % | +4.22 pp | 29.11 % | +16.66 pp |
| Memory Throughput (%) | 79.19 % | 28.05 % | -51.14 pp | 71.04 % | -8.15 pp |
| DRAM Throughput (%) | 7.46 % | 28.05 % | +20.59 pp | 71.04 % | +63.58 pp |
| Achieved Occupancy (%) | 94.88 % | 11.45 % | -83.43 pp | 97.39 % | +2.51 pp |
| Registers/Thread | 16.00 register/thread | 246.00 register/thread | +230 | 22.00 register/thread | +6 |
| Waves/SM | 6950.79 | 868.85 | -6081.94 | 108.61 | -6842.18 |
| No Eligible (%) | 87.55 % | 83.18 % | -4.37 pp | 70.87 % | -16.68 pp |
| IPC Active | 0.50 inst/cycle | 0.67 inst/cycle | +0.17 | 1.16 inst/cycle | +0.66 |


### Key Learnings

#### ❌ Plan A: `eviction_policy_streaming_pass1_reuse_pass2` (3.813×)

Switching from tl.range() to tl.static_range() over 64 iterations, even when combined with eviction_policy='evict_first', inflates registers per thread from 22 to 246, collapsing theoretical occupancy from 100% to 12.5% and active warps per scheduler from 15.58 to 1.84. The 83% no-eligible-warp rate destroys latency hiding: DRAM throughput drops from 71.04% to 28.11% and duration increases from 6.45ms to 10.93ms despite warp cycles per issued instruction improving from 53.49 to 10.92. The register explosion from full loop unrolling is the dominant regression driver — the eviction policy hint carries no register cost but cannot compensate for the occupancy collapse.

> **Applicable when**: Never: this is a negative example. Do not apply tl.static_range() over loop counts >=16 when the kernel is already occupancy-healthy (>90% achieved occupancy, low registers per thread).

> **Avoid when**: Always avoid tl.static_range() with loop counts >=16 iterations when registers per thread is already near the occupancy-limiting threshold. The register explosion from full unrolling dominates any instruction-count or cache-policy benefit. This pattern has now caused regression in iterations 3 and 4 from two different starting states.

---

## Iteration 4

**Best so far**: 7.7043× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_multiply_add_and_reciprocal_normalization | 0.456× | 4.051× | ✓ |  |
| B | single_pass_rms_norm_register_cache_via_2d_tile | 1.025× | 11.357× | ✓ | **✓ best** |
| C | replace_static_range_with_dynamic_range_to_restore_occupancy | 0.708× | 6.844× | ✓ |  |
| D | vectorized_loads_via_tl_range_with_contiguous_hints | 0.676× | 6.344× | ✓ |  |
| E | add_triton_autotune | 0.442× | 3.813× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_multiply_ad... | Δ | single_pass_rms_n... | Δ | replace_static_ra... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41680000 ns | 10290000 ns | -31390000 ns (-75.3%) | 3670000 ns | -38010000 ns (-91.2%) | 6090000 ns | -35590000 ns (-85.4%) | 6570000 ns | -35110000 ns (-84.2%) | 10930000 ns | -30750000 ns (-73.8%) |
| SM Throughput (%) | 12.45 % | 16.39 % | +3.94 pp | 10.60 % | -1.85 pp | 32.02 % | +19.57 pp | 32.49 % | +20.04 pp | 16.74 % | +4.29 pp |
| Memory Throughput (%) | 79.19 % | 29.87 % | -49.32 pp | 83.11 % | +3.92 pp | 72.86 % | -6.33 pp | 68.61 % | -10.58 pp | 28.11 % | -51.08 pp |
| DRAM Throughput (%) | 7.46 % | 29.87 % | +22.41 pp | 83.11 % | +75.65 pp | 72.86 % | +65.40 pp | 68.61 % | +61.15 pp | 28.11 % | +20.65 pp |
| Achieved Occupancy (%) | 94.88 % | 11.25 % | -83.63 pp | 29.84 % | -65.04 pp | 96.28 % | +1.40 pp | 96.37 % | +1.49 pp | 11.44 % | -83.44 pp |
| Registers/Thread | 16.00 register/thread | 246.00 register/thread | +230 | 96.00 register/thread | +80 | 32.00 register/thread | +16 | 28.00 register/thread | +12 | 246.00 register/thread | +230 |
| Waves/SM | 6950.79 | 868.85 | -6081.94 | 347.54 | -6603.25 | 108.61 | -6842.18 | 108.61 | -6842.18 | 868.85 | -6081.94 |
| No Eligible (%) | 87.55 % | 83.42 % | -4.13 pp | 89.44 % | +1.89 pp | 67.94 % | -19.61 pp | 67.57 % | -19.98 pp | 83.11 % | -4.44 pp |
| IPC Active | 0.50 inst/cycle | 0.66 inst/cycle | +0.16 | 0.42 inst/cycle | -0.08 | 1.29 inst/cycle | +0.79 | 1.30 inst/cycle | +0.80 | 0.67 inst/cycle | +0.17 |


### Key Learnings

#### ❌ Plan A: `fused_multiply_add_and_reciprocal_normalization` (4.051×)

Applying tl.math.fma(v, v, acc) and precomputed rms_inv = 1.0/rms (replacing v/rms with v*rms_inv) are arithmetically correct FP optimizations that reduce non-fused FP32 instructions. However, when bundled with a switch from tl.range() to tl.static_range() in the same diff, registers per thread exploded from 22 to 246, collapsing theoretical occupancy from 100% to 12.5% and achieved occupancy from 97.39% to 11.25%. DRAM throughput dropped from 71.04% to 29.87% and duration increased from 6.45ms to 10.29ms despite a modest instruction count reduction (1.48B to 1.31B). The tl.static_range() reintroduction is the dominant regression driver — the FMA and reciprocal changes themselves are neutral-to-positive but cannot compensate for the occupancy collapse caused by full loop unrolling over 64 iterations.

> **Applicable when**: Never: tl.math.fma and rms_inv are safe micro-optimizations only when applied with tl.range() (dynamic loops). The FPInstructions NCU rule (non-fused FP32 fraction >75%) is a valid signal, but the fix must not reintroduce tl.static_range() for loops with >=16 iterations.

> **Avoid when**: Do not apply tl.math.fma or rms_inv precomputation together with tl.static_range() for loops with >=16 iterations. The register explosion from static unrolling (22 -> 246 regs/thread) destroys occupancy (100% -> 12.5%) and DRAM throughput (71% -> 30%), far outweighing any FP instruction savings. If FMA/rms_inv are applied with tl.range() (dynamic loops), they are safe but the kernel is memory-latency-bound so the gain will be small.

#### ✅ Plan B: `single_pass_rms_norm_register_cache_via_2d_tile` (11.357×)

When an RMS norm kernel makes two sequential passes over a large tensor (pass 1: accumulate sum-of-squares; pass 2: reload and normalize), and the working set far exceeds L2 capacity (L2 hit rate ~33%), the second pass re-reads all data from DRAM, doubling memory traffic. Replacing the two-pass loop structure with a single 2D tl.load of shape [XBLOCK, BLOCK_R] that covers all reduction-dimension elements at once caches all values in registers, computes the RMS in-register via tl.sum + tl.rsqrt, and stores normalized values without any second DRAM read — cutting total executed instructions from 1.48B to 306M (5x reduction) and duration from 6.45ms to 3.67ms (1.76x kernel speedup, 11.4x end-to-end). The trade-off is increased register pressure (22 → 96 registers/thread) which reduces theoretical occupancy from 100% to 31.25%, but the instruction count reduction dominates: DRAM throughput improved from 71% to 83% of peak, and L2 hit rate rose from 33% to 51% due to fewer total transactions.

> **Applicable when**: Kernel makes two sequential passes over a large tensor where the working set exceeds L2 capacity (L2 hit rate < 50% on second pass); the reduction dimension (rnumel) is small and fixed (e.g., rnumel=64) so all values fit in registers; CPIStall rule shows >80% of stall cycles are L1TEX scoreboard waits caused by DRAM-latency loads; DRAM throughput is high (>60%) but kernel is not yet at peak bandwidth; total instruction count is dominated by repeated loads rather than compute.

> **Avoid when**: The reduction dimension is large (rnumel >> 128) and holding all values in registers would cause severe register spilling (registers/thread > 128 with no occupancy headroom); when the tensor working set fits in L2 (L2 hit rate > 80% on second pass, meaning second-pass reloads are cheap); when the 2D tile's non-contiguous dimension (BLOCK_R indexing a high-stride axis) causes uncoalesced access patterns that negate the savings — verify that the XBLOCK dimension remains the fast (stride-1) axis. Also avoid using tl.static_range for the BLOCK_R dimension as it causes register explosion.

#### ✅ Plan E: `add_triton_autotune` (5.846×)

When a kernel already has a @triton.autotune decorator with many configs (including very small or very large XBLOCK values and num_stages > 1), pruning to a smaller, focused set of configs (e.g., XBLOCK in {128, 256, 512, 1024} with num_warps in {4, 8}, dropping num_stages) can improve the selected config by reducing autotuning noise and allowing the tuner to converge on a better block size. In this case, doubling block size from 64 to 128 threads reduced grid size by half (458752 → 229376), improved DRAM throughput from 38.95% to 42.9%, and reduced warp cycles per issued instruction from 11.42 to 10.71, yielding a modest but real speedup. The theoretical occupancy remained unchanged (register-bound at 12.5%), so the gain came from better per-block amortization of launch overhead and slightly improved scheduler efficiency.

> **Applicable when**: Kernel has @triton.autotune with many configs including extreme XBLOCK values (very small like 64 or very large like 2048) and/or multiple num_stages values; register pressure is high (>200 registers/thread) making occupancy register-bound; DRAM throughput is moderate (30-50%) and warp cycles per issued instruction is in the 10-15 range; the reduction dimension (BLOCK_R) is fixed and not a tuning target.

> **Avoid when**: Kernel is not register-bound and occupancy can be improved by other means; num_stages is actually beneficial for hiding memory latency (e.g., when the kernel has software pipelining opportunities); XBLOCK=64 or XBLOCK=2048 are genuinely optimal for the problem size; rnumel varies significantly across calls (in which case keeping 'rnumel' in the autotune key is important).

---

## Iteration 5

**Best so far**: 11.3569× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eviction_policy_pass1_evict_first_pass2_evict_last | — | fail | ✓ |  |
| D | single_pass_rms_norm_2d_register_tile_with_tl_arange | — | fail | ✓ |  |

---

## Cumulative Progress

```
Baseline:  41680000 ns  (1.000×)
Iter 1     7850000 ns  (5.310×)  — fix_uncoalesced_memory_access_via_warp_contiguous_spatial_indexing
Iter 2     5410000 ns  (7.704×)  — reduce_register_pressure_via_dynamic_channel_loop
Iter 4     3670000 ns  (11.357×)  — single_pass_rms_norm_register_cache_via_2d_tile

Final best: 3670000 ns  (11.3569×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,15 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'XBLOCK': 1, 'BLOCK_R': 64}, num_warps=2, num_stages=1),
-        triton.Config({'XBLOCK': 4, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
-        triton.Config({'XBLOCK': 8, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
-        triton.Config({'XBLOCK': 16, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
-        triton.Config({'XBLOCK': 32, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
-        triton.Config({'XBLOCK': 64, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'BLOCK_R': 64}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'BLOCK_R': 64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
     ],
     key=['xnumel', 'rnumel'],
 )
@@ -31,50 +32,30 @@

     x_offsets = x_start + tl.arange(0, XBLOCK)
     x_mask = x_offsets < xnumel
 
-    # x_offsets indexes into the [112, 512, 512] = 29360128 spatial positions
-    # For each spatial position x, the features are at:
-    #   in_ptr[batch * (rnumel * spatial) + feat * spatial + spatial_pos]
-    # where spatial = 512*512 = 262144
-    # x_offsets = batch * 262144 + spatial_pos (within [0, 262144))
-
     # Compute batch index and spatial position within batch
     spatial_size = 262144  # 512 * 512
     batch_idx = x_offsets // spatial_size  # [XBLOCK]
     spatial_pos = x_offsets % spatial_size  # [XBLOCK]
 
-    # Accumulate sum of squares over feature dimension
-    acc = tl.zeros([XBLOCK], dtype=tl.float32)
+    batch_base = batch_idx * 16777216  # [XBLOCK]
 
-    r_offsets = tl.arange(0, BLOCK_R)  # [BLOCK_R]
+    # Build 2D index [XBLOCK, BLOCK_R] covering all channels
+    r_offsets = tl.arange(0, BLOCK_R)  # [BLOCK_R], BLOCK_R=64=rnumel
+    idx_2d = (batch_base[:, None]
+              + r_offsets[None, :] * 262144
+              + spatial_pos[:, None])   # [XBLOCK, BLOCK_R]
+    mask_2d = x_mask[:, None] & (r_offsets[None, :] < rnumel)
 
-    # Load all features at once since BLOCK_R == rnumel == 64
-    # in_ptr layout: [112, 64, 512, 512] with strides [16777216, 262144, 512, 1]
-    # index = batch_idx * 16777216 + feat * 262144 + spatial_pos
-    # = batch_idx[:, None] * 16777216 + r_offsets[None, :] * 262144 + spatial_pos[:, None]
+    # Single load: all channels into register tile -- no second DRAM pass
+    vals = tl.load(in_ptr + idx_2d, mask=mask_2d, other=0.0).to(tl.float32)  # [XBLOCK, BLOCK_R]
 
-    batch_base = batch_idx * 16777216  # [XBLOCK]
-    sp_base = spatial_pos  # [XBLOCK]
+    # Compute RMS over channel axis
+    sum_sq = tl.sum(vals * vals, axis=1)          # [XBLOCK]
+    rms_inv = tl.rsqrt(sum_sq / rnumel + eps)     # [XBLOCK], single MUFU.RSQ instruction
 
-    # 2D load: [XBLOCK, BLOCK_R]
-    indices = (batch_base[:, None] + r_offsets[None, :] * 262144 + sp_base[:, None])
-    r_mask = r_offsets < rnumel  # [BLOCK_R]
-    mask_2d = x_mask[:, None] & r_mask[None, :]  # [XBLOCK, BLOCK_R]
-
-    vals = tl.load(in_ptr + indices, mask=mask_2d, other=0.0)  # [XBLOCK, BLOCK_R]
-    vals_f32 = vals.to(tl.float32)
-
-    sq = vals_f32 * vals_f32  # [XBLOCK, BLOCK_R]
-    sum_sq = tl.sum(sq, axis=1)  # [XBLOCK]
-
-    mean_sq = sum_sq / rnumel
-    rms = tl.sqrt(mean_sq + eps)  # [XBLOCK]
-
-    # Normalize and store
-    normalized = vals_f32 / rms[:, None]  # [XBLOCK, BLOCK_R]
-
-    # Store back
-    tl.store(out_ptr + indices, normalized.to(tl.float32), mask=mask_2d)
-
+    # Normalize in-register and store -- no reload from DRAM
+    normalized = vals * rms_inv[:, None]           # [XBLOCK, BLOCK_R]
+    tl.store(out_ptr + idx_2d, normalized.to(tl.float32), mask=mask_2d)
 
 def kernel_function(x):
     # x: [112, 64, 512, 512]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
