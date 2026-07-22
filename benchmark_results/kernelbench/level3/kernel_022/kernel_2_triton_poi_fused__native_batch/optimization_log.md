# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17090 ns — **Best latency**: 16900 ns (**1.0112×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restructure_iteration_space_to_eliminate_modulo_and_gather`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 17
- **Duration**: 17090 ns
- **SM Throughput**: 50.12 % | **Memory**: 22.24 % | **DRAM**: 14.38 %
- **Achieved Occupancy**: 56.66 %
- **IPC Active**: 2.30 inst/cycle | **No Eligible**: 41.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.953× | 0.916× | ✓ |  |
| B | autotune_xblock_tile_size_for_latency_hiding | 1.025× | 0.395× | ✓ |  |
| C | add_autotune_vectorized_loads_larger_xblock | 0.761× | 0.708× | ✓ |  |
| D | autotune_xblock_to_hide_l1tex_latency | 1.026× | 0.394× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | add_autotune_vect... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 18660 ns | +1570 ns (+9.2%) | 43300 ns | +26210 ns (+153.4%) | 24130 ns | +7040 ns (+41.2%) | 43330 ns | +26240 ns (+153.5%) |
| SM Throughput (%) | 50.12 % | 70.78 % | +20.66 pp | 83.73 % | +33.61 pp | 79.16 % | +29.04 pp | 83.57 % | +33.45 pp |
| Memory Throughput (%) | 22.24 % | 27.39 % | +5.15 pp | 19.55 % | -2.69 pp | 23.34 % | +1.10 pp | 19.46 % | -2.78 pp |
| DRAM Throughput (%) | 14.38 % | 13.17 % | -1.21 pp | 5.67 % | -8.71 pp | 10.17 % | -4.21 pp | 5.66 % | -8.72 pp |
| Achieved Occupancy (%) | 56.66 % | 68.03 % | +11.37 pp | 86.17 % | +29.51 pp | 86.55 % | +29.89 pp | 86.12 % | +29.46 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 22.27 | +11.13 | 11.14 | +0.00 | 22.27 | +11.13 |
| No Eligible (%) | 41.77 % | 17.50 % | -24.27 pp | 12.35 % | -29.42 pp | 13.76 % | -28.01 pp | 12.18 % | -29.59 pp |
| IPC Active | 2.30 inst/cycle | 3.33 inst/cycle | +1.03 | 3.51 inst/cycle | +1.21 | 3.44 inst/cycle | +1.14 | 3.51 inst/cycle | +1.21 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_tile_size_for_latency_hiding` (0.395×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) caused registers per thread to jump from 17 to 32, halving the number of blocks that can reside per SM (Block Limit Registers dropped from 21 to 8). Although occupancy improved from 56.7% to 86.2% and idle-slot cycles fell from 41.77% to 12.35%, the executed instruction count exploded from 6.8M to 28.5M (4.2x), and memory throughput collapsed from 705 GB/s to 278 GB/s, causing a 2.5x slowdown. The autotune decorator itself (or the addition of xmask bounds checks) triggered a more complex code path that inflated register usage and instruction count far beyond what the latency-hiding benefit could recover.

> **Applicable when**: Never — this specific combination caused a severe regression. The pattern of adding autotune with larger XBLOCK is generally valid for latency-hiding, but only when: (1) the grid lambda is correctly updated to use meta['XBLOCK'], (2) adding xmask does not inflate register count, and (3) the autotune overhead is amortized over many calls.

> **Avoid when**: When the kernel body uses tl.full([XBLOCK], True) masks (no bounds check) and xnumel is exactly divisible by the base XBLOCK — adding xmask = xindex < xnumel introduces a data-dependent predicate that the compiler cannot eliminate, inflating register usage. Also avoid when the grid is not updated to use meta['XBLOCK'] from autotune, causing the grid to remain fixed at the original tile size while the kernel processes a larger tile, leading to incorrect results or wasted work. Avoid when instruction count is already low and the kernel is memory-throughput-bound rather than latency-bound.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | algebraic_simplification_to_enable_fma_fusion | 1.019× | 1.002× | ✓ |  |
| B | replace_integer_reciprocal_with_rsqrt_to_enable_fma_fusion | 0.965× | 0.957× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_instructions | 1.027× | 0.672× | ✓ |  |
| D | simplify_compute_chain_for_fma_fusion_and_remove_autotune_register_bloat | 1.029× | 1.011× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.015× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | algebraic_simplif... | Δ | replace_integer_r... | Δ | fuse_fp32_multipl... | Δ | simplify_compute_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 17060 ns | -30 ns (-0.2%) | 17860 ns | +770 ns (+4.5%) | 25440 ns | +8350 ns (+48.9%) | 16900 ns | -190 ns (-1.1%) | 17090 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 50.12 % | 40.62 % | -9.50 pp | 49.66 % | -0.46 pp | 69.51 % | +19.39 pp | 34.07 % | -16.05 pp | 59.63 % | +9.51 pp |
| Memory Throughput (%) | 22.24 % | 30.46 % | +8.22 pp | 33.05 % | +10.81 pp | 34.28 % | +12.04 pp | 22.28 % | +0.04 pp | 22.49 % | +0.25 pp |
| DRAM Throughput (%) | 14.38 % | 14.40 % | +0.02 pp | 13.74 % | -0.64 pp | 9.64 % | -4.74 pp | 14.54 % | +0.16 pp | 14.36 % | -0.02 pp |
| Achieved Occupancy (%) | 56.66 % | 32.69 % | -23.97 pp | 37.23 % | -19.43 pp | 80.44 % | +23.78 pp | 40.41 % | -16.25 pp | 59.42 % | +2.76 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 22.27 | +11.13 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.77 % | 50.31 % | +8.54 pp | 42.00 % | +0.23 pp | 24.84 % | -16.93 pp | 67.70 % | +25.93 pp | 31.16 % | -10.61 pp |
| IPC Active | 2.30 inst/cycle | 1.97 inst/cycle | -0.33 | 2.37 inst/cycle | +0.07 | 2.99 inst/cycle | +0.69 | 1.32 inst/cycle | -0.98 | 2.75 inst/cycle | +0.45 |


### Key Learnings

#### ❌ Plan C: `fuse_fp32_multiply_add_to_fma_instructions` (0.672×)

Replacing the int-division anti-pattern and separate FMUL/FADD with libdevice.rsqrt() and libdevice.fma() is a sound arithmetic optimization, but the accompanying autotune config change (adding XBLOCK=1024/512 configs and removing XBLOCK=2048) caused the autotuner to select XBLOCK=256 with block_size=256, which doubled registers per thread from 24 to 32. This collapsed Block Limit Registers from 21 to 8 blocks/SM, doubled waves per SM from 11.14 to 22.27, increased total executed instructions from 10.3M to 14.0M, and degraded memory throughput from 645 GB/s to 473 GB/s — causing a 36% slowdown despite the FMA arithmetic improvement. The FMA rewrite itself may be correct, but the block size regression dominated the outcome.

> **Applicable when**: The libdevice.rsqrt() + libdevice.fma() arithmetic rewrite is applicable when: (1) the kernel has an int-division anti-pattern (int(1)/float) that prevents PTX FMA fusion, (2) FPInstructions NCU rule reports a high non-fused-to-fused ratio (>2:1), and (3) the autotune config set is NOT changed simultaneously — the arithmetic rewrite must be isolated from block size changes.

> **Avoid when**: Do NOT combine the FMA arithmetic rewrite with autotune config changes that increase block size. When registers per thread are already at 24+, adding larger XBLOCK configs risks the autotuner selecting a config that pushes registers to 32, collapsing Block Limit Registers and doubling waves per SM. The register pressure regression dominates any FMA throughput gain. Also avoid when the kernel is already memory-throughput-bound (L1 hit rate >70%, memory throughput >600 GB/s) — FMA fusion primarily helps compute-bound kernels.

---

## Iteration 3

**Best so far**: 1.0112× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.042× | 1.011× | ✓ |  |
| B | hoist_channel_invariant_rsqrt_out_of_per_element_loop | 1.033× | 0.998× | ✓ |  |
| C | fix_launch_grid_to_enable_autotune_with_pipelined_stages | 0.966× | 0.952× | ✓ |  |
| D | reduce_block_size_to_restore_register_limited_occupancy | 1.051× | 1.011× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | hoist_channel_inv... | Δ | fix_launch_grid_t... | Δ | reduce_block_size... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16900 ns | -190 ns (-1.1%) | 17120 ns | +30 ns (+0.2%) | 17950 ns | +860 ns (+5.0%) | 16900 ns | -190 ns (-1.1%) |
| SM Throughput (%) | 50.12 % | 33.90 % | -16.22 pp | 67.44 % | +17.32 pp | 49.78 % | -0.34 pp | 35.88 % | -14.24 pp |
| Memory Throughput (%) | 22.24 % | 22.21 % | -0.03 pp | 45.60 % | +23.36 pp | 33.17 % | +10.93 pp | 22.40 % | +0.16 pp |
| DRAM Throughput (%) | 14.38 % | 14.54 % | +0.16 pp | 14.34 % | -0.04 pp | 13.68 % | -0.70 pp | 14.54 % | +0.16 pp |
| Achieved Occupancy (%) | 56.66 % | 40.97 % | -15.69 pp | 43.05 % | -13.61 pp | 37.30 % | -19.36 pp | 42.40 % | -14.26 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.77 % | 67.05 % | +25.28 pp | 53.49 % | +11.72 pp | 41.36 % | -0.41 pp | 57.75 % | +15.98 pp |
| IPC Active | 2.30 inst/cycle | 1.32 inst/cycle | -0.98 | 1.87 inst/cycle | -0.43 | 2.37 inst/cycle | +0.07 | 1.66 inst/cycle | -0.64 |

---

## Iteration 4

**Best so far**: 1.0112× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_runtime_modulo_with_constexpr_channel_index_to_eliminate_integer_division_stall | 1.040× | 0.995× | ✓ |  |
| B | add_evict_first_to_streaming_tensor_load_to_improve_l2_hit_rate | 1.038× | 0.998× | ✓ |  |
| C | restructure_iteration_space_to_eliminate_modulo_and_gather | 1.028× | 1.011× | ✓ |  |
| D | precompute_per_channel_rsqrt_to_eliminate_per_element_mufu_stall | 0.862× | 0.813× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_runtime_m... | Δ | add_evict_first_t... | Δ | restructure_itera... | Δ | precompute_per_ch... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 17180 ns | +90 ns (+0.5%) | 17120 ns | +30 ns (+0.2%) | 16900 ns | -190 ns (-1.1%) | 21020 ns | +3930 ns (+23.0%) |
| SM Throughput (%) | 50.12 % | 34.07 % | -16.05 pp | 38.29 % | -11.83 pp | 34.04 % | -16.08 pp | 65.06 % | +14.94 pp |
| Memory Throughput (%) | 22.24 % | 22.30 % | +0.06 pp | 22.11 % | -0.13 pp | 21.83 % | -0.41 pp | 75.04 % | +52.80 pp |
| DRAM Throughput (%) | 14.38 % | 14.28 % | -0.10 pp | 14.34 % | -0.04 pp | 10.89 % | -3.49 pp | 11.69 % | -2.69 pp |
| Achieved Occupancy (%) | 56.66 % | 40.84 % | -15.82 pp | 43.74 % | -12.92 pp | 40.96 % | -15.70 pp | 87.45 % | +30.79 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 31.00 register/thread | +14 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.77 % | 67.69 % | +25.92 pp | 54.50 % | +12.73 pp | 65.86 % | +24.09 pp | 32.72 % | -9.05 pp |
| IPC Active | 2.30 inst/cycle | 1.28 inst/cycle | -1.02 | 1.82 inst/cycle | -0.48 | 1.37 inst/cycle | -0.93 | 2.67 inst/cycle | +0.37 |

---

## Iteration 5

**Best so far**: 1.0112× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | revert_to_flat_1d_layout_with_increased_num_warps_for_l1tex_latency_hiding | 1.030× | 1.010× | ✓ |  |
| C | halve_block_size_to_double_resident_blocks_per_sm_for_l1tex_latency_hiding | 1.042× | 1.010× | ✓ |  |
| D | revert_tl_gather_to_per_element_evict_last_loads | 1.042× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_to_flat_1d... | Δ | halve_block_size_... | Δ | revert_tl_gather_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17090 ns | 16930 ns | -160 ns (-0.9%) | 16930 ns | -160 ns (-0.9%) | 17020 ns | -70 ns (-0.4%) |
| SM Throughput (%) | 50.12 % | 34.02 % | -16.10 pp | 34.09 % | -16.03 pp | 34.01 % | -16.11 pp |
| Memory Throughput (%) | 22.24 % | 22.28 % | +0.04 pp | 22.29 % | +0.05 pp | 22.22 % | -0.02 pp |
| DRAM Throughput (%) | 14.38 % | 14.49 % | +0.11 pp | 14.50 % | +0.12 pp | 14.45 % | +0.07 pp |
| Achieved Occupancy (%) | 56.66 % | 40.90 % | -15.76 pp | 40.95 % | -15.71 pp | 40.87 % | -15.79 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 16.00 register/thread | -1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.77 % | 66.80 % | +25.03 pp | 68.02 % | +26.25 pp | 67.26 % | +25.49 pp |
| IPC Active | 2.30 inst/cycle | 1.31 inst/cycle | -0.99 | 1.27 inst/cycle | -1.03 | 1.32 inst/cycle | -0.98 |

---

## Cumulative Progress

```
Baseline:  17090 ns  (1.000×)
Iter 2     16900 ns  (1.011×)  — simplify_compute_chain_for_fma_fusion_and_remove_autotune_register_bloat

Final best: 16900 ns  (1.0112×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 3010560
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x2 = xindex
-    x0 = (xindex % 24)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    N_SPATIAL: tl.constexpr = 125440  # 10 * 112 * 112
+    CBLOCK: tl.constexpr = 32         # power-of-2 >= 24
+    SP_BLOCK: tl.constexpr = XBLOCK // CBLOCK  # spatial rows per CTA = 4
+    # Load channel params once per CTA
+    c_idx    = tl.arange(0, CBLOCK)                           # [CBLOCK]
+    c_mask   = c_idx < 24
+    mean_c   = tl.load(in_ptr0 + c_idx, mask=c_mask, other=0.0)   # [CBLOCK]
+    var_c    = tl.load(in_ptr1 + c_idx, mask=c_mask, other=0.0)   # [CBLOCK]
+    weight_c = tl.load(in_ptr2 + c_idx, mask=c_mask, other=0.0)   # [CBLOCK]
+    bias_c   = tl.load(in_ptr3 + c_idx, mask=c_mask, other=0.0)   # [CBLOCK]
+    scale_c  = libdevice.rsqrt(var_c + 1e-5) * weight_c           # [CBLOCK]
+    # 2D tile: [SP_BLOCK, CBLOCK]
+    pid = tl.program_id(0)
+    sp_offset = pid * SP_BLOCK
+    sp_idx = sp_offset + tl.arange(0, SP_BLOCK)[:, None]    # [SP_BLOCK, 1]
+    ch_idx = tl.arange(0, CBLOCK)[None, :]                  # [1, CBLOCK]
+    sp_mask = sp_idx < N_SPATIAL
+    ch_mask = ch_idx < 24
+    mask2d  = sp_mask & ch_mask                              # [SP_BLOCK, CBLOCK]
+    flat_idx = sp_idx * 24 + ch_idx                         # [SP_BLOCK, CBLOCK]
+    tmp0 = tl.load(in_out_ptr0 + flat_idx, mask=mask2d)
+    # Broadcast channel params -- no gather, no modulo
+    mean_val  = mean_c[None, :]   # [1, CBLOCK] broadcast
+    scale_val = scale_c[None, :]
+    bias_val  = bias_c[None, :]
+    normed = (tmp0 - mean_val) * scale_val + bias_val
+    result = tl.maximum(tl.full([1, 1], 0.0, tl.float32), normed)
+    tl.store(in_out_ptr0 + flat_idx, result, mask=mask2d)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
