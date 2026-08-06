# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 43040 ns — **Best latency**: 39070 ns (**1.1016×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_k_to_eliminate_smem_pressure_and_improve_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_36_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.24
- **Registers/thread**: 94
- **Duration**: 43040 ns
- **SM Throughput**: 9.52 % | **Memory**: 63.49 % | **DRAM**: 0.88 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_eliminate_bank_conflicts_and_improve_occupancy | 0.592× | 0.575× | ✓ |  |
| C | autotune_tile_sizes_to_eliminate_shared_memory_bank_conflicts | 0.592× | 0.577× | ✓ |  |
| D | increase_tile_sizes_with_autotune | 0.331× | 0.310× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43040 ns | 74820 ns | +31780 ns (+73.8%) | 74660 ns | +31620 ns (+73.5%) | 139010 ns | +95970 ns (+223.0%) |
| SM Throughput (%) | 9.52 % | 16.40 % | +6.88 pp | 16.42 % | +6.90 pp | 17.88 % | +8.36 pp |
| Memory Throughput (%) | 63.49 % | 63.54 % | +0.05 pp | 63.61 % | +0.12 pp | 40.64 % | -22.85 pp |
| DRAM Throughput (%) | 0.88 % | 0.51 % | -0.37 pp | 0.51 % | -0.37 pp | 0.28 % | -0.60 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 255.00 register/thread | +161 |
| Waves/SM | 0.24 | 0.18 | -0.06 | 0.18 | -0.06 | 0.36 | +0.12 |
| No Eligible (%) | 86.27 % | 76.62 % | -9.65 pp | 76.61 % | -9.66 pp | 73.05 % | -13.22 pp |
| IPC Active | 0.55 inst/cycle | 0.93 inst/cycle | +0.38 | 0.93 inst/cycle | +0.38 | 1.08 inst/cycle | +0.53 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_and_pipeline_shared_memory_stalls | 0.593× | 0.577× | ✓ |  |
| B | promote_tile_sizes_to_autotune_constexpr_args_with_dynamic_grid | 0.999× | 1.001× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tiles_with_autotune | 1.087× | 1.105× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | promote_tile_size... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43040 ns | 74620 ns | +31580 ns (+73.4%) | 42980 ns | -60 ns (-0.1%) | 38940 ns | -4100 ns (-9.5%) |
| SM Throughput (%) | 9.52 % | 16.45 % | +6.93 pp | 9.52 % | +0.00 pp | 9.47 % | -0.05 pp |
| Memory Throughput (%) | 63.49 % | 63.73 % | +0.24 pp | 63.53 % | +0.04 pp | 60.97 % | -2.52 pp |
| DRAM Throughput (%) | 0.88 % | 0.51 % | -0.37 pp | 0.88 % | +0.00 pp | 0.97 % | +0.09 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 94.00 register/thread | +0 | 64.00 register/thread | -30 |
| Waves/SM | 0.24 | 0.18 | -0.06 | 0.24 | +0.00 | 0.09 | -0.15 |
| No Eligible (%) | 86.27 % | 76.59 % | -9.68 pp | 86.24 % | -0.03 pp | 86.26 % | -0.01 pp |
| IPC Active | 0.55 inst/cycle | 0.93 inst/cycle | +0.38 | 0.55 inst/cycle | +0.00 | 0.55 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.1053× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_via_num_stages_autotune | 1.087× | 0.862× | ✓ | **✓ best** |
| B | eliminate_shared_memory_bank_conflicts_via_b_tile_transpose | 0.997× | 1.000× | ✓ |  |
| C | autotune_with_dynamic_grid_and_register_aware_configs | 1.018× | 0.912× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | eliminate_shared_... | Δ | autotune_with_dyn... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43040 ns | 49950 ns | +6910 ns (+16.1%) | 43040 ns | +0 ns (+0.0%) | 47200 ns | +4160 ns (+9.7%) |
| SM Throughput (%) | 9.52 % | 6.77 % | -2.75 pp | 9.46 % | -0.06 pp | 7.35 % | -2.17 pp |
| Memory Throughput (%) | 63.49 % | 49.27 % | -14.22 pp | 63.47 % | -0.02 pp | 52.62 % | -10.87 pp |
| DRAM Throughput (%) | 0.88 % | 0.75 % | -0.13 pp | 0.88 % | +0.00 pp | 0.80 % | -0.08 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 94.00 register/thread | 62.00 register/thread | -32 | 94.00 register/thread | +0 | 96.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.09 | -0.15 | 0.24 | +0.00 | 0.12 | -0.12 |
| No Eligible (%) | 86.27 % | 90.08 % | +3.81 pp | 86.32 % | +0.05 pp | 78.80 % | -7.47 pp |
| IPC Active | 0.55 inst/cycle | 0.40 inst/cycle | -0.15 | 0.54 inst/cycle | -0.01 | 0.42 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan A: `software_pipeline_via_num_stages_autotune` (0.862×)

Adding @triton.autotune with varying num_stages (1–4) and num_warps (4, 8) to a small-grid GEMM kernel (96 blocks, 0.09 waves/SM) caused a regression: duration increased from 38.94µs to 49.95µs (0.86x). The autotune selected a config that halved dynamic shared memory per block (16.4 KB → 8.2 KB) and slightly reduced registers (64 → 62), but warp cycles per issued instruction worsened from 7.3 to 10.03 and No Eligible scheduler cycles increased from 86.26% to 90.08%. The 12-way shared memory bank conflicts remained completely unchanged (3,612,672 excessive wavefronts in both before and after), confirming that num_stages pipelining does not address structural bank conflicts caused by column-major B-matrix access strides. When the grid is severely underutilized (0.09 waves/SM) and bank conflicts are structural, software pipelining adds overhead without hiding the dominant MIO stall source.

> **Applicable when**: Never apply num_stages autotune as the primary optimization when: (1) shared memory bank conflicts are structural (12-way, >75% excessive wavefronts) due to non-unit strides in the B matrix, (2) the grid is severely underutilized (<0.1 waves/SM), and (3) achieved occupancy is already stuck at a low value (6.25%) regardless of pipeline depth.

> **Avoid when**: Do not apply @triton.autotune with num_stages variants when the dominant stall source is structural shared memory bank conflicts (UncoalescedSharedAccess rule, 12-way conflicts, >75% excessive wavefronts). Software pipelining hides global memory latency but cannot hide shared memory bank conflict latency — the bank conflicts must be eliminated structurally (e.g., padding shared memory, changing access order, or transposing the tile layout) before pipelining can help. Also avoid when grid << SM count (waves_per_sm < 0.1), as the autotune overhead may select a suboptimal config.

---

## Iteration 4

**Best so far**: 1.1053× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_bank_conflicts_via_direct_b_load_without_transpose | 0.993× | 0.999× | ✓ |  |
| B | pad_shared_memory_b_tile_to_eliminate_bank_conflicts | 1.020× | 0.911× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_b_tile_transpose_with_tl_trans | 1.095× | 1.102× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_bank_co... | Δ | pad_shared_memory... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43040 ns | 43070 ns | +30 ns (+0.1%) | 47230 ns | +4190 ns (+9.7%) | 39070 ns | -3970 ns (-9.2%) |
| SM Throughput (%) | 9.52 % | 9.50 % | -0.02 pp | 7.38 % | -2.14 pp | 9.42 % | -0.10 pp |
| Memory Throughput (%) | 63.49 % | 63.37 % | -0.12 pp | 52.79 % | -10.70 pp | 60.68 % | -2.81 pp |
| DRAM Throughput (%) | 0.88 % | 0.88 % | +0.00 pp | 0.80 % | -0.08 pp | 0.96 % | +0.08 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 96.00 register/thread | +2 | 64.00 register/thread | -30 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 | 0.09 | -0.15 |
| No Eligible (%) | 86.27 % | 86.34 % | +0.07 pp | 78.81 % | -7.46 pp | 86.32 % | +0.05 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.42 inst/cycle | -0.13 | 0.55 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.1053× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_kernel_multi_tile_per_sm | 1.094× | 1.097× | ✓ |  |
| B | reduce_block_k_to_eliminate_smem_pressure_and_improve_occupancy | 1.087× | 1.102× | ✓ |  |
| C | transpose_b_tile_load_with_dynamic_grid_to_eliminate_18way_bank_conflicts | 1.021× | 0.913× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | reduce_block_k_to... | Δ | transpose_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43040 ns | 39230 ns | -3810 ns (-8.9%) | 39070 ns | -3970 ns (-9.2%) | 47140 ns | +4100 ns (+9.5%) |
| SM Throughput (%) | 9.52 % | 9.33 % | -0.19 pp | 9.45 % | -0.07 pp | 7.38 % | -2.14 pp |
| Memory Throughput (%) | 63.49 % | 60.73 % | -2.76 pp | 60.84 % | -2.65 pp | 52.81 % | -10.68 pp |
| DRAM Throughput (%) | 0.88 % | 0.96 % | +0.08 pp | 0.96 % | +0.08 pp | 0.81 % | -0.07 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 94.00 register/thread | 70.00 register/thread | -24 | 64.00 register/thread | -30 | 96.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.10 | -0.14 | 0.09 | -0.15 | 0.12 | -0.12 |
| No Eligible (%) | 86.27 % | 86.45 % | +0.18 pp | 86.33 % | +0.06 pp | 78.82 % | -7.45 pp |
| IPC Active | 0.55 inst/cycle | 0.54 inst/cycle | -0.01 | 0.55 inst/cycle | +0.00 | 0.42 inst/cycle | -0.13 |


### Key Learnings

#### ✅ Plan A: `persistent_kernel_multi_tile_per_sm` (1.097×)

When a kernel launches fewer blocks than the GPU has SMs (e.g. 96 blocks on 132-SM H200, 0.09 waves/SM), padding the grid to NUM_SMS and wrapping the tile body in a `while tile_id < num_tiles` loop ensures all SMs receive a CTA. When num_tiles < NUM_SMS, the loop degenerates to an at-most-once guard: CTAs with cta_id >= num_tiles exit immediately, while the active CTAs process their single tile. The profiling showed only a modest ~1.10x speedup rather than the theoretical ~1.37x maximum, because the kernel is MIO-stall-dominated (CPIStall 76% of cycles, 12-way shared memory bank conflicts) and the idle-SM fraction (~27%) is not the primary bottleneck. Registers per thread increased slightly (64→70) due to the while-loop control variables, which reduced Block Limit Registers from 8 to 7 and theoretical occupancy from 50% to 43.75%, partially offsetting the grid-padding benefit.

> **Applicable when**: Grid size is less than SM count (waves_per_sm < 1.0, especially < 0.5), LaunchConfiguration NCU rule is triggered, and the kernel is not already severely register-limited (registers_per_thread should have headroom below the block limit). The while-loop adds a few control registers, so apply only when Block Limit Registers >= 8 before the change.

> **Avoid when**: When num_tiles is already >= NUM_SMS (waves_per_sm >= 1.0), the grid-padding adds no benefit and the while-loop overhead is pure cost. Also avoid when the kernel is dominated by structural shared memory bank conflicts (12-way, >75% excessive wavefronts) or MIO stalls (CPIStall >70% of cycles) — in those cases, filling idle SMs yields only marginal gains (~10%) because the bottleneck is per-warp stall latency, not SM utilization. Additionally, if registers_per_thread is already near the block limit (e.g. Block Limit Registers = 8 and adding the while-loop pushes to 7), theoretical occupancy will decrease, partially negating the benefit.

---

## Cumulative Progress

```
Baseline:  43040 ns  (1.000×)
Iter 2     38940 ns  (1.105×)  — reduce_register_pressure_via_smaller_tiles_with_autotune

Final best: 39070 ns  (1.1016×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36(arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -60,24 +61,24 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 672*idx_m
         a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
+        # Transposed B load: [BLOCK_N, BLOCK_K] with K as inner (contiguous) dimension
+        # This eliminates 12-way bank conflicts caused by stride_bn=672
+        b_mask_T = offs_k[None, :] < (K - k_idx * BLOCK_K)  # shape [1, BLOCK_K]
+        b_k_idx_vals_T = offs_k[None, :] + (k_idx * BLOCK_K)  # shape [1, BLOCK_K]
+        idx_k_T = b_k_idx_vals_T   # [1, BLOCK_K] K-index, inner, stride_bk=1 contiguous
+        idx_n_T = offs_b_n[:, None]  # [BLOCK_N, 1] N-index, outer
+        xindex_T = idx_k_T + 672 * idx_n_T  # [BLOCK_N, BLOCK_K]
+        b_T = tl.load(B + xindex_T, mask=b_mask_T, other=0.0)  # [BLOCK_N, BLOCK_K]
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +90,6 @@

     # inductor generates a suffix
     xindex = idx_n + 192*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
