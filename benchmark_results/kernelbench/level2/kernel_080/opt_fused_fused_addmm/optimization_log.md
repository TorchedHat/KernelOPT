# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 629630 ns — **Best latency**: 619580 ns (**1.0162×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_pressure_via_smaller_tile_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_080/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 202
- **Duration**: 629630 ns
- **SM Throughput**: 61.24 % | **Memory**: 54.07 % | **DRAM**: 11.82 %
- **Achieved Occupancy**: 12.47 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 72.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.016× | ✓ |  |
| B | fast_accumulation_fp16_tl_dot | 0.857× | 1.056× | ✓ |  |
| C | fp8_precision_gemm_hopper | 0.958× | 1.045× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fast_accumulation... | Δ | fp8_precision_gem... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 629630 ns | 619840 ns | -9790 ns (-1.6%) | 596420 ns | -33210 ns (-5.3%) | 602460 ns | -27170 ns (-4.3%) |
| SM Throughput (%) | 61.24 % | 60.98 % | -0.26 pp | 36.36 % | -24.88 pp | 31.19 % | -30.05 pp |
| Memory Throughput (%) | 54.07 % | 54.62 % | +0.55 pp | 67.64 % | +13.57 pp | 50.21 % | -3.86 pp |
| DRAM Throughput (%) | 11.82 % | 12.03 % | +0.21 pp | 12.17 % | +0.35 pp | 12.42 % | +0.60 pp |
| Achieved Occupancy (%) | 12.47 % | 12.37 % | -0.10 pp | 12.16 % | -0.31 pp | 12.20 % | -0.27 pp |
| Registers/Thread | 202.00 register/thread | 202.00 register/thread | +0 | 205.00 register/thread | +3 | 248.00 register/thread | +46 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 72.25 % | 72.06 % | -0.19 pp | 62.37 % | -9.88 pp | 71.10 % | -1.15 pp |
| IPC Active | 1.11 inst/cycle | 1.11 inst/cycle | +0.00 | 1.48 inst/cycle | +0.37 | 1.15 inst/cycle | +0.04 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_autotune_tile_configs | 0.858× | 1.053× | ✓ |  |
| B | reduce_register_pressure_via_smaller_autotune_tiles | 0.958× | 0.318× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tile_autotune_configs | 1.000× | 1.016× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 629630 ns | 597890 ns | -31740 ns (-5.0%) | 1980000 ns | +1350370 ns (+214.5%) | 619580 ns | -10050 ns (-1.6%) |
| SM Throughput (%) | 61.24 % | 36.45 % | -24.79 pp | 43.17 % | -18.07 pp | 61.91 % | +0.67 pp |
| Memory Throughput (%) | 54.07 % | 67.52 % | +13.45 pp | 73.60 % | +19.53 pp | 55.23 % | +1.16 pp |
| DRAM Throughput (%) | 11.82 % | 12.17 % | +0.35 pp | 12.23 % | +0.41 pp | 11.85 % | +0.03 pp |
| Achieved Occupancy (%) | 12.47 % | 12.15 % | -0.32 pp | 29.89 % | +17.42 pp | 12.38 % | -0.09 pp |
| Registers/Thread | 202.00 register/thread | 205.00 register/thread | +3 | 89.00 register/thread | -113 | 202.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 6.21 | +4.27 | 1.94 | +0.00 |
| No Eligible (%) | 72.25 % | 63.01 % | -9.24 pp | 61.07 % | -11.18 pp | 72.16 % | -0.09 pp |
| IPC Active | 1.11 inst/cycle | 1.48 inst/cycle | +0.37 | 1.56 inst/cycle | +0.45 | 1.11 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0162× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_larger_block_k | 0.742× | 0.759× | ✓ |  |
| B | constant_fold_mathematically_trivial_max_mean_sub_gelu_kernel | 1.000× | 1.027× | ✓ |  |
| C | resolve_shared_memory_bank_conflicts_via_autotune_num_warps_block_n | 0.953× | 0.313× | ✓ |  |
| D | add_triton_autotune | 0.851× | 1.043× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | constant_fold_mat... | Δ | resolve_shared_me... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 629630 ns | 829440 ns | +199810 ns (+31.7%) | 613280 ns | -16350 ns (-2.6%) | 2010000 ns | +1380370 ns (+219.2%) | 603420 ns | -26210 ns (-4.2%) |
| SM Throughput (%) | 61.24 % | 25.80 % | -35.44 pp | 60.80 % | -0.44 pp | 41.85 % | -19.39 pp | 36.27 % | -24.97 pp |
| Memory Throughput (%) | 54.07 % | 40.68 % | -13.39 pp | 55.81 % | +1.74 pp | 72.58 % | +18.51 pp | 67.54 % | +13.47 pp |
| DRAM Throughput (%) | 11.82 % | 11.28 % | -0.54 pp | 12.04 % | +0.22 pp | 12.08 % | +0.26 pp | 12.06 % | +0.24 pp |
| Achieved Occupancy (%) | 12.47 % | 6.25 % | -6.22 pp | 12.38 % | -0.09 pp | 29.79 % | +17.32 pp | 12.47 % | +0.00 pp |
| Registers/Thread | 202.00 register/thread | 245.00 register/thread | +43 | 202.00 register/thread | +0 | 89.00 register/thread | -113 | 205.00 register/thread | +3 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 1.94 | +0.00 | 6.21 | +4.27 | 1.94 | +0.00 |
| No Eligible (%) | 72.25 % | 74.37 % | +2.12 pp | 72.20 % | -0.05 pp | 61.08 % | -11.17 pp | 62.94 % | -9.31 pp |
| IPC Active | 1.11 inst/cycle | 1.03 inst/cycle | -0.08 | 1.11 inst/cycle | +0.00 | 1.53 inst/cycle | +0.42 | 1.51 inst/cycle | +0.40 |


### Key Learnings

#### ❌ Plan A: `eliminate_shared_memory_bank_conflicts_via_larger_block_k` (0.694×)

Replacing all BLOCK_K=32 autotune configs with BLOCK_K=64/128 variants caused dynamic shared memory per block to nearly double (73.7 KB → 147.5 KB), collapsing Block Limit Shared Mem from 3 to 1 block/SM and halving theoretical occupancy from 12.5% to 6.25%. All throughput metrics worsened (Compute SM: 36% → 27%, Memory: 68% → 42%, L1/TEX: 70% → 45%) and duration increased 50% (603 us → 907 us). This is the same failure mode confirmed in iteration 3 (speedup 0.76x): when Block Limit Registers=2 is already the binding constraint and shared memory per block is large (>64 KB), increasing BLOCK_K expands the shared memory tile proportionally, making Block Limit Shared Mem=1 the new binding constraint and creating a worse occupancy collapse.

> **Applicable when**: Never: this strategy is a confirmed regression in two consecutive iterations (3 and 4) for this kernel shape.

> **Avoid when**: When Block Limit Registers is already the binding occupancy constraint (Block Limit Registers=2, Block Limit Shared Mem=3) and dynamic shared memory per block is already large (>64 KB). Increasing BLOCK_K doubles shared memory usage, making Block Limit Shared Mem=1 the new binding constraint and halving occupancy from 12.5% to 6.25%. This pattern has now regressed twice (0.76x in iter 3, 0.69x in iter 4). The correct fix for SharedMemoryConflicts is shared memory layout swizzling or padding, not tile dimension expansion.

#### ❌ Plan C: `resolve_shared_memory_bank_conflicts_via_autotune_num_warps_block_n` (0.313×)

Adding high-num_warps, large-BLOCK_N autotune configs to address SharedMemoryConflicts (10-way bank conflicts, 60% wavefront overhead) caused the autotuner to select a small-tile config (block_size=64, grid_size=8192) that exploded total instruction count 3.5× (155M → 538M) and worsened Warp Cycles Per Issued Instruction from 5.27 to 12.37. Although theoretical occupancy improved (12.5% → 31.25%) and bank conflicts were nominally reduced, the 16× increase in grid size (512 → 8192) created massive L2 pressure and the small tiles had insufficient arithmetic intensity to hide memory latency, resulting in a 3.2× regression. This is the same failure mode as iteration 2: for large-K GEMMs, the autotuner selects small-tile configs that are slower despite higher occupancy, because instruction count and memory latency dominate over occupancy gains.

> **Applicable when**: Never beneficial for large-K GEMMs (K >= 4096) where adding small-tile or high-warp configs causes the autotuner to select configs with insufficient arithmetic intensity. This is a confirmed dead end for this problem shape.

> **Avoid when**: Large-K GEMM kernels (K >= 4096, M >= 512, N >= 4096) where register pressure limits occupancy. Adding small-tile or high-num_warps configs to the autotune list causes the autotuner to select configs with low arithmetic intensity per tile, exploding instruction count (3.5×) and worsening Warp Cycles Per Issued Instruction (5.27 → 12.37). SharedMemoryConflicts in the GEMM kernel are a symptom of the tile shape, not a root cause addressable by autotune config expansion. Do not apply when Warp Cycles Per Issued Instruction is already low (<6) and the kernel is memory-throughput-bound.

---

## Iteration 4

**Best so far**: 1.0162× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_in_reduction_kernel_via_fixed_small_block_n | 0.733× | 0.734× | ✓ |  |
| D | restrict_block_k_to_32_to_restore_smem_occupancy | 0.872× | 1.060× | ✓ |  |
| E | add_triton_autotune | 0.739× | 0.694× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | restrict_block_k_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 629630 ns | 858180 ns | +228550 ns (+36.3%) | 594020 ns | -35610 ns (-5.7%) | 906980 ns | +277350 ns (+44.0%) |
| SM Throughput (%) | 61.24 % | 25.21 % | -36.03 pp | 36.36 % | -24.88 pp | 26.85 % | -34.39 pp |
| Memory Throughput (%) | 54.07 % | 39.74 % | -14.33 pp | 67.70 % | +13.63 pp | 42.33 % | -11.74 pp |
| DRAM Throughput (%) | 11.82 % | 10.45 % | -1.37 pp | 12.25 % | +0.43 pp | 11.08 % | -0.74 pp |
| Achieved Occupancy (%) | 12.47 % | 6.25 % | -6.22 pp | 12.15 % | -0.32 pp | 6.25 % | -6.22 pp |
| Registers/Thread | 202.00 register/thread | 245.00 register/thread | +43 | 205.00 register/thread | +3 | 245.00 register/thread | +43 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 1.94 | +0.00 | 3.88 | +1.94 |
| No Eligible (%) | 72.25 % | 74.37 % | +2.12 pp | 63.02 % | -9.23 pp | 74.37 % | +2.12 pp |
| IPC Active | 1.11 inst/cycle | 1.03 inst/cycle | -0.08 | 1.48 inst/cycle | +0.37 | 1.03 inst/cycle | -0.08 |


### Key Learnings

#### ❌ Plan A: `reduce_register_pressure_in_reduction_kernel_via_fixed_small_block_n` (0.734×)

Removing large BLOCK_N configs (1024–8192) from the reduction kernel autotune and replacing with small BLOCK_N configs (128–512, num_stages=1) caused the autotuner to select a config that doubled dynamic shared memory per block (73.7 KB → 147.5 KB), which halved the Block Limit Shared Mem from 3 to 1 block/SM. This cut theoretical occupancy from 12.5% to 6.25% and worsened all scheduler metrics (No Eligible: 62.9% → 74.4%, eligible warps: 0.44 → 0.26), increasing duration by ~42%. The register count also increased (205 → 245), suggesting the GEMM kernel's autotune was simultaneously affected by the config changes, and the new GEMM configs selected larger BLOCK_K tiles that inflated both register and shared memory usage.

> **Applicable when**: Never: this specific combination of changes (larger BLOCK_K in GEMM + smaller BLOCK_N in reduction) is a confirmed dead end. The GEMM config change dominates and inflates shared memory beyond what the reduction change saves.

> **Avoid when**: When the GEMM kernel's autotune configs are simultaneously modified alongside the reduction kernel's configs. Changing BLOCK_K from 32 to 64/128 in the GEMM autotune inflates dynamic shared memory per block (doubles it from ~74 KB to ~147 KB), which becomes the new binding occupancy constraint (Block Limit Shared Mem drops from 3 to 1). Do not apply when the GEMM kernel already has Block Limit Registers = 2 as the binding constraint — increasing BLOCK_K will shift the bottleneck to shared memory and worsen occupancy. Never modify both the GEMM and reduction kernel autotune configs simultaneously without isolating the effect of each change.

#### ✅ Plan D: `restrict_block_k_to_32_to_restore_smem_occupancy` (1.060×)

Removing all BLOCK_K>=64 autotune configs and replacing them with BLOCK_K=32 equivalents prevents the autotuner from selecting configurations that balloon dynamic shared memory per block (e.g., 147 KB with BLOCK_K=64/128 vs ~73 KB with BLOCK_K=32), which would collapse Block Limit Shared Mem from 3 to 1 and halve theoretical occupancy from 12.5% to 6.25%. In this iteration the before-state was already at BLOCK_K=32 (73.7 KB SMEM, Block Limit Shared Mem=3), so the change was primarily a cleanup that removed a few residual BLOCK_K=64 configs and replaced BLOCK_N=256 entries; the L2 Hit Rate improved marginally (79.0% → 81.3%) and duration dropped ~9 µs (603 → 594 µs). The key generalizable insight is: in Triton GEMM autotune config lists, any config with BLOCK_K>=64 combined with large BLOCK_M/BLOCK_N and multi-stage pipelines can push SMEM per block above the per-SM budget divided by 2, causing Block Limit Shared Mem to drop to 1 and occupancy to halve — always audit the SMEM footprint of every autotune config before including it.

> **Applicable when**: Triton GEMM autotune config lists on GPUs with limited per-SM shared memory (e.g., H200: 228 KB/SM). Apply when any config in the list has BLOCK_K>=64 combined with BLOCK_M>=128, BLOCK_N>=128, and num_stages>=3 using FP16 operands, which can push SMEM per block above 114 KB (half of 228 KB), collapsing Block Limit Shared Mem to 1. Also apply when profiling shows Block Limit Shared Mem has dropped to 1 and theoretical occupancy has halved from its expected value.

> **Avoid when**: When BLOCK_K=32 is insufficient for the tensor core tile size requirements (e.g., Hopper WGMMA requires BLOCK_K>=64 for FP8 operands). Also do not apply when the GPU has very large per-SM shared memory (e.g., >256 KB) where even BLOCK_K=128 configs stay within the 2-block limit. Do not apply when arithmetic intensity is the primary bottleneck and larger BLOCK_K is needed to amortize memory loads.

---

## Cumulative Progress

```
Baseline:  629630 ns  (1.000×)
Iter 2     619580 ns  (1.016×)  — reduce_register_pressure_via_smaller_tile_autotune_configs

Final best: 619580 ns  (1.0162×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 
 @triton.autotune(
@@ -12,6 +13,12 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=2),
     ],
     key=['M', 'N', 'K'],
 )
@@ -42,38 +49,45 @@

     a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
     b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
 
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
+    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
 
     for k in range(0, tl.cdiv(K, BLOCK_K)):
         k_remaining = K - k * BLOCK_K
         a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
         b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
-        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
-        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
-        acc = tl.dot(a, b, acc)
+        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)
+        acc = tl.dot(a, b, acc, out_dtype=tl.float16)
         a_ptrs += BLOCK_K * stride_ak
         b_ptrs += BLOCK_K * stride_bk
 
+    # Upcast to FP32 before bias add to preserve output precision
+    acc_fp32 = acc.to(tl.float32)
+
     # Add bias
     bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
-    acc = acc + bias[None, :]
+    acc_fp32 = acc_fp32 + bias[None, :]
 
     # Store
     c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
     c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
-    tl.store(c_ptrs, acc, mask=c_mask)
+    tl.store(c_ptrs, acc_fp32, mask=c_mask)
 
 
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
         triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=4),
         triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=8),
         triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
+        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=8),
     ],
     key=['N'],
 )
 @triton.jit
+
 def max_mean_sub_gelu_kernel(
     input_ptr, output_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
