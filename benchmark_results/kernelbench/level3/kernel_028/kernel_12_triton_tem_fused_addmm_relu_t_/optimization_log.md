# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_view_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 208640 ns — **Best latency**: 195420 ns (**1.0676×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`promote_block_m_block_n_block_k_to_autotune_with_correct_dynamic_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_view_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_12_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (112,1,1), Waves/SM: 0.42
- **Registers/thread**: 128
- **Duration**: 208640 ns
- **SM Throughput**: 10.64 % | **Memory**: 80.41 % | **DRAM**: 0.73 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 87.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.001× | ✓ | **✓ best** |
| B | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 0.818× | 0.819× | ✓ |  |
| C | autotune_tile_sizes_to_eliminate_smem_bank_conflicts_and_improve_occupancy | 0.942× | 0.943× | ✓ |  |
| D | increase_block_n_to_eliminate_shared_memory_bank_conflicts | 0.877× | 0.840× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 208640 ns | 208380 ns | -260 ns (-0.1%) | 254780 ns | +46140 ns (+22.1%) | 221340 ns | +12700 ns (+6.1%) | 248380 ns | +39740 ns (+19.0%) |
| SM Throughput (%) | 10.64 % | 10.64 % | +0.00 pp | 16.64 % | +6.00 pp | 19.30 % | +8.66 pp | 16.98 % | +6.34 pp |
| Memory Throughput (%) | 80.41 % | 80.37 % | -0.04 pp | 65.95 % | -14.46 pp | 75.97 % | -4.44 pp | 67.59 % | -12.82 pp |
| DRAM Throughput (%) | 0.73 % | 0.73 % | +0.00 pp | 0.60 % | -0.13 pp | 0.68 % | -0.05 pp | 0.62 % | -0.11 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 208.00 register/thread | +80 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.28 | -0.14 | 0.21 | -0.21 | 0.85 | +0.43 |
| No Eligible (%) | 87.30 % | 87.30 % | +0.00 pp | 79.59 % | -7.71 pp | 76.90 % | -10.40 pp | 78.74 % | -8.56 pp |
| IPC Active | 0.51 inst/cycle | 0.51 inst/cycle | +0.00 | 0.82 inst/cycle | +0.31 | 0.92 inst/cycle | +0.41 | 0.85 inst/cycle | +0.34 |

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_cut_smem_footprint_and_raise_occupancy_via_proper_autotune | 1.010× | 1.012× | ✓ | **✓ best** |
| C | reduce_tile_sizes_to_recover_occupancy_via_proper_autotune_promotion | 0.940× | 0.940× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.818× | 0.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | reduce_tile_sizes... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 208640 ns | 206140 ns | -2500 ns (-1.2%) | 221920 ns | +13280 ns (+6.4%) | 254430 ns | +45790 ns (+21.9%) |
| SM Throughput (%) | 10.64 % | 10.84 % | +0.20 pp | 19.29 % | +8.65 pp | 16.66 % | +6.02 pp |
| Memory Throughput (%) | 80.41 % | 81.31 % | +0.90 pp | 75.91 % | -4.50 pp | 66.04 % | -14.37 pp |
| DRAM Throughput (%) | 0.73 % | 0.74 % | +0.01 pp | 0.68 % | -0.05 pp | 0.60 % | -0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.21 | -0.21 | 0.21 | -0.21 | 0.28 | -0.14 |
| No Eligible (%) | 87.30 % | 87.06 % | -0.24 pp | 76.87 % | -10.43 pp | 79.59 % | -7.71 pp |
| IPC Active | 0.51 inst/cycle | 0.52 inst/cycle | +0.01 | 0.92 inst/cycle | +0.41 | 0.82 inst/cycle | +0.31 |

---

## Iteration 3

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_transposition_to_eliminate_smem_bank_conflicts | 1.010× | 1.013× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_block_ptr_swizzle_padding | 0.885× | 0.887× | ✓ |  |
| C | smem_padding_to_eliminate_bank_conflicts_on_transposed_b_load | 0.819× | 0.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | eliminate_smem_ba... | Δ | smem_padding_to_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 208640 ns | 206050 ns | -2590 ns (-1.2%) | 235330 ns | +26690 ns (+12.8%) | 254370 ns | +45730 ns (+21.9%) |
| SM Throughput (%) | 10.64 % | 10.84 % | +0.20 pp | 18.48 % | +7.84 pp | 16.65 % | +6.01 pp |
| Memory Throughput (%) | 80.41 % | 81.33 % | +0.92 pp | 72.39 % | -8.02 pp | 65.98 % | -14.43 pp |
| DRAM Throughput (%) | 0.73 % | 0.74 % | +0.01 pp | 0.64 % | -0.09 pp | 0.60 % | -0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.21 | -0.21 | 0.21 | -0.21 | 0.28 | -0.14 |
| No Eligible (%) | 87.30 % | 87.06 % | -0.24 pp | 77.98 % | -9.32 pp | 79.61 % | -7.69 pp |
| IPC Active | 0.51 inst/cycle | 0.52 inst/cycle | +0.01 | 0.88 inst/cycle | +0.37 | 0.82 inst/cycle | +0.31 |

---

## Iteration 4

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tl_trans_smem_staging_via_native_trans_b_dot | 1.012× | 1.012× | ✓ |  |
| C | reduce_block_m_block_k_to_shrink_smem_and_registers_for_occupancy_recovery | 1.477× | 1.489× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | reduce_block_m_bl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 208640 ns | 206180 ns | -2460 ns (-1.2%) | 140130 ns | -68510 ns (-32.8%) |
| SM Throughput (%) | 10.64 % | 10.85 % | +0.21 pp | 15.43 % | +4.79 pp |
| Memory Throughput (%) | 80.41 % | 81.40 % | +0.99 pp | 60.01 % | -20.40 pp |
| DRAM Throughput (%) | 0.73 % | 0.74 % | +0.01 pp | 1.08 % | +0.35 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.21 | -0.21 | 0.11 | -0.31 |
| No Eligible (%) | 87.30 % | 87.06 % | -0.24 pp | 62.93 % | -24.37 pp |
| IPC Active | 0.51 inst/cycle | 0.52 inst/cycle | +0.01 | 0.74 inst/cycle | +0.23 |

---

## Iteration 5

**Best so far**: 1.4889× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tl_trans_by_direct_b_load_in_k_n_layout | 1.465× | 1.489× | ✓ |  |
| B | tma_loads_via_block_ptr_to_eliminate_mio_stalls_and_address_overhead | 1.373× | 1.392× | ✓ |  |
| C | promote_block_m_block_n_block_k_to_autotune_with_correct_dynamic_grid | 1.557× | 1.068× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.480× | 1.490× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | tma_loads_via_blo... | Δ | promote_block_m_b... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 208640 ns | 140160 ns | -68480 ns (-32.8%) | 149890 ns | -58750 ns (-28.2%) | 195420 ns | -13220 ns (-6.3%) | 140060 ns | -68580 ns (-32.9%) |
| SM Throughput (%) | 10.64 % | 15.44 % | +4.80 pp | 14.82 % | +4.18 pp | 11.60 % | +0.96 pp | 15.45 % | +4.81 pp |
| Memory Throughput (%) | 80.41 % | 60.07 % | -20.34 pp | 56.42 % | -23.99 pp | 81.14 % | +0.73 pp | 60.10 % | -20.31 pp |
| DRAM Throughput (%) | 0.73 % | 1.08 % | +0.35 pp | 1.01 % | +0.28 pp | 0.77 % | +0.04 pp | 1.08 % | +0.35 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 94.00 register/thread | -34 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.11 | -0.31 | 0.11 | -0.31 | 0.17 | -0.25 | 0.11 | -0.31 |
| No Eligible (%) | 87.30 % | 62.91 % | -24.39 pp | 63.84 % | -23.46 pp | 86.15 % | -1.15 pp | 62.92 % | -24.38 pp |
| IPC Active | 0.51 inst/cycle | 0.74 inst/cycle | +0.23 | 0.72 inst/cycle | +0.21 | 0.55 inst/cycle | +0.04 | 0.74 inst/cycle | +0.23 |


### Key Learnings

#### ❌ Plan A: `eliminate_tl_trans_by_direct_b_load_in_k_n_layout` (1.489×)

Replacing a transposed B load ([BLOCK_N, BLOCK_K] followed by tl.trans()) with a direct [BLOCK_K, BLOCK_N] load produced zero measurable change: duration, bank conflict way (12), conflict wavefront % (77.75%), warp cycles per instruction (2.7), executed instructions, and all occupancy metrics were identical before and after. The Triton compiler appears to generate equivalent SMEM staging regardless of whether tl.trans() is called explicitly or the tile is loaded in the target layout directly — the bank conflict pattern is determined by the underlying memory access strides, not the Python-level transpose call. When SharedMemoryConflicts persist at 12-way with 77.75% conflict wavefronts after eliminating tl.trans(), the root cause is the strided global memory access pattern (stride_bn=2048) forcing non-contiguous SMEM writes, not the transpose operation itself.

> **Applicable when**: Never: this direction produces no measurable improvement when SharedMemoryConflicts are driven by strided global memory strides (stride_bn >> 1) rather than by the transpose operation itself.

> **Avoid when**: When SharedMemoryConflicts (12-way bank conflicts, 77.75% conflict wavefront %) persist despite the tl.trans() removal — this indicates the conflict is caused by the underlying B matrix stride pattern (stride_bn=2048), not the transpose staging. Do not apply when all profiling metrics (duration, bank conflict way, conflict wavefront %, warp cycles/inst, executed instructions) are identical before and after the change.

---

## Cumulative Progress

```
Baseline:  208640 ns  (1.000×)
Iter 1     208380 ns  (1.001×)  — tf32_tensor_cores
Iter 2     206140 ns  (1.012×)  — reduce_block_k_to_cut_smem_footprint_and_raise_occupancy_via_proper_autotune
Iter 4     140130 ns  (1.489×)  — reduce_block_m_block_k_to_shrink_smem_and_registers_for_occupancy_recovery

Final best: 195420 ns  (1.0676×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_relu_t_view_12(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 32
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -60,18 +76,20 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 2048*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 512*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Load B directly as [BLOCK_K, BLOCK_N] without tl.trans() to eliminate SMEM bank conflicts.
+        # B[k, n] = B_ptr + k * stride_bk + n * stride_bn = B_ptr + k*1 + n*2048
+        # Loading as [BLOCK_K, BLOCK_N]: rows=k (fast dim stride=1), cols=n (stride=2048)
+        b_k_base = k_idx * BLOCK_K
+        offs_bk = offs_k[:, None] + b_k_base  # shape [BLOCK_K, 1]
+        offs_bn = offs_b_n[None, :]            # shape [1, BLOCK_N]
+        # address: k*1 + n*2048 -> [BLOCK_K, BLOCK_N], no tl.trans() needed
+        b = tl.load(B + offs_bk + 2048 * offs_bn)  # shape [BLOCK_K, BLOCK_N]
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -86,8 +104,6 @@

     # inductor generates a suffix
     xindex = idx_n + 512*idx_m
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
