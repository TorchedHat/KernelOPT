# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_56` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 58720 ns — **Best latency**: 58430 ns (**1.0050×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_shared_memory_bank_conflicts_via_coalesced_load_and_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_56`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_56_triton_tem_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 58720 ns
- **SM Throughput**: 15.41 % | **Memory**: 59.89 % | **DRAM**: 1.50 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 1.006× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.999× | 0.802× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 58720 ns | 58400 ns | -320 ns (-0.5%) | 73180 ns | +14460 ns (+24.6%) |
| SM Throughput (%) | 15.41 % | 15.47 % | +0.06 pp | 11.96 % | -3.45 pp |
| Memory Throughput (%) | 59.89 % | 60.13 % | +0.24 pp | 26.93 % | -32.96 pp |
| DRAM Throughput (%) | 1.50 % | 1.51 % | +0.01 pp | 1.21 % | -0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 254.00 register/thread | +126 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 76.94 % | -0.11 pp | 63.85 % | -13.20 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.72 inst/cycle | -0.20 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_shared_memory_bank_conflicts_via_coalesced_load_and_transpose | 1.000× | 1.005× | ✓ | **✓ best** |
| B | tma_loads | 0.999× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_shar... | Δ | tma_loads | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 58720 ns | 58430 ns | -290 ns (-0.5%) | 58140 ns | -580 ns (-1.0%) |
| SM Throughput (%) | 15.41 % | 15.43 % | +0.02 pp | 15.97 % | +0.56 pp |
| Memory Throughput (%) | 59.89 % | 59.97 % | +0.08 pp | 60.78 % | +0.89 pp |
| DRAM Throughput (%) | 1.50 % | 1.51 % | +0.01 pp | 1.52 % | +0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 76.97 % | -0.08 pp | 76.41 % | -0.64 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.94 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.005× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_transposed_smem_load | 0.915× | 0.924× | ✓ |  |
| C | eliminate_tl_trans_to_remove_shared_memory_bank_conflicts | 0.996× | 1.000× | ✓ |  |
| D | eliminate_b_matrix_transpose_load_direct_k_n_layout | 0.995× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | eliminate_tl_tran... | Δ | eliminate_b_matri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58720 ns | 63550 ns | +4830 ns (+8.2%) | 58720 ns | +0 ns (+0.0%) | 58940 ns | +220 ns (+0.4%) |
| SM Throughput (%) | 15.41 % | 14.44 % | -0.97 pp | 15.51 % | +0.10 pp | 15.53 % | +0.12 pp |
| Memory Throughput (%) | 59.89 % | 55.63 % | -4.26 pp | 60.27 % | +0.38 pp | 60.36 % | +0.47 pp |
| DRAM Throughput (%) | 1.50 % | 1.39 % | -0.11 pp | 1.50 % | +0.00 pp | 1.50 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 154.00 register/thread | +26 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.23 | +0.05 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 78.71 % | +1.66 pp | 76.94 % | -0.11 pp | 76.95 % | -0.10 pp |
| IPC Active | 0.92 inst/cycle | 0.85 inst/cycle | -0.07 | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.005× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_bank_conflicts_via_block_ptr_and_correct_autotune_grid | 0.996× | 0.824× | ✓ |  |
| B | autotune_with_dynamic_grid_to_fix_occupancy_and_sm_utilization | 0.915× | 0.797× | ✓ |  |
| C | autotune_with_dynamic_grid_and_direct_b_load_no_transpose | 0.911× | 0.854× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_bank_conflict... | Δ | autotune_with_dyn... | Δ | autotune_with_dyn... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58720 ns | 71230 ns | +12510 ns (+21.3%) | 73700 ns | +14980 ns (+25.5%) | 68800 ns | +10080 ns (+17.2%) |
| SM Throughput (%) | 15.41 % | 12.85 % | -2.56 pp | 11.80 % | -3.61 pp | 13.19 % | -2.22 pp |
| Memory Throughput (%) | 59.89 % | 49.44 % | -10.45 pp | 26.65 % | -33.24 pp | 51.39 % | -8.50 pp |
| DRAM Throughput (%) | 1.50 % | 1.24 % | -0.26 pp | 1.20 % | -0.30 pp | 1.28 % | -0.22 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 254.00 register/thread | +126 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 81.03 % | +3.98 pp | 63.40 % | -13.65 pp | 80.45 % | +3.40 pp |
| IPC Active | 0.92 inst/cycle | 0.76 inst/cycle | -0.16 | 0.73 inst/cycle | -0.19 | 0.78 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  58720 ns  (1.000×)
Iter 2     58430 ns  (1.005×)  — fix_b_matrix_shared_memory_bank_conflicts_via_coalesced_load_and_transpose

Final best: 58430 ns  (1.0050×)
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
 def triton_tem_fused_convolution_56(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -60,18 +61,20 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 512*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 144*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Load B directly as [BLOCK_K, BLOCK_N] to avoid tl.trans() shared memory staging
+        # B[k, n] is at offset k + 512*n (stride_bk=1, stride_bn=512)
+        # b_xindex[k, n] = (offs_k[k] + k_idx*BLOCK_K) + 512*offs_b_n[n]
+        b_k_base = k_idx * BLOCK_K
+        b_k_idx = offs_k[:, None] + b_k_base                # [BLOCK_K, 1]
+        b_n_idx = offs_b_n[None, :]                          # [1, BLOCK_N]
+        b_xindex = b_k_idx + 512 * b_n_idx                  # [BLOCK_K, BLOCK_N]
+        b = tl.load(B + b_xindex)                           # [BLOCK_K, BLOCK_N] -- no tl.trans needed!
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -88,8 +91,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
