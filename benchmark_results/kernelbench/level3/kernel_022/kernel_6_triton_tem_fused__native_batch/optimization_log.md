# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45860 ns — **Best latency**: 45220 ns (**1.0142×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tma_loads_via_tensor_descriptor`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_6_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45860 ns
- **SM Throughput**: 30.72 % | **Memory**: 77.55 % | **DRAM**: 13.15 %
- **Achieved Occupancy**: 17.91 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.996× | 1.011× | ✓ |  |
| B | tma_loads_via_block_ptr_and_reduce_register_pressure | 1.005× | 1.012× | ✓ | **✓ best** |
| C | eviction_policies_streaming_loads | 0.994× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_via_blo... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45860 ns | 45380 ns | -480 ns (-1.0%) | 45310 ns | -550 ns (-1.2%) | 45700 ns | -160 ns (-0.3%) |
| SM Throughput (%) | 30.72 % | 30.58 % | -0.14 pp | 30.11 % | -0.61 pp | 30.40 % | -0.32 pp |
| Memory Throughput (%) | 77.55 % | 77.17 % | -0.38 pp | 77.13 % | -0.42 pp | 76.65 % | -0.90 pp |
| DRAM Throughput (%) | 13.15 % | 13.31 % | +0.16 pp | 13.32 % | +0.17 pp | 13.17 % | +0.02 pp |
| Achieved Occupancy (%) | 17.91 % | 17.88 % | -0.03 pp | 17.92 % | +0.01 pp | 17.91 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.77 % | 66.67 % | -0.10 pp | 67.54 % | +0.77 pp | 66.74 % | -0.03 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.30 inst/cycle | -0.03 | 1.33 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0121× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.995× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45860 ns | 45630 ns | -230 ns (-0.5%) |
| SM Throughput (%) | 30.72 % | 30.27 % | -0.45 pp |
| Memory Throughput (%) | 77.55 % | 76.34 % | -1.21 pp |
| DRAM Throughput (%) | 13.15 % | 13.24 % | +0.09 pp |
| Achieved Occupancy (%) | 17.91 % | 17.95 % | +0.04 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 |
| No Eligible (%) | 66.77 % | 67.05 % | +0.28 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0121× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_tensor_descriptor | 1.008× | 1.014× | ✓ | **✓ best** |
| D | autotune_tile_sizes_with_dynamic_grid_to_increase_parallelism | 0.971× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_ten... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45860 ns | 45220 ns | -640 ns (-1.4%) | 45820 ns | -40 ns (-0.1%) |
| SM Throughput (%) | 30.72 % | 30.33 % | -0.39 pp | 29.97 % | -0.75 pp |
| Memory Throughput (%) | 77.55 % | 77.87 % | +0.32 pp | 76.23 % | -1.32 pp |
| DRAM Throughput (%) | 13.15 % | 13.36 % | +0.21 pp | 13.22 % | +0.07 pp |
| Achieved Occupancy (%) | 17.91 % | 17.92 % | +0.01 pp | 17.93 % | +0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.77 % | 67.37 % | +0.60 pp | 67.39 % | +0.62 pp |
| IPC Active | 1.33 inst/cycle | 1.31 inst/cycle | -0.02 | 1.30 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.0142× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_b_matrix_load_index_to_eliminate_shared_memory_bank_conflicts | 0.973× | 0.988× | ✓ |  |
| D | fix_b_matrix_load_layout_to_eliminate_shared_memory_bank_conflicts | 0.968× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45860 ns | 46400 ns | +540 ns (+1.2%) | 46140 ns | +280 ns (+0.6%) |
| SM Throughput (%) | 30.72 % | 29.91 % | -0.81 pp | 29.73 % | -0.99 pp |
| Memory Throughput (%) | 77.55 % | 76.06 % | -1.49 pp | 75.63 % | -1.92 pp |
| DRAM Throughput (%) | 13.15 % | 13.03 % | -0.12 pp | 13.32 % | +0.17 pp |
| Achieved Occupancy (%) | 17.91 % | 17.94 % | +0.03 pp | 17.91 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.77 % | 67.39 % | +0.62 pp | 67.42 % | +0.65 pp |
| IPC Active | 1.33 inst/cycle | 1.30 inst/cycle | -0.03 | 1.30 inst/cycle | -0.03 |

---

## Iteration 5

**Best so far**: 1.0142× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tl_trans_to_remove_shared_memory_bank_conflicts | 0.973× | 0.992× | ✓ |  |
| B | eliminate_b_transpose_to_fix_shared_memory_bank_conflicts | 0.968× | 0.990× | ✓ |  |
| C | rewrite_b_load_with_correct_masking_and_reduce_block_n_to_fix_bank_conflicts | 0.960× | 0.982× | ✓ |  |
| D | add_triton_autotune | 0.971× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | eliminate_b_trans... | Δ | rewrite_b_load_wi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45860 ns | 46210 ns | +350 ns (+0.8%) | 46340 ns | +480 ns (+1.0%) | 46720 ns | +860 ns (+1.9%) | 46050 ns | +190 ns (+0.4%) |
| SM Throughput (%) | 30.72 % | 29.87 % | -0.85 pp | 30.00 % | -0.72 pp | 29.28 % | -1.44 pp | 29.61 % | -1.11 pp |
| Memory Throughput (%) | 77.55 % | 76.00 % | -1.55 pp | 76.29 % | -1.26 pp | 74.52 % | -3.03 pp | 75.31 % | -2.24 pp |
| DRAM Throughput (%) | 13.15 % | 13.01 % | -0.14 pp | 13.05 % | -0.10 pp | 12.94 % | -0.21 pp | 13.13 % | -0.02 pp |
| Achieved Occupancy (%) | 17.91 % | 17.95 % | +0.04 pp | 17.93 % | +0.02 pp | 17.98 % | +0.07 pp | 17.99 % | +0.08 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.77 % | 67.50 % | +0.73 pp | 67.46 % | +0.69 pp | 68.08 % | +1.31 pp | 67.37 % | +0.60 pp |
| IPC Active | 1.33 inst/cycle | 1.30 inst/cycle | -0.03 | 1.30 inst/cycle | -0.03 | 1.28 inst/cycle | -0.05 | 1.30 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  45860 ns  (1.000×)
Iter 1     45310 ns  (1.012×)  — tma_loads_via_block_ptr_and_reduce_register_pressure
Iter 3     45220 ns  (1.014×)  — tma_loads_via_tensor_descriptor

Final best: 45220 ns  (1.0142×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 128
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -60,21 +61,22 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 16*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        # Load B as [BLOCK_N, BLOCK_K]: consecutive threads vary along N (stride-1 in N), no bank conflicts
+        # B has stride_bk=1, stride_bn=16, so B[k, n] is at offset k*1 + n*16 = k + 16*n
+        # b_T[n, k] = B[k, n] = B + k + 16*n
+        idx_n_col = offs_b_n[:, None]                      # shape [BLOCK_N, 1]
+        idx_k_row = offs_k[None, :] + (k_idx * BLOCK_K)   # shape [1, BLOCK_K]
+        # offset = k + 16*n (stride_bk=1, stride_bn=16)
+        xindex_b = idx_k_row + 16 * idx_n_col              # shape [BLOCK_N, BLOCK_K], stride-1 in N
+        b_T = tl.load(B + xindex_b, eviction_policy='evict_first')  # [BLOCK_N, BLOCK_K]
+        acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        # tl.trans(b_T) produces [BLOCK_K, BLOCK_N] matching tl.dot(a=[BLOCK_M,BLOCK_K], b=[BLOCK_K,BLOCK_N])
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -85,9 +87,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
