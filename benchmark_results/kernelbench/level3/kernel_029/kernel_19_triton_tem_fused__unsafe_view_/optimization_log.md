# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 146980 ns — **Best latency**: 143460 ns (**1.0245×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tma_loads_for_gemm_bank_conflict_elimination`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_19_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (738,1,1), Waves/SM: 1.86
- **Registers/thread**: 168
- **Duration**: 146980 ns
- **SM Throughput**: 36.35 % | **Memory**: 81.80 % | **DRAM**: 1.09 %
- **Achieved Occupancy**: 17.55 %
- **IPC Active**: 1.59 inst/cycle | **No Eligible**: 60.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_streaming_gemm_loads | 1.000× | 1.003× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_reduce_bank_conflicts_and_improve_occupancy | 0.995× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 146980 ns | 146530 ns | -450 ns (-0.3%) | 146590 ns | -390 ns (-0.3%) |
| SM Throughput (%) | 36.35 % | 36.51 % | +0.16 pp | 36.45 % | +0.10 pp |
| Memory Throughput (%) | 81.80 % | 82.10 % | +0.30 pp | 82.02 % | +0.22 pp |
| DRAM Throughput (%) | 1.09 % | 1.09 % | +0.00 pp | 1.11 % | +0.02 pp |
| Achieved Occupancy (%) | 17.55 % | 17.52 % | -0.03 pp | 17.53 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 60.22 % | 60.31 % | +0.09 pp | 60.23 % | +0.01 pp |
| IPC Active | 1.59 inst/cycle | 1.59 inst/cycle | +0.00 | 1.59 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0031× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_grid_aligned_tiles_to_eliminate_partial_wave_and_reduce_register_pressure | 0.991× | 1.006× | ✓ |  |
| B | tma_loads_to_eliminate_shared_memory_bank_conflicts | 1.022× | 0.616× | ✓ |  |
| C | fix_b_matrix_load_layout_to_eliminate_bank_conflicts | 0.997× | 1.005× | ✓ |  |
| D | tma_loads_for_gemm_bank_conflict_elimination | 1.024× | 1.024× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_grid_ali... | Δ | tma_loads_to_elim... | Δ | fix_b_matrix_load... | Δ | tma_loads_for_gem... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 146980 ns | 146110 ns | -870 ns (-0.6%) | 238530 ns | +91550 ns (+62.3%) | 146300 ns | -680 ns (-0.5%) | 143460 ns | -3520 ns (-2.4%) |
| SM Throughput (%) | 36.35 % | 36.39 % | +0.04 pp | 23.18 % | -13.17 pp | 36.36 % | +0.01 pp | 37.24 % | +0.89 pp |
| Memory Throughput (%) | 81.80 % | 81.86 % | +0.06 pp | 89.09 % | +7.29 pp | 81.82 % | +0.02 pp | 84.44 % | +2.64 pp |
| DRAM Throughput (%) | 1.09 % | 1.05 % | -0.04 pp | 0.64 % | -0.45 pp | 1.06 % | -0.03 pp | 1.07 % | -0.02 pp |
| Achieved Occupancy (%) | 17.55 % | 17.53 % | -0.02 pp | 23.58 % | +6.03 pp | 17.54 % | -0.01 pp | 17.54 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 2.80 | +0.94 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 60.22 % | 60.22 % | +0.00 pp | 74.88 % | +14.66 pp | 60.28 % | +0.06 pp | 59.45 % | -0.77 pp |
| IPC Active | 1.59 inst/cycle | 1.59 inst/cycle | +0.00 | 1.00 inst/cycle | -0.59 | 1.59 inst/cycle | +0.00 | 1.62 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.0245× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_shared_memory_transpose_b_with_padding_to_eliminate_bank_conflicts | 0.997× | 1.009× | ✓ |  |
| B | load_b_transposed_to_eliminate_column_stride_bank_conflicts | 1.005× | 0.618× | ✓ |  |
| C | transpose_b_tile_into_padded_shared_memory_to_eliminate_bank_conflicts | 0.997× | 0.999× | ✓ |  |
| D | revert_to_scalar_pointers_with_smem_padding_to_fix_bank_conflicts | 0.996× | 1.000× | ✓ |  |
| E | add_triton_autotune | 0.999× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_shared_m... | Δ | load_b_transposed... | Δ | transpose_b_tile_... | Δ | revert_to_scalar_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 146980 ns | 145730 ns | -1250 ns (-0.9%) | 237890 ns | +90910 ns (+61.9%) | 147170 ns | +190 ns (+0.1%) | 146940 ns | -40 ns (-0.0%) | 146180 ns | -800 ns (-0.5%) |
| SM Throughput (%) | 36.35 % | 36.13 % | -0.22 pp | 23.17 % | -13.18 pp | 36.20 % | -0.15 pp | 36.06 % | -0.29 pp | 36.44 % | +0.09 pp |
| Memory Throughput (%) | 81.80 % | 81.96 % | +0.16 pp | 89.03 % | +7.23 pp | 81.47 % | -0.33 pp | 82.30 % | +0.50 pp | 82.01 % | +0.21 pp |
| DRAM Throughput (%) | 1.09 % | 1.06 % | -0.03 pp | 0.66 % | -0.43 pp | 1.06 % | -0.03 pp | 1.07 % | -0.02 pp | 1.07 % | -0.02 pp |
| Achieved Occupancy (%) | 17.55 % | 17.55 % | +0.00 pp | 23.60 % | +6.05 pp | 17.53 % | -0.02 pp | 17.52 % | -0.03 pp | 17.54 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 2.80 | +0.94 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 60.22 % | 60.40 % | +0.18 pp | 74.86 % | +14.64 pp | 60.21 % | -0.01 pp | 60.69 % | +0.47 pp | 60.23 % | +0.01 pp |
| IPC Active | 1.59 inst/cycle | 1.58 inst/cycle | -0.01 | 1.01 inst/cycle | -0.58 | 1.59 inst/cycle | +0.00 | 1.57 inst/cycle | -0.02 | 1.59 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0245× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_m_to_eliminate_partial_wave_and_reduce_register_pressure | 0.542× | 0.482× | ✓ |  |
| C | increase_num_stages_to_hide_mio_shared_memory_stall_latency | 1.000× | 1.004× | ✓ |  |
| D | tf32_tensor_cores_with_reverted_b_load | 0.996× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_m_... | Δ | increase_num_stag... | Δ | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 146980 ns | 304930 ns | +157950 ns (+107.5%) | 146340 ns | -640 ns (-0.4%) | 147100 ns | +120 ns (+0.1%) |
| SM Throughput (%) | 36.35 % | 34.49 % | -1.86 pp | 36.39 % | +0.04 pp | 36.47 % | +0.12 pp |
| Memory Throughput (%) | 81.80 % | 77.94 % | -3.86 pp | 81.88 % | +0.08 pp | 82.06 % | +0.26 pp |
| DRAM Throughput (%) | 1.09 % | 0.66 % | -0.43 pp | 1.07 % | -0.02 pp | 1.05 % | -0.04 pp |
| Achieved Occupancy (%) | 17.55 % | 12.49 % | -5.06 pp | 17.55 % | +0.00 pp | 17.54 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 252.00 register/thread | +84 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 1.86 | 5.59 | +3.73 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 60.22 % | 62.54 % | +2.32 pp | 60.31 % | +0.09 pp | 60.26 % | +0.04 pp |
| IPC Active | 1.59 inst/cycle | 1.50 inst/cycle | -0.09 | 1.59 inst/cycle | +0.00 | 1.59 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  146980 ns  (1.000×)
Iter 1     146530 ns  (1.003×)  — eviction_policies_streaming_gemm_loads
Iter 2     143460 ns  (1.024×)  — tma_loads_for_gemm_bank_conflict_elimination

Final best: 143460 ns  (1.0245×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -57,23 +71,26 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Loop-invariant N offsets for B load (coalesced, K as fast dimension)
+    offs_n_tile = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 192*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 768*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Coalesced B load: K as fast dimension (stride_bk=1), N as slow dimension (stride_bn=192)
+        # offs_k_tile[:, None] varies on axis-0 with stride 1 (coalesced, bank-conflict-free)
+        offs_k_tile = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
+        b_ptrs = B + offs_k_tile[:, None] * stride_bk + offs_n_tile[None, :] * stride_bn
+        b_mask = (offs_k_tile[:, None] < K) & (offs_n_tile[None, :] < N)
+        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        acc += tl.dot(a, b_tile, allow_tf32=False, out_dtype=ACC_TYPE)
 
 
     # rematerialize rm and rn to save registers
@@ -86,8 +103,6 @@

     # inductor generates a suffix
     xindex = idx_n + 768*idx_m
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
