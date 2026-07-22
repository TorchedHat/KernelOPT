# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32640 ns — **Best latency**: 32900 ns (**0.9921×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_25`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_25_triton_tem_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 128
- **Duration**: 32640 ns
- **SM Throughput**: 18.48 % | **Memory**: 72.00 % | **DRAM**: 5.08 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.87 inst/cycle | **No Eligible**: 78.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 0.998× | ✓ |  |
| B | add_autotune_with_tma_block_ptr_and_num_stages | 0.944× | 0.959× | ✓ |  |
| C | fix_b_matrix_coalesced_load_and_add_autotune | 1.002× | 0.991× | ✓ | **✓ best** |
| D | add_autotune_to_explore_tile_sizes_and_warps | 0.999× | 0.700× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_with... | Δ | fix_b_matrix_coal... | Δ | add_autotune_to_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32640 ns | 32700 ns | +60 ns (+0.2%) | 34050 ns | +1410 ns (+4.3%) | 32930 ns | +290 ns (+0.9%) | 46660 ns | +14020 ns (+43.0%) |
| SM Throughput (%) | 18.48 % | 18.58 % | +0.10 pp | 17.65 % | -0.83 pp | 18.67 % | +0.19 pp | 12.81 % | -5.67 pp |
| Memory Throughput (%) | 72.00 % | 72.39 % | +0.39 pp | 68.73 % | -3.27 pp | 72.74 % | +0.74 pp | 29.27 % | -42.73 pp |
| DRAM Throughput (%) | 5.08 % | 5.07 % | -0.01 pp | 4.87 % | -0.21 pp | 5.04 % | -0.04 pp | 3.57 % | -1.51 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 128.00 register/thread | +0 | 255.00 register/thread | +127 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 78.43 % | 78.34 % | -0.09 pp | 79.09 % | +0.66 pp | 78.32 % | -0.11 pp | 69.15 % | -9.28 pp |
| IPC Active | 0.87 inst/cycle | 0.87 inst/cycle | +0.00 | 0.84 inst/cycle | -0.03 | 0.86 inst/cycle | -0.01 | 0.61 inst/cycle | -0.26 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_b_matrix_transpose_to_fix_shared_memory_bank_conflicts | 0.995× | 0.991× | ✓ |  |
| C | fix_b_matrix_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 0.950× | 0.961× | ✓ |  |
| D | fix_register_pressure_by_promoting_tile_sizes_to_autotune_params | 1.003× | 0.700× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_b_matri... | Δ | fix_b_matrix_bloc... | Δ | fix_register_pres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32640 ns | 32930 ns | +290 ns (+0.9%) | 33980 ns | +1340 ns (+4.1%) | 46660 ns | +14020 ns (+43.0%) |
| SM Throughput (%) | 18.48 % | 18.68 % | +0.20 pp | 17.64 % | -0.84 pp | 12.82 % | -5.66 pp |
| Memory Throughput (%) | 72.00 % | 72.79 % | +0.79 pp | 68.68 % | -3.32 pp | 29.30 % | -42.70 pp |
| DRAM Throughput (%) | 5.08 % | 5.04 % | -0.04 pp | 4.88 % | -0.20 pp | 3.57 % | -1.51 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 255.00 register/thread | +127 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 78.43 % | 78.26 % | -0.17 pp | 78.99 % | +0.56 pp | 69.22 % | -9.21 pp |
| IPC Active | 0.87 inst/cycle | 0.86 inst/cycle | -0.01 | 0.84 inst/cycle | -0.03 | 0.61 inst/cycle | -0.26 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_b_global_load_with_tl_trans_before_dot | 0.995× | 0.994× | ✓ |  |
| B | fix_b_matrix_load_as_transposed_tile_to_eliminate_bank_conflicts | 0.942× | 0.943× | ✓ |  |
| C | promote_tile_sizes_to_kernel_params_and_autotune_correctly | 1.005× | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_b_globa... | Δ | fix_b_matrix_load... | Δ | promote_tile_size... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32640 ns | 32830 ns | +190 ns (+0.6%) | 34620 ns | +1980 ns (+6.1%) | 32900 ns | +260 ns (+0.8%) |
| SM Throughput (%) | 18.48 % | 18.56 % | +0.08 pp | 17.66 % | -0.82 pp | 18.38 % | -0.10 pp |
| Memory Throughput (%) | 72.00 % | 72.33 % | +0.33 pp | 69.00 % | -3.00 pp | 71.60 % | -0.40 pp |
| DRAM Throughput (%) | 5.08 % | 5.05 % | -0.03 pp | 4.79 % | -0.29 pp | 5.04 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 78.43 % | 78.31 % | -0.12 pp | 79.16 % | +0.73 pp | 78.02 % | -0.41 pp |
| IPC Active | 0.87 inst/cycle | 0.86 inst/cycle | -0.01 | 0.83 inst/cycle | -0.04 | 0.87 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tl_trans_bank_conflicts_and_cap_registers_with_maxnreg | 0.871× | 0.486× | ✓ |  |
| B | fix_b_matrix_global_load_coalescing_with_tl_trans_for_dot | 1.002× | 0.994× | ✓ |  |
| C | load_b_as_k_by_n_tile_to_eliminate_tl_trans_bank_conflicts | 0.945× | 0.944× | ✓ |  |
| D | add_triton_autotune | 0.995× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | fix_b_matrix_glob... | Δ | load_b_as_k_by_n_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32640 ns | 67100 ns | +34460 ns (+105.6%) | 32830 ns | +190 ns (+0.6%) | 34590 ns | +1950 ns (+6.0%) | 32990 ns | +350 ns (+1.1%) |
| SM Throughput (%) | 18.48 % | 15.85 % | -2.63 pp | 18.53 % | +0.05 pp | 17.57 % | -0.91 pp | 18.58 % | +0.10 pp |
| Memory Throughput (%) | 72.00 % | 51.71 % | -20.29 pp | 72.13 % | +0.13 pp | 68.39 % | -3.61 pp | 72.39 % | +0.39 pp |
| DRAM Throughput (%) | 5.08 % | 2.59 % | -2.49 pp | 5.05 % | -0.03 pp | 4.79 % | -0.29 pp | 5.03 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 78.43 % | 85.32 % | +6.89 pp | 78.53 % | +0.10 pp | 79.11 % | +0.68 pp | 78.32 % | -0.11 pp |
| IPC Active | 0.87 inst/cycle | 0.59 inst/cycle | -0.28 | 0.86 inst/cycle | -0.01 | 0.83 inst/cycle | -0.04 | 0.86 inst/cycle | -0.01 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_smem_bank_conflicts_via_block_ptr_with_correct_order | 0.912× | 0.903× | ✓ |  |
| B | reduce_block_k_to_break_shared_memory_occupancy_limit | 0.936× | 0.934× | ✓ |  |
| D | restore_coalesced_b_load_with_tl_trans_and_remove_useless_maxnreg | 0.991× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ | reduce_block_k_to... | Δ | restore_coalesced... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32640 ns | 36160 ns | +3520 ns (+10.8%) | 34940 ns | +2300 ns (+7.0%) | 32860 ns | +220 ns (+0.7%) |
| SM Throughput (%) | 18.48 % | 16.79 % | -1.69 pp | 17.38 % | -1.10 pp | 18.70 % | +0.22 pp |
| Memory Throughput (%) | 72.00 % | 65.03 % | -6.97 pp | 67.41 % | -4.59 pp | 72.86 % | +0.86 pp |
| DRAM Throughput (%) | 5.08 % | 4.68 % | -0.40 pp | 4.73 % | -0.35 pp | 5.05 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.23 | -0.08 | 0.31 | +0.00 |
| No Eligible (%) | 78.43 % | 79.96 % | +1.53 pp | 79.95 % | +1.52 pp | 78.30 % | -0.13 pp |
| IPC Active | 0.87 inst/cycle | 0.79 inst/cycle | -0.08 | 0.80 inst/cycle | -0.07 | 0.85 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  32640 ns  (1.000×)

Final best: 32900 ns  (0.9921×)
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
 def triton_tem_fused_convolution_25(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -60,18 +61,15 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        b_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)  # [1, BLOCK_K] row vector
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 256*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b_T = tl.load(B + offs_b_n[:, None] * stride_bn + b_k_idx_vals * stride_bk)  # [BLOCK_N, BLOCK_K] coalesced
+        b = tl.trans(b_T)  # [BLOCK_K, BLOCK_N] for tl.dot
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -88,7 +86,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
