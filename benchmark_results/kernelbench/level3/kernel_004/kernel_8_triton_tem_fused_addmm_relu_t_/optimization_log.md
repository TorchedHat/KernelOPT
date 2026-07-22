# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9310 ns — **Best latency**: 10270 ns (**0.9065×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_8_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.14
- **Registers/thread**: 72
- **Duration**: 9310 ns
- **SM Throughput**: 9.27 % | **Memory**: 56.07 % | **DRAM**: 3.06 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.49 inst/cycle | **No Eligible**: 87.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.480× | 0.387× | ✓ |  |
| B | tma_loads_and_autotune_tile_sizes | 0.929× | 0.685× | ✓ |  |
| C | memory_coalescing_b_matrix_load_fix_with_autotune | 1.101× | 0.906× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_underutilization_and_bank_conflicts | 0.667× | 0.564× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_and_aut... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9310 ns | 24030 ns | +14720 ns (+158.1%) | 13600 ns | +4290 ns (+46.1%) | 10270 ns | +960 ns (+10.3%) | 16510 ns | +7200 ns (+77.3%) |
| SM Throughput (%) | 9.27 % | 14.12 % | +4.85 pp | 10.23 % | +0.96 pp | 8.19 % | -1.08 pp | 15.66 % | +6.39 pp |
| Memory Throughput (%) | 56.07 % | 59.71 % | +3.64 pp | 69.37 % | +13.30 pp | 50.84 % | -5.23 pp | 63.40 % | +7.33 pp |
| DRAM Throughput (%) | 3.06 % | 1.22 % | -1.84 pp | 2.10 % | -0.96 pp | 2.77 % | -0.29 pp | 1.76 % | -1.30 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp | 6.25 % | +0.01 pp | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 72.00 register/thread | 128.00 register/thread | +56 | 80.00 register/thread | +8 | 64.00 register/thread | -8 | 160.00 register/thread | +88 |
| Waves/SM | 0.14 | 0.48 | +0.34 | 0.16 | +0.02 | 0.12 | -0.02 | 0.32 | +0.18 |
| No Eligible (%) | 87.39 % | 83.40 % | -3.99 pp | 87.09 % | -0.30 pp | 89.59 % | +2.20 pp | 80.26 % | -7.13 pp |
| IPC Active | 0.49 inst/cycle | 0.66 inst/cycle | +0.17 | 0.52 inst/cycle | +0.03 | 0.41 inst/cycle | -0.08 | 0.77 inst/cycle | +0.28 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_eliminate_bank_conflicts | 1.074× | 0.879× | ✓ |  |
| B | revert_tma_to_scalar_pointers_and_fix_block_n_padding | 1.077× | 0.900× | ✓ |  |
| C | reduce_register_pressure_by_shrinking_tiles_to_restore_occupancy | 0.982× | 0.989× | ✓ |  |
| D | reduce_tile_sizes_to_restore_occupancy_and_fix_b_matrix_layout | 1.001× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | revert_tma_to_sca... | Δ | reduce_register_p... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9310 ns | 10590 ns | +1280 ns (+13.7%) | 10340 ns | +1030 ns (+11.1%) | 9410 ns | +100 ns (+1.1%) | 9340 ns | +30 ns (+0.3%) |
| SM Throughput (%) | 9.27 % | 8.14 % | -1.13 pp | 8.12 % | -1.15 pp | 9.45 % | +0.18 pp | 9.25 % | -0.02 pp |
| Memory Throughput (%) | 56.07 % | 50.57 % | -5.50 pp | 51.79 % | -4.28 pp | 57.14 % | +1.07 pp | 55.90 % | -0.17 pp |
| DRAM Throughput (%) | 3.06 % | 2.69 % | -0.37 pp | 2.76 % | -0.30 pp | 3.03 % | -0.03 pp | 3.05 % | -0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 72.00 register/thread | 64.00 register/thread | -8 | 64.00 register/thread | -8 | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 0.14 | 0.12 | -0.02 | 0.12 | -0.02 | 0.14 | +0.00 | 0.14 | +0.00 |
| No Eligible (%) | 87.39 % | 89.33 % | +1.94 pp | 89.96 % | +2.57 pp | 87.70 % | +0.31 pp | 87.53 % | +0.14 pp |
| IPC Active | 0.49 inst/cycle | 0.42 inst/cycle | -0.07 | 0.41 inst/cycle | -0.08 | 0.50 inst/cycle | +0.01 | 0.50 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | transpose_b_load_to_eliminate_shared_memory_bank_conflicts | 0.981× | 0.997× | ✓ |  |
| C | fix_b_matrix_bank_conflicts_via_transposed_load | 1.075× | 0.921× | ✓ |  |
| D | increase_occupancy_via_register_cap_to_hide_bank_conflict_latency | 0.945× | 0.864× | ✓ |  |


### Metric Comparison

| Metric | Baseline | transpose_b_load_... | Δ | fix_b_matrix_bank... | Δ | increase_occupanc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9310 ns | 9340 ns | +30 ns (+0.3%) | 10110 ns | +800 ns (+8.6%) | 10780 ns | +1470 ns (+15.8%) |
| SM Throughput (%) | 9.27 % | 9.46 % | +0.19 pp | 8.15 % | -1.12 pp | 7.87 % | -1.40 pp |
| Memory Throughput (%) | 56.07 % | 57.18 % | +1.11 pp | 52.04 % | -4.03 pp | 48.25 % | -7.82 pp |
| DRAM Throughput (%) | 3.06 % | 3.05 % | -0.01 pp | 2.82 % | -0.24 pp | 2.64 % | -0.42 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 64.00 register/thread | -8 | 48.00 register/thread | -24 |
| Waves/SM | 0.14 | 0.14 | +0.00 | 0.12 | -0.02 | 0.10 | -0.04 |
| No Eligible (%) | 87.39 % | 87.62 % | +0.23 pp | 89.60 % | +2.21 pp | 89.62 % | +2.23 pp |
| IPC Active | 0.49 inst/cycle | 0.50 inst/cycle | +0.01 | 0.41 inst/cycle | -0.08 | 0.41 inst/cycle | -0.08 |

---

## Cumulative Progress

```
Baseline:  9310 ns  (1.000×)

Final best: 10270 ns  (0.9065×)
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
 def triton_tem_fused_addmm_relu_t_8(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -63,18 +64,17 @@

         b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        idx_k = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 84*idx_m
         a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 20*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 84*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
+        # B layout: (20, 84) with strides (84, 1) — B[n,k] = B_ptr + n*84 + k
+        # Load as [BLOCK_K, BLOCK_N]: k contiguous (stride=1), n strided (stride=84)
+        xindex_b = idx_k + offs_b_n[None, :] * 84
+        b = tl.load(B + xindex_b, mask=b_mask, other=0.0)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -91,7 +91,6 @@

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
