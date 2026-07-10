# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_bias_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1280000 ns — **Best latency**: 1280000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_bias_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_055/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 198
- **Duration**: 1280000 ns
- **SM Throughput**: 59.60 % | **Memory**: 69.11 % | **DRAM**: 69.11 %
- **Achieved Occupancy**: 12.15 %
- **IPC Active**: 1.05 inst/cycle | **No Eligible**: 73.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tf32_and_expand_autotune_configs | 0.997× | 0.992× | ✓ |  |
| B | enable_tf32_and_expand_autotune_configs_for_small_m_gemm | 0.880× | 0.237× | ✓ |  |
| C | restructure_for_small_m_with_tf32_and_reduced_register_pressure | 0.877× | 0.260× | ✓ |  |
| D | reduce_register_pressure_via_small_block_m_autotune_configs | 0.998× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_tf32_and_e... | Δ | enable_tf32_and_e... | Δ | restructure_for_s... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1280000 ns | 1290000 ns | +10000 ns (+0.8%) | 5390000 ns | +4110000 ns (+321.1%) | 4920000 ns | +3640000 ns (+284.4%) | 1300000 ns | +20000 ns (+1.6%) |
| SM Throughput (%) | 59.60 % | 58.17 % | -1.43 pp | 20.53 % | -39.07 pp | 22.61 % | -36.99 pp | 58.61 % | -0.99 pp |
| Memory Throughput (%) | 69.11 % | 68.16 % | -0.95 pp | 42.77 % | -26.34 pp | 44.50 % | -24.61 pp | 68.07 % | -1.04 pp |
| DRAM Throughput (%) | 69.11 % | 68.16 % | -0.95 pp | 17.82 % | -51.29 pp | 18.94 % | -50.17 pp | 68.07 % | -1.04 pp |
| Achieved Occupancy (%) | 12.15 % | 12.13 % | -0.02 pp | 6.25 % | -5.90 pp | 6.25 % | -5.90 pp | 12.13 % | -0.02 pp |
| Registers/Thread | 198.00 register/thread | 198.00 register/thread | +0 | 148.00 register/thread | -50 | 184.00 register/thread | -14 | 198.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 15.52 | +14.55 | 7.76 | +6.79 | 0.97 | +0.00 |
| No Eligible (%) | 73.75 % | 74.25 % | +0.50 pp | 80.42 % | +6.67 pp | 76.61 % | +2.86 pp | 74.47 % | +0.72 pp |
| IPC Active | 1.05 inst/cycle | 1.03 inst/cycle | -0.02 | 0.78 inst/cycle | -0.27 | 0.94 inst/cycle | -0.11 | 1.03 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | restore_smem_safe_large_block_m_autotune_configs | 0.952× | 0.992× | ✓ |  |
| D | replace_autotune_configs_with_smem_budget_safe_tiles_to_restore_2_blocks_per_sm | 0.951× | 0.992× | ✓ |  |
| E | add_triton_autotune | 0.997× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_smem_safe... | Δ | replace_autotune_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1280000 ns | 1290000 ns | +10000 ns (+0.8%) | 1290000 ns | +10000 ns (+0.8%) | 1300000 ns | +20000 ns (+1.6%) |
| SM Throughput (%) | 59.60 % | 58.68 % | -0.92 pp | 58.76 % | -0.84 pp | 58.53 % | -1.07 pp |
| Memory Throughput (%) | 69.11 % | 68.58 % | -0.53 pp | 68.44 % | -0.67 pp | 68.04 % | -1.07 pp |
| DRAM Throughput (%) | 69.11 % | 68.58 % | -0.53 pp | 68.44 % | -0.67 pp | 68.04 % | -1.07 pp |
| Achieved Occupancy (%) | 12.15 % | 12.14 % | -0.01 pp | 12.14 % | -0.01 pp | 12.14 % | -0.01 pp |
| Registers/Thread | 198.00 register/thread | 198.00 register/thread | +0 | 198.00 register/thread | +0 | 198.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.75 % | 74.37 % | +0.62 pp | 74.21 % | +0.46 pp | 74.43 % | +0.68 pp |
| IPC Active | 1.05 inst/cycle | 1.04 inst/cycle | -0.01 | 1.04 inst/cycle | -0.01 | 1.03 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  1280000 ns  (1.000×)

Final best: 1280000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,16 +4,28 @@

 
 # GEMM kernel: computes out = x @ W.T + bias
 # x: [M, K], W: [N, K], bias: [N], out: [M, N]
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # --- Original configs (unchanged) ---
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # --- New configs: larger BLOCK_K for higher arithmetic intensity ---
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        # --- New configs: small BLOCK_M for M=128 (more CTAs, better occupancy) ---
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -51,7 +63,7 @@

         mask_k = offs_k < k_remaining
         x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
         w = tl.load(w_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
-        acc = tl.dot(x, w, acc)
+        acc = tl.dot(x, w, acc, allow_tf32=True)
         x_ptrs += BLOCK_K * stride_xk
         w_ptrs += BLOCK_K * stride_wk
 
@@ -66,11 +78,6 @@

     tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)
 
 
-# Fused kernel: max_pool1d(kernel_size=2) + sum + scale
-# Input: [M, N] (result of linear)
-# max_pool: for each row, take max of pairs -> [M, N//2]
-# sum: sum over N//2 -> [M]
-# scale: multiply by scale_factor -> [M]
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
@@ -85,6 +92,7 @@

     key=['N_HALF'],
 )
 @triton.jit
+
 def maxpool_sum_scale_kernel(
     In_ptr, Out_ptr,
     M, N, N_HALF,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
