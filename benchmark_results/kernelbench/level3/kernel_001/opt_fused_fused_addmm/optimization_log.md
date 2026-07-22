# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4100000 ns — **Best latency**: 7450000 ns (**0.5503×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_bias_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_001/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (128,1,1), Waves/SM: 0.97
- **Registers/thread**: 246
- **Duration**: 4100000 ns
- **SM Throughput**: 37.35 % | **Memory**: 82.98 % | **DRAM**: 5.38 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | reduce_register_pressure_via_smaller_tiles_to_fix_bank_conflicts_and_occupancy | 0.972× | 0.551× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4100000 ns | 7440000 ns | +3340000 ns (+81.5%) |
| SM Throughput (%) | 37.35 % | 22.11 % | -15.24 pp |
| Memory Throughput (%) | 82.98 % | 91.53 % | +8.55 pp |
| DRAM Throughput (%) | 5.38 % | 3.07 % | -2.31 pp |
| Achieved Occupancy (%) | 12.50 % | 15.71 % | +3.21 pp |
| Registers/Thread | 246.00 register/thread | 160.00 register/thread | -86 |
| Waves/SM | 0.97 | 1.29 | +0.32 |
| No Eligible (%) | 61.23 % | 76.70 % | +15.47 pp |
| IPC Active | 1.55 inst/cycle | 0.93 inst/cycle | -0.62 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_block_ptr_swizzled_layout | 1.007× | 0.550× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_block_ptr_with_explicit_order | 1.003× | 0.550× | ✓ |  |
| D | fix_shared_memory_bank_conflicts_via_block_ptr_swizzle_and_grid_alignment | 1.004× | 0.998× | ✓ |  |
| E | add_triton_autotune | 1.013× | 0.550× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | fix_shared_memory... | Δ | fix_shared_memory... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4100000 ns | 7460000 ns | +3360000 ns (+82.0%) | 7460000 ns | +3360000 ns (+82.0%) | 4110000 ns | +10000 ns (+0.2%) | 7450000 ns | +3350000 ns (+81.7%) |
| SM Throughput (%) | 37.35 % | 21.84 % | -15.51 pp | 21.79 % | -15.56 pp | 37.35 % | +0.00 pp | 22.12 % | -15.23 pp |
| Memory Throughput (%) | 82.98 % | 91.88 % | +8.90 pp | 91.35 % | +8.37 pp | 82.96 % | -0.02 pp | 91.58 % | +8.60 pp |
| DRAM Throughput (%) | 5.38 % | 3.05 % | -2.33 pp | 3.07 % | -2.31 pp | 5.38 % | +0.00 pp | 3.06 % | -2.32 pp |
| Achieved Occupancy (%) | 12.50 % | 15.75 % | +3.25 pp | 15.75 % | +3.25 pp | 12.50 % | +0.00 pp | 15.74 % | +3.24 pp |
| Registers/Thread | 246.00 register/thread | 148.00 register/thread | -98 | 166.00 register/thread | -80 | 255.00 register/thread | +9 | 160.00 register/thread | -86 |
| Waves/SM | 0.97 | 1.29 | +0.32 | 1.29 | +0.32 | 0.97 | +0.00 | 1.29 | +0.32 |
| No Eligible (%) | 61.23 % | 77.01 % | +15.78 pp | 77.00 % | +15.77 pp | 61.18 % | -0.05 pp | 76.70 % | +15.47 pp |
| IPC Active | 1.55 inst/cycle | 0.91 inst/cycle | -0.64 | 0.92 inst/cycle | -0.63 | 1.55 inst/cycle | +0.00 | 0.93 inst/cycle | -0.62 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | explicit_shared_memory_padding_to_eliminate_bank_conflicts | 1.013× | 0.550× | ✓ |  |
| C | restore_large_tile_autotune_configs_to_recover_original_performance | 0.903× | 0.965× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_shared_m... | Δ | restore_large_til... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4100000 ns | 7450000 ns | +3350000 ns (+81.7%) | 4250000 ns | +150000 ns (+3.7%) |
| SM Throughput (%) | 37.35 % | 22.11 % | -15.24 pp | 36.00 % | -1.35 pp |
| Memory Throughput (%) | 82.98 % | 91.54 % | +8.56 pp | 80.08 % | -2.90 pp |
| DRAM Throughput (%) | 5.38 % | 3.06 % | -2.32 pp | 5.19 % | -0.19 pp |
| Achieved Occupancy (%) | 12.50 % | 15.74 % | +3.24 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 246.00 register/thread | 160.00 register/thread | -86 | 254.00 register/thread | +8 |
| Waves/SM | 0.97 | 1.29 | +0.32 | 0.97 | +0.00 |
| No Eligible (%) | 61.23 % | 76.70 % | +15.47 pp | 62.51 % | +1.28 pp |
| IPC Active | 1.55 inst/cycle | 0.93 inst/cycle | -0.62 | 1.50 inst/cycle | -0.05 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4100000 ns  (1.000×)

Final best: 7450000 ns  (0.5503×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,20 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # grid_m=1 for M=128, grid_n=128 -> 128 blocks (~1 full wave on 132 SMs)
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # grid=2*128=256 blocks (1.94 waves, cleaner than 7.76 partial waves)
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -37,24 +43,33 @@

     
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
     
-    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
-    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
+    # Use tl.make_block_ptr with explicit order for conflict-free shared memory swizzle
+    A_block_ptr = tl.make_block_ptr(
+        base=A_ptr,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),  # row-major: innermost dim is K-axis, enables conflict-free SMEM swizzle
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B_ptr,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),  # col-major for B: innermost dim is N-axis, conflict-free SMEM layout
+    )
     
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
-        mask_k = offs_k < k_remaining
-        
-        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
-        b = tl.load(B_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
-        
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc = tl.dot(a, b, acc, allow_tf32=False)
-        
-        A_ptrs += BLOCK_K * stride_ak
-        B_ptrs += BLOCK_K * stride_bk
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
     
     # Load bias and add
     bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
@@ -67,7 +82,6 @@

     C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
     mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     tl.store(C_ptrs, acc.to(tl.float32), mask=mask)
-
 
 def kernel_function(w0, b0, w1, b1, w2, b2, x):
     M = x.shape[0]  # 128
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
