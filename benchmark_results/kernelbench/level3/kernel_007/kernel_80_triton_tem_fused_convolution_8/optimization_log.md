# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_80` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48770 ns — **Best latency**: 54240 ns (**0.8992×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_80`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_80_triton_tem_fused_convolution_8/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.16
- **Registers/thread**: 89
- **Duration**: 48770 ns
- **SM Throughput**: 6.35 % | **Memory**: 43.10 % | **DRAM**: 0.87 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.547× | 0.532× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_and_add_autotune | 1.001× | 0.866× | ✓ |  |
| D | autotune_tile_sizes_to_increase_grid_coverage_and_reduce_bank_conflicts | 1.024× | 0.899× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_shared_memory... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48770 ns | 91740 ns | +42970 ns (+88.1%) | 56350 ns | +7580 ns (+15.5%) | 54240 ns | +5470 ns (+11.2%) |
| SM Throughput (%) | 6.35 % | 11.03 % | +4.68 pp | 5.04 % | -1.31 pp | 5.01 % | -1.34 pp |
| Memory Throughput (%) | 43.10 % | 42.84 % | -0.26 pp | 37.32 % | -5.78 pp | 36.07 % | -7.03 pp |
| DRAM Throughput (%) | 0.87 % | 0.46 % | -0.41 pp | 0.76 % | -0.11 pp | 0.78 % | -0.09 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 89.00 register/thread | 128.00 register/thread | +39 | 150.00 register/thread | +61 | 96.00 register/thread | +7 |
| Waves/SM | 0.16 | 0.12 | -0.04 | 0.08 | -0.08 | 0.05 | -0.11 |
| No Eligible (%) | 86.24 % | 76.62 % | -9.62 pp | 78.48 % | -7.76 pp | 78.61 % | -7.63 pp |
| IPC Active | 0.55 inst/cycle | 0.93 inst/cycle | +0.38 | 0.43 inst/cycle | -0.12 | 0.43 inst/cycle | -0.12 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  48770 ns  (1.000×)

Final best: 54240 ns  (0.8992×)
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
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_80(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -12,7 +26,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,10 +72,10 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
         b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
@@ -75,9 +89,7 @@

         xindex = idx_n + 128*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 832*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -91,8 +103,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
