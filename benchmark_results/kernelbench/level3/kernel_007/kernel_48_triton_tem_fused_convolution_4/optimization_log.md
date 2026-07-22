# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_48` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 30400 ns — **Best latency**: 30400 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_48`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_48_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.31
- **Registers/thread**: 88
- **Duration**: 30400 ns
- **SM Throughput**: 11.59 % | **Memory**: 77.16 % | **DRAM**: 2.80 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.53 inst/cycle | **No Eligible**: 86.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_pipeline_stages | 0.491× | 0.468× | ✓ |  |
| C | memory_coalescing_via_tile_reshape_and_autotune | 0.999× | 0.991× | ✓ |  |
| D | increase_tile_sizes_and_add_autotune_for_gemm | 0.523× | 0.501× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30400 ns | 64960 ns | +34560 ns (+113.7%) | 30690 ns | +290 ns (+1.0%) | 60700 ns | +30300 ns (+99.7%) |
| SM Throughput (%) | 11.59 % | 18.52 % | +6.93 pp | 11.74 % | +0.15 pp | 19.81 % | +8.22 pp |
| Memory Throughput (%) | 77.16 % | 72.29 % | -4.87 pp | 76.90 % | -0.26 pp | 77.75 % | +0.59 pp |
| DRAM Throughput (%) | 2.80 % | 1.31 % | -1.49 pp | 2.76 % | -0.04 pp | 1.41 % | -1.39 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 88.00 register/thread | 128.00 register/thread | +40 | 64.00 register/thread | -24 | 128.00 register/thread | +40 |
| Waves/SM | 0.31 | 0.23 | -0.08 | 0.12 | -0.19 | 0.31 | +0.00 |
| No Eligible (%) | 86.71 % | 79.39 % | -7.32 pp | 86.59 % | -0.12 pp | 77.72 % | -8.99 pp |
| IPC Active | 0.53 inst/cycle | 0.82 inst/cycle | +0.29 | 0.54 inst/cycle | +0.01 | 0.89 inst/cycle | +0.36 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | proper_autotune_with_tile_sizes_as_kernel_params | 0.946× | 0.937× | ✓ |  |
| B | fix_tile_sizes_as_kernel_parameters_to_eliminate_register_spills | 0.761× | 0.584× | ✓ |  |


### Metric Comparison

| Metric | Baseline | proper_autotune_w... | Δ | fix_tile_sizes_as... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 30400 ns | 32450 ns | +2050 ns (+6.7%) | 52060 ns | +21660 ns (+71.2%) |
| SM Throughput (%) | 11.59 % | 10.89 % | -0.70 pp | 12.22 % | +0.63 pp |
| Memory Throughput (%) | 77.16 % | 72.49 % | -4.67 pp | 85.09 % | +7.93 pp |
| DRAM Throughput (%) | 2.80 % | 2.61 % | -0.19 pp | 1.63 % | -1.17 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 88.00 register/thread | 72.00 register/thread | -16 | 80.00 register/thread | -8 |
| Waves/SM | 0.31 | 0.16 | -0.15 | 0.23 | -0.08 |
| No Eligible (%) | 86.71 % | 87.63 % | +0.92 pp | 86.37 % | -0.34 pp |
| IPC Active | 0.53 inst/cycle | 0.50 inst/cycle | -0.03 | 0.54 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  30400 ns  (1.000×)

Final best: 30400 ns  (1.0000×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_48(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +26,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,7 +72,7 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
@@ -72,9 +86,7 @@

         xindex = idx_n + 64*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +100,6 @@

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
