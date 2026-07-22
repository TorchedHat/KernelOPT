# Kernel Optimization Log

## Summary

- **Kernel**: `argmax_dim1_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17590000 ns — **Best latency**: 17590000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `argmax_dim1_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_051/synthesis/kernel_0_triton_red_fused_argmax_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524160,1,1), Waves/SM: 248.18
- **Registers/thread**: 32
- **Duration**: 17590000 ns
- **SM Throughput**: 57.88 % | **Memory**: 81.96 % | **DRAM**: 9.98 %
- **Achieved Occupancy**: 99.62 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 51.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_via_warp_reduction_reindex | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 17590000 ns | 17590000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 57.88 % | 57.87 % | -0.01 pp |
| Memory Throughput (%) | 81.96 % | 81.93 % | -0.03 pp |
| DRAM Throughput (%) | 9.98 % | 9.98 % | +0.00 pp |
| Achieved Occupancy (%) | 99.62 % | 99.61 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 248.18 | 248.18 | +0.00 |
| No Eligible (%) | 51.78 % | 51.77 % | -0.01 pp |
| IPC Active | 1.93 inst/cycle | 1.93 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_evict_first | 0.975× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 17590000 ns | 17990000 ns | +400000 ns (+2.3%) |
| SM Throughput (%) | 57.88 % | 56.59 % | -1.29 pp |
| Memory Throughput (%) | 81.96 % | 81.65 % | -0.31 pp |
| DRAM Throughput (%) | 9.98 % | 9.75 % | -0.23 pp |
| Achieved Occupancy (%) | 99.62 % | 99.60 % | -0.02 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 |
| Waves/SM | 248.18 | 248.18 | +0.00 |
| No Eligible (%) | 51.78 % | 52.80 % | +1.02 pp |
| IPC Active | 1.93 inst/cycle | 1.89 inst/cycle | -0.04 |

---

## Cumulative Progress

```
Baseline:  17590000 ns  (1.000×)

Final best: 17590000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,16 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 1}, num_warps=4),
-        triton.Config({'BLOCK_R': 256, 'BLOCK_X': 1}, num_warps=4),
-        triton.Config({'BLOCK_R': 512, 'BLOCK_X': 1}, num_warps=8),
-        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 1}, num_warps=8),
-        triton.Config({'BLOCK_R': 2048, 'BLOCK_X': 1}, num_warps=16),
-        triton.Config({'BLOCK_R': 4096, 'BLOCK_X': 1}, num_warps=16),
+        triton.Config({'BLOCK_R': 128,  'BLOCK_X': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_R': 2048, 'BLOCK_X': 1}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_R': 4096, 'BLOCK_X': 1}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_R': 4096, 'BLOCK_X': 1}, num_warps=32, num_stages=1),
     ],
     key=['R', 'N'],
 )
@@ -42,6 +44,8 @@

     # Iterate over reduction dimension in blocks
     for r_start in tl.range(0, R, BLOCK_R):
         r_offsets = r_start + tl.arange(0, BLOCK_R)
+        # Vectorization hints: r_offsets is contiguous and BLOCK_R-aligned
+        r_offsets = tl.max_contiguous(tl.multiple_of(r_offsets, BLOCK_R), BLOCK_R)
         mask = r_offsets < R
         # Load x[batch_idx, r_offsets, col_idx]
         ptrs = base + r_offsets * N
@@ -65,7 +69,6 @@

             max_idx = tl.minimum(max_idx, local_idx)
 
     tl.store(out_ptr + batch_idx * N + col_idx, max_idx)
-
 
 def kernel_function(x):
     B, R, N = x.shape  # 128, 4096, 4095
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
