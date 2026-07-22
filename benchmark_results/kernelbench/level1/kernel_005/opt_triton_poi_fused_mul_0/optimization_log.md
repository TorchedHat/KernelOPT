# Kernel Optimization Log

## Summary

- **Kernel**: `mul_scalar_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2010000 ns — **Best latency**: 2010000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `mul_scalar_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_005/synthesis/kernel_0_triton_poi_fused_mul_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 18
- **Duration**: 2010000 ns
- **SM Throughput**: 9.04 % | **Memory**: 86.73 % | **DRAM**: 86.73 %
- **Achieved Occupancy**: 84.94 %
- **IPC Active**: 0.36 inst/cycle | **No Eligible**: 90.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_evict_first | 0.976× | 0.995× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 0.999× | 1.000× | ✓ |  |
| C | vectorized_loads_and_larger_block_size | 0.976× | 0.776× | ✓ |  |
| D | increase_block_size_and_add_software_pipelining | 0.971× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2010000 ns | 2020000 ns | +10000 ns (+0.5%) | 2010000 ns | +0 ns (+0.0%) | 2590000 ns | +580000 ns (+28.9%) | 2050000 ns | +40000 ns (+2.0%) |
| SM Throughput (%) | 9.04 % | 11.63 % | +2.59 pp | 8.98 % | -0.06 pp | 26.27 % | +17.23 pp | 10.34 % | +1.30 pp |
| Memory Throughput (%) | 86.73 % | 86.21 % | -0.52 pp | 86.86 % | +0.13 pp | 67.21 % | -19.52 pp | 84.91 % | -1.82 pp |
| DRAM Throughput (%) | 86.73 % | 86.21 % | -0.52 pp | 86.86 % | +0.13 pp | 67.21 % | -19.52 pp | 84.91 % | -1.82 pp |
| Achieved Occupancy (%) | 84.94 % | 85.04 % | +0.10 pp | 85.02 % | +0.08 pp | 49.92 % | -35.02 pp | 85.02 % | +0.08 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 18.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 1985.94 | +1489.46 | 496.48 | +0.00 |
| No Eligible (%) | 90.99 % | 88.31 % | -2.68 pp | 90.84 % | -0.15 pp | 78.64 % | -12.35 pp | 89.66 % | -1.33 pp |
| IPC Active | 0.36 inst/cycle | 0.47 inst/cycle | +0.11 | 0.36 inst/cycle | +0.00 | 0.85 inst/cycle | +0.49 | 0.42 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_compression_write_streaming_store_and_fma_fusion | 1.000× | 0.995× | ✓ |  |
| B | persistent_kernel_reduce_grid_launch_overhead | 0.925× | 0.944× | ✓ |  |
| C | store_evict_first_to_reduce_l2_pollution | 0.967× | 0.995× | ✓ |  |
| D | fix_autotune_config_space_and_fuse_fp32_multiply_to_fma | 0.993× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_compression_wr... | Δ | persistent_kernel... | Δ | store_evict_first... | Δ | fix_autotune_conf... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2010000 ns | 2020000 ns | +10000 ns (+0.5%) | 2130000 ns | +120000 ns (+6.0%) | 2020000 ns | +10000 ns (+0.5%) | 2010000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 9.04 % | 8.97 % | -0.07 pp | 9.26 % | +0.22 pp | 11.63 % | +2.59 pp | 8.99 % | -0.05 pp |
| Memory Throughput (%) | 86.73 % | 86.47 % | -0.26 pp | 81.80 % | -4.93 pp | 86.16 % | -0.57 pp | 86.85 % | +0.12 pp |
| DRAM Throughput (%) | 86.73 % | 86.47 % | -0.26 pp | 81.80 % | -4.93 pp | 86.16 % | -0.57 pp | 86.85 % | +0.12 pp |
| Achieved Occupancy (%) | 84.94 % | 85.16 % | +0.22 pp | 52.62 % | -32.32 pp | 85.15 % | +0.21 pp | 84.94 % | +0.00 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 198.59 | -297.89 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 90.99 % | 90.78 % | -0.21 pp | 90.72 % | -0.27 pp | 88.16 % | -2.83 pp | 90.95 % | -0.04 pp |
| IPC Active | 0.36 inst/cycle | 0.36 inst/cycle | +0.00 | 0.38 inst/cycle | +0.02 | 0.47 inst/cycle | +0.11 | 0.37 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  2010000 ns  (1.000×)

Final best: 2010000 ns  (1.0000×)
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
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256},   num_warps=4,  num_stages=2),
     ],
     key=['n_elements'],
 )
@@ -25,11 +27,11 @@

     pid = tl.program_id(0)
     block_start = pid * BLOCK_SIZE
     offsets = block_start + tl.arange(0, BLOCK_SIZE)
+    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(in_ptr + offsets, mask=mask)
     result = x * scalar
     tl.store(out_ptr + offsets, result, mask=mask)
-
 
 def kernel_function(a, s):
     output = torch.empty_like(a)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
