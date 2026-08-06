# Kernel Optimization Log

## Summary

- **Kernel**: `batch_norm_inference_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2010000 ns — **Best latency**: 2010000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `batch_norm_inference_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_033/synthesis/kernel_0_triton_poi_fused__native_batch_norm_legi.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 32
- **Duration**: 2010000 ns
- **SM Throughput**: 65.36 % | **Memory**: 86.81 % | **DRAM**: 86.81 %
- **Achieved Occupancy**: 86.02 %
- **IPC Active**: 2.62 inst/cycle | **No Eligible**: 35.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_and_scalar_broadcast_for_channel_params | 0.974× | 0.948× | ✓ |  |
| D | increase_block_size_and_add_num_stages_for_memory_latency_hiding | 0.999× | 0.985× | ✓ |  |
| E | add_triton_autotune | 0.958× | 0.390× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_block_si... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2010000 ns | 2120000 ns | +110000 ns (+5.5%) | 2040000 ns | +30000 ns (+1.5%) | 5160000 ns | +3150000 ns (+156.7%) |
| SM Throughput (%) | 65.36 % | 19.62 % | -45.74 pp | 69.53 % | +4.17 pp | 59.90 % | -5.46 pp |
| Memory Throughput (%) | 86.81 % | 82.22 % | -4.59 pp | 85.56 % | -1.25 pp | 33.77 % | -53.04 pp |
| DRAM Throughput (%) | 86.81 % | 82.22 % | -4.59 pp | 85.56 % | -1.25 pp | 33.77 % | -53.04 pp |
| Achieved Occupancy (%) | 86.02 % | 49.48 % | -36.54 pp | 83.64 % | -2.38 pp | 60.51 % | -25.51 pp |
| Registers/Thread | 32.00 register/thread | 48.00 register/thread | +16 | 28.00 register/thread | -4 | 18.00 register/thread | -14 |
| Waves/SM | 496.48 | 198.59 | -297.89 | 992.97 | +496.49 | 3971.88 | +3475.40 |
| No Eligible (%) | 35.54 % | 80.30 % | +44.76 pp | 30.38 % | -5.16 pp | 40.09 % | +4.55 pp |
| IPC Active | 2.62 inst/cycle | 0.80 inst/cycle | -1.82 | 2.82 inst/cycle | +0.20 | 2.40 inst/cycle | -0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | precompute_scalar_fma_coefficients_to_reduce_register_pressure_and_fuse_fp32_ops | 0.975× | 0.944× | ✓ |  |
| D | add_triton_autotune | 0.958× | 0.390× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_scalar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2010000 ns | 2130000 ns | +120000 ns (+6.0%) | 5160000 ns | +3150000 ns (+156.7%) |
| SM Throughput (%) | 65.36 % | 10.80 % | -54.56 pp | 59.90 % | -5.46 pp |
| Memory Throughput (%) | 86.81 % | 81.80 % | -5.01 pp | 33.77 % | -53.04 pp |
| DRAM Throughput (%) | 86.81 % | 81.80 % | -5.01 pp | 33.77 % | -53.04 pp |
| Achieved Occupancy (%) | 86.02 % | 48.72 % | -37.30 pp | 60.52 % | -25.50 pp |
| Registers/Thread | 32.00 register/thread | 48.00 register/thread | +16 | 18.00 register/thread | -14 |
| Waves/SM | 496.48 | 198.59 | -297.89 | 3971.88 | +3475.40 |
| No Eligible (%) | 35.54 % | 89.01 % | +53.47 pp | 40.08 % | +4.54 pp |
| IPC Active | 2.62 inst/cycle | 0.44 inst/cycle | -2.18 | 2.40 inst/cycle | -0.22 |

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

@@ -3,14 +3,20 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=3),
     ],
     key=['numel'],
 )
@@ -47,7 +53,6 @@

     out = x_norm * weight + bias
 
     tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)
-
 
 def kernel_function(p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x):
     # x: [N, C, H, W]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
