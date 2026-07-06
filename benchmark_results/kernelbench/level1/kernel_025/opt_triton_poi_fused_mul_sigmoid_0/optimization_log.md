# Kernel Optimization Log

## Summary

- **Kernel**: `swish_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 3100000 ns (**0.9742×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `swish_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_025/synthesis/kernel_0_triton_poi_fused_mul_sigmoid_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 28
- **Duration**: 3020000 ns
- **SM Throughput**: 34.26 % | **Memory**: 86.63 % | **DRAM**: 86.63 %
- **Achieved Occupancy**: 86.05 %
- **IPC Active**: 1.38 inst/cycle | **No Eligible**: 65.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_evict_first | 0.976× | 0.953× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 0.995× | 0.962× | ✓ |  |
| C | add_software_pipelining_and_expand_autotune_configs | 1.001× | 0.782× | ✓ | **✓ best** |
| D | increase_block_size_and_add_software_pipelining | 1.001× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | vectorized_loads_... | Δ | add_software_pipe... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3170000 ns | +150000 ns (+5.0%) | 3140000 ns | +120000 ns (+4.0%) | 3860000 ns | +840000 ns (+27.8%) | 3090000 ns | +70000 ns (+2.3%) |
| SM Throughput (%) | 34.26 % | 34.37 % | +0.11 pp | 32.76 % | -1.50 pp | 29.67 % | -4.59 pp | 32.93 % | -1.33 pp |
| Memory Throughput (%) | 86.63 % | 82.53 % | -4.10 pp | 83.24 % | -3.39 pp | 67.88 % | -18.75 pp | 84.56 % | -2.07 pp |
| DRAM Throughput (%) | 86.63 % | 82.53 % | -4.10 pp | 83.24 % | -3.39 pp | 67.88 % | -18.75 pp | 84.56 % | -2.07 pp |
| Achieved Occupancy (%) | 86.05 % | 74.76 % | -11.29 pp | 76.89 % | -9.16 pp | 33.93 % | -52.12 pp | 72.19 % | -13.86 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 18.00 register/thread | -10 | 32.00 register/thread | +4 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 372.36 | -372.37 | 1489.45 | +744.72 | 372.36 | -372.37 |
| No Eligible (%) | 65.52 % | 65.90 % | +0.38 pp | 68.54 % | +3.02 pp | 70.19 % | +4.67 pp | 68.31 % | +2.79 pp |
| IPC Active | 1.38 inst/cycle | 1.36 inst/cycle | -0.02 | 1.26 inst/cycle | -0.12 | 1.19 inst/cycle | -0.19 | 1.28 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_sigmoid_with_tl_sigmoid_for_fma_fusion | 1.001× | 0.974× | ✓ | **✓ best** |
| B | fused_fp32_sigmoid_via_fast_math_intrinsic | 0.995× | 0.959× | ✓ |  |
| C | fused_fp32_instructions_via_tl_sigmoid_and_fast_exp | 0.976× | 0.953× | ✓ |  |
| D | remove_small_block_configs_to_restore_occupancy | 1.001× | 0.391× | ✓ |  |
| E | add_triton_autotune | 1.001× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_si... | Δ | fused_fp32_sigmoi... | Δ | fused_fp32_instru... | Δ | remove_small_bloc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3100000 ns | +80000 ns (+2.6%) | 3150000 ns | +130000 ns (+4.3%) | 3170000 ns | +150000 ns (+5.0%) | 7730000 ns | +4710000 ns (+156.0%) | 3090000 ns | +70000 ns (+2.3%) |
| SM Throughput (%) | 34.26 % | 32.97 % | -1.29 pp | 32.58 % | -1.68 pp | 34.32 % | +0.06 pp | 27.95 % | -6.31 pp | 32.91 % | -1.35 pp |
| Memory Throughput (%) | 86.63 % | 84.54 % | -2.09 pp | 83.20 % | -3.43 pp | 82.48 % | -4.15 pp | 33.86 % | -52.77 pp | 84.59 % | -2.04 pp |
| DRAM Throughput (%) | 86.63 % | 84.54 % | -2.09 pp | 83.20 % | -3.43 pp | 82.48 % | -4.15 pp | 33.86 % | -52.77 pp | 84.59 % | -2.04 pp |
| Achieved Occupancy (%) | 86.05 % | 72.25 % | -13.80 pp | 76.96 % | -9.09 pp | 74.74 % | -11.31 pp | 48.00 % | -38.05 pp | 71.94 % | -14.11 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 372.36 | -372.37 | 372.36 | -372.37 | 5957.82 | +5213.09 | 372.36 | -372.37 |
| No Eligible (%) | 65.52 % | 68.17 % | +2.65 pp | 68.58 % | +3.06 pp | 65.90 % | +0.38 pp | 72.05 % | +6.53 pp | 68.18 % | +2.66 pp |
| IPC Active | 1.38 inst/cycle | 1.28 inst/cycle | -0.10 | 1.26 inst/cycle | -0.12 | 1.36 inst/cycle | -0.02 | 1.12 inst/cycle | -0.26 | 1.27 inst/cycle | -0.11 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_native_half_precision_compute | 1.001× | 0.977× | ✓ |  |
| B | write_through_store_to_bypass_l2_saturation | 0.995× | 0.962× | ✓ |  |
| C | persistent_grid_stride_loop_to_reduce_launch_overhead_and_hide_memory_latency | 0.975× | 0.962× | ✓ |  |
| D | large_block_configs_with_streaming_evict_first_load | 0.973× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | write_through_sto... | Δ | persistent_grid_s... | Δ | large_block_confi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3090000 ns | +70000 ns (+2.3%) | 3140000 ns | +120000 ns (+4.0%) | 3140000 ns | +120000 ns (+4.0%) | 3100000 ns | +80000 ns (+2.6%) |
| SM Throughput (%) | 34.26 % | 32.92 % | -1.34 pp | 32.44 % | -1.82 pp | 36.20 % | +1.94 pp | 34.94 % | +0.68 pp |
| Memory Throughput (%) | 86.63 % | 84.60 % | -2.03 pp | 83.22 % | -3.41 pp | 83.27 % | -3.36 pp | 84.43 % | -2.20 pp |
| DRAM Throughput (%) | 86.63 % | 84.60 % | -2.03 pp | 83.22 % | -3.41 pp | 83.27 % | -3.36 pp | 84.43 % | -2.20 pp |
| Achieved Occupancy (%) | 86.05 % | 72.05 % | -14.00 pp | 76.92 % | -9.13 pp | 76.80 % | -9.25 pp | 85.98 % | -0.07 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 372.36 | -372.37 | 372.36 | -372.37 | 744.73 | +0.00 |
| No Eligible (%) | 65.52 % | 68.19 % | +2.67 pp | 68.43 % | +2.91 pp | 64.08 % | -1.44 pp | 65.06 % | -0.46 pp |
| IPC Active | 1.38 inst/cycle | 1.28 inst/cycle | -0.10 | 1.26 inst/cycle | -0.12 | 1.43 inst/cycle | +0.05 | 1.41 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cache_streaming_store_to_relieve_l2_saturation | 0.998× | 0.965× | ✓ |  |
| B | fused_fma_swish_via_tanh_identity_to_eliminate_nonfused_fp32_instructions | 0.973× | 0.974× | ✓ |  |
| D | dual_chunk_unrolled_loop_to_reduce_l2_transaction_pressure | 0.966× | 0.896× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cache_streaming_s... | Δ | fused_fma_swish_v... | Δ | dual_chunk_unroll... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3130000 ns | +110000 ns (+3.6%) | 3100000 ns | +80000 ns (+2.6%) | 3370000 ns | +350000 ns (+11.6%) |
| SM Throughput (%) | 34.26 % | 32.60 % | -1.66 pp | 35.10 % | +0.84 pp | 66.58 % | +32.32 pp |
| Memory Throughput (%) | 86.63 % | 83.62 % | -3.01 pp | 84.36 % | -2.27 pp | 77.60 % | -9.03 pp |
| DRAM Throughput (%) | 86.63 % | 83.62 % | -3.01 pp | 84.36 % | -2.27 pp | 77.60 % | -9.03 pp |
| Achieved Occupancy (%) | 86.05 % | 72.19 % | -13.86 pp | 86.07 % | +0.02 pp | 41.06 % | -44.99 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 58.00 register/thread | +30 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 65.52 % | 68.64 % | +3.12 pp | 65.05 % | -0.47 pp | 33.31 % | -32.21 pp |
| IPC Active | 1.38 inst/cycle | 1.25 inst/cycle | -0.13 | 1.40 inst/cycle | +0.02 | 2.69 inst/cycle | +1.31 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)

Final best: 3100000 ns  (0.9742×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,37 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=32, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256},   num_warps=4,  num_stages=2),
+        # Existing configs exploring XBLOCK (128, 256, 512, 1024) with num_warps 4/8
+        triton.Config({'BLOCK_SIZE': 128},   num_warps=4),
+        triton.Config({'BLOCK_SIZE': 256},   num_warps=8),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=8),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
+        # New num_stages=1 variants to reduce register pressure from software pipelining
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=32, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
     ],
     key=['n_elements'],
 )
@@ -27,14 +50,10 @@

     mask = offsets < n_elements
 
     x = tl.load(x_ptr + offsets, mask=mask)
-    # sigmoid(x) = 1 / (1 + exp(-x))
     # swish(x) = x * sigmoid(x)
-    x_f32 = x.to(tl.float32)
-    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_f32))
-    result = x_f32 * sigmoid_x
+    result = x * tl.sigmoid(x)
 
-    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)
-
+    tl.store(out_ptr + offsets, result, mask=mask)
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
