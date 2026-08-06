# Kernel Optimization Log

## Summary

- **Kernel**: `conv_transpose1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16230000 ns — **Best latency**: 16260000 ns (**0.9982×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv_transpose1d_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_064/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131200,1,1), Waves/SM: 331.31
- **Registers/thread**: 139
- **Duration**: 16230000 ns
- **SM Throughput**: 48.88 % | **Memory**: 65.57 % | **DRAM**: 8.06 %
- **Achieved Occupancy**: 18.68 %
- **IPC Active**: 1.89 inst/cycle | **No Eligible**: 52.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.000× | ✓ |  |
| B | restructure_k_loop_to_eliminate_uncoalesced_access_and_reduce_register_pressure | 0.936× | 0.582× | ✓ |  |
| C | eviction_policies_for_streaming_loads | 1.002× | 1.002× | ✓ |  |
| D | reduce_register_pressure_via_autotune_configs | 1.004× | 0.723× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | restructure_k_loo... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16230000 ns | 16230000 ns | +0 ns (+0.0%) | 27880000 ns | +11650000 ns (+71.8%) | 16190000 ns | -40000 ns (-0.2%) | 22440000 ns | +6210000 ns (+38.3%) |
| SM Throughput (%) | 48.88 % | 48.87 % | -0.01 pp | 11.02 % | -37.86 pp | 49.00 % | +0.12 pp | 60.52 % | +11.64 pp |
| Memory Throughput (%) | 65.57 % | 65.56 % | -0.01 pp | 85.20 % | +19.63 pp | 65.69 % | +0.12 pp | 89.58 % | +24.01 pp |
| DRAM Throughput (%) | 8.06 % | 8.07 % | +0.01 pp | 4.70 % | -3.36 pp | 8.09 % | +0.03 pp | 9.74 % | +1.68 pp |
| Achieved Occupancy (%) | 18.68 % | 18.68 % | +0.00 pp | 12.47 % | -6.21 pp | 18.67 % | -0.01 pp | 24.85 % | +6.17 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 | 255.00 register/thread | +116 | 139.00 register/thread | +0 | 128.00 register/thread | -11 |
| Waves/SM | 331.31 | 331.31 | +0.00 | 496.97 | +165.66 | 331.31 | +0.00 | 496.73 | +165.42 |
| No Eligible (%) | 52.70 % | 52.71 % | +0.01 pp | 88.97 % | +36.27 pp | 52.56 % | -0.14 pp | 42.55 % | -10.15 pp |
| IPC Active | 1.89 inst/cycle | 1.89 inst/cycle | +0.00 | 0.44 inst/cycle | -1.45 | 1.90 inst/cycle | +0.01 | 2.30 inst/cycle | +0.41 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_output_store_by_transposing_accumulator_tile | 0.902× | 1.002× | ✓ |  |
| C | pretranspose_input_to_channel_last_for_coalesced_x_loads | 1.004× | 0.999× | ✓ |  |
| D | revert_to_flat_k_loop_with_transposed_x_for_coalesced_access | 1.189× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesce_output_s... | Δ | pretranspose_inpu... | Δ | revert_to_flat_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16230000 ns | 16190000 ns | -40000 ns (-0.2%) | 16240000 ns | +10000 ns (+0.1%) | 16260000 ns | +30000 ns (+0.2%) |
| SM Throughput (%) | 48.88 % | 49.00 % | +0.12 pp | 48.85 % | -0.03 pp | 49.47 % | +0.59 pp |
| Memory Throughput (%) | 65.57 % | 65.71 % | +0.14 pp | 65.54 % | -0.03 pp | 68.38 % | +2.81 pp |
| DRAM Throughput (%) | 8.06 % | 8.08 % | +0.02 pp | 8.06 % | +0.00 pp | 8.05 % | -0.01 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp | 18.68 % | +0.00 pp | 18.68 % | +0.00 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 | 139.00 register/thread | +0 | 134.00 register/thread | -5 |
| Waves/SM | 331.31 | 331.31 | +0.00 | 331.31 | +0.00 | 331.31 | +0.00 |
| No Eligible (%) | 52.70 % | 52.55 % | -0.15 pp | 52.72 % | +0.02 pp | 50.99 % | -1.71 pp |
| IPC Active | 1.89 inst/cycle | 1.90 inst/cycle | +0.01 | 1.89 inst/cycle | +0.00 | 1.96 inst/cycle | +0.07 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_weight_coalescing_via_correct_ic_kk_oc_permutation | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_weight_coales... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16230000 ns | 16240000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 48.88 % | 48.86 % | -0.02 pp |
| Memory Throughput (%) | 65.57 % | 65.55 % | -0.02 pp |
| DRAM Throughput (%) | 8.06 % | 8.06 % | +0.00 pp |
| Achieved Occupancy (%) | 18.68 % | 18.68 % | +0.00 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 |
| Waves/SM | 331.31 | 331.31 | +0.00 |
| No Eligible (%) | 52.70 % | 52.71 % | +0.01 pp |
| IPC Active | 1.89 inst/cycle | 1.89 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_streaming_x_reused_w | 0.998× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16230000 ns | 16190000 ns | -40000 ns (-0.2%) |
| SM Throughput (%) | 48.88 % | 49.00 % | +0.12 pp |
| Memory Throughput (%) | 65.57 % | 65.70 % | +0.13 pp |
| DRAM Throughput (%) | 8.06 % | 8.09 % | +0.03 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 |
| Waves/SM | 331.31 | 331.31 | +0.00 |
| No Eligible (%) | 52.70 % | 52.55 % | -0.15 pp |
| IPC Active | 1.89 inst/cycle | 1.90 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  16230000 ns  (1.000×)

Final best: 16260000 ns  (0.9982×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -6,15 +6,16 @@

 # Input: [batch, in_channels, L_in]
 # Output: [batch, out_channels, L_out] where L_out = L_in + kernel_size - 1 (stride=1, padding=0)
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
     ],
     key=['M', 'N', 'K'],
@@ -83,7 +84,6 @@

     mask_y = valid_m[:, None] & valid_n[None, :]
     y_ptrs = Y_ptr + pid_b * stride_yb + offs_n[None, :] * stride_yc + offs_m[:, None] * stride_yl
     tl.store(y_ptrs, acc.to(tl.float32), mask=mask_y)
-
 
 def kernel_function(weight, x):
     # weight: [in_channels, out_channels, kernel_size]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
