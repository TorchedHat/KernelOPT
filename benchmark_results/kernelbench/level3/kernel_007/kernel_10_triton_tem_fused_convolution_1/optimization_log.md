# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27260 ns — **Best latency**: 27420 ns (**0.9942×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_10_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 128
- **Duration**: 27260 ns
- **SM Throughput**: 17.07 % | **Memory**: 66.33 % | **DRAM**: 4.57 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.996× | 0.994× | ✓ |  |
| B | tma_loads_with_autotune_to_fix_bank_conflicts_and_occupancy | 1.028× | 0.770× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_reduction | 0.995× | 0.742× | ✓ |  |
| D | add_autotune_reduce_register_pressure_increase_occupancy | 1.000× | 0.750× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | memory_coalescing... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27260 ns | 27420 ns | +160 ns (+0.6%) | 35390 ns | +8130 ns (+29.8%) | 36740 ns | +9480 ns (+34.8%) | 36350 ns | +9090 ns (+33.3%) |
| SM Throughput (%) | 17.07 % | 17.00 % | -0.07 pp | 12.66 % | -4.41 pp | 12.31 % | -4.76 pp | 12.21 % | -4.86 pp |
| Memory Throughput (%) | 66.33 % | 66.07 % | -0.26 pp | 29.22 % | -37.11 pp | 28.33 % | -38.00 pp | 28.11 % | -38.22 pp |
| DRAM Throughput (%) | 4.57 % | 4.54 % | -0.03 pp | 3.54 % | -1.03 pp | 3.41 % | -1.16 pp | 3.45 % | -1.12 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 252.00 register/thread | +124 | 255.00 register/thread | +127 | 255.00 register/thread | +127 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 79.76 % | 79.43 % | -0.33 pp | 69.30 % | -10.46 pp | 70.33 % | -9.43 pp | 70.25 % | -9.51 pp |
| IPC Active | 0.81 inst/cycle | 0.82 inst/cycle | +0.01 | 0.60 inst/cycle | -0.21 | 0.60 inst/cycle | -0.21 | 0.59 inst/cycle | -0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_correctness_then_autotune_tiles | 1.002× | 1.003× | ✓ |  |
| B | tma_loads_with_hoisted_block_ptr_to_fix_b_matrix_bank_conflicts | 0.920× | 0.905× | ✓ |  |
| D | fix_b_matrix_load_to_coalesced_2d_indexing | 1.001× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | tma_loads_with_ho... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27260 ns | 27170 ns | -90 ns (-0.3%) | 30110 ns | +2850 ns (+10.5%) | 27070 ns | -190 ns (-0.7%) |
| SM Throughput (%) | 17.07 % | 17.04 % | -0.03 pp | 15.47 % | -1.60 pp | 16.87 % | -0.20 pp |
| Memory Throughput (%) | 66.33 % | 66.25 % | -0.08 pp | 59.81 % | -6.52 pp | 65.56 % | -0.77 pp |
| DRAM Throughput (%) | 4.57 % | 4.58 % | +0.01 pp | 4.14 % | -0.43 pp | 4.60 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 79.76 % | 79.48 % | -0.28 pp | 81.72 % | +1.96 pp | 79.48 % | -0.28 pp |
| IPC Active | 0.81 inst/cycle | 0.82 inst/cycle | +0.01 | 0.73 inst/cycle | -0.08 | 0.81 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  27260 ns  (1.000×)

Final best: 27420 ns  (0.9942×)
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
 def triton_tem_fused_convolution_10(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -89,8 +90,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
