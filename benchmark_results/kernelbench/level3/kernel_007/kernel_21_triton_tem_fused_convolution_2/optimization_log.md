# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 47360 ns — **Best latency**: 83840 ns (**0.5649×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_21_triton_tem_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 156
- **Duration**: 47360 ns
- **SM Throughput**: 24.90 % | **Memory**: 56.41 % | **DRAM**: 3.53 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.21 inst/cycle | **No Eligible**: 69.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 0.565× | ✓ | **✓ best** |
| B | tma_loads_with_autotune_to_fix_bank_conflicts_and_occupancy | 0.994× | 0.984× | ✓ |  |
| D | autotune_tile_sizes_reduce_register_pressure | 1.005× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 47360 ns | 83840 ns | +36480 ns (+77.0%) | 48130 ns | +770 ns (+1.6%) | 47100 ns | -260 ns (-0.5%) |
| SM Throughput (%) | 24.90 % | 13.85 % | -11.05 pp | 24.61 % | -0.29 pp | 24.66 % | -0.24 pp |
| Memory Throughput (%) | 56.41 % | 20.34 % | -36.07 pp | 55.43 % | -0.98 pp | 55.87 % | -0.54 pp |
| DRAM Throughput (%) | 3.53 % | 2.00 % | -1.53 pp | 3.47 % | -0.06 pp | 3.55 % | +0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 156.00 register/thread | 255.00 register/thread | +99 | 168.00 register/thread | +12 | 156.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.23 | -0.08 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 69.76 % | 65.94 % | -3.82 pp | 70.05 % | +0.29 pp | 69.98 % | +0.22 pp |
| IPC Active | 1.21 inst/cycle | 0.68 inst/cycle | -0.53 | 1.20 inst/cycle | -0.01 | 1.20 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_index_to_eliminate_bank_conflicts | 0.969× | 0.964× | ✓ |  |
| B | fix_b_matrix_shared_memory_layout_to_eliminate_bank_conflicts | 0.994× | 0.980× | ✓ |  |
| D | add_triton_autotune | 1.000× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_shar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 47360 ns | 49120 ns | +1760 ns (+3.7%) | 48350 ns | +990 ns (+2.1%) | 47550 ns | +190 ns (+0.4%) |
| SM Throughput (%) | 24.90 % | 24.07 % | -0.83 pp | 24.56 % | -0.34 pp | 24.67 % | -0.23 pp |
| Memory Throughput (%) | 56.41 % | 54.54 % | -1.87 pp | 55.30 % | -1.11 pp | 55.89 % | -0.52 pp |
| DRAM Throughput (%) | 3.53 % | 3.40 % | -0.13 pp | 3.46 % | -0.07 pp | 3.52 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 156.00 register/thread | 168.00 register/thread | +12 | 168.00 register/thread | +12 | 156.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 69.76 % | 70.93 % | +1.17 pp | 69.92 % | +0.16 pp | 69.79 % | +0.03 pp |
| IPC Active | 1.21 inst/cycle | 1.16 inst/cycle | -0.05 | 1.20 inst/cycle | -0.01 | 1.21 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_global_load_to_eliminate_bank_conflicts_correct_index | 0.979× | 0.971× | ✓ |  |
| B | fix_b_matrix_block_ptr_order_to_eliminate_bank_conflicts | 0.999× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_glob... | Δ | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 47360 ns | 48770 ns | +1410 ns (+3.0%) | 47940 ns | +580 ns (+1.2%) |
| SM Throughput (%) | 24.90 % | 24.03 % | -0.87 pp | 24.73 % | -0.17 pp |
| Memory Throughput (%) | 56.41 % | 54.46 % | -1.95 pp | 55.69 % | -0.72 pp |
| DRAM Throughput (%) | 3.53 % | 3.43 % | -0.10 pp | 3.49 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 156.00 register/thread | 168.00 register/thread | +12 | 168.00 register/thread | +12 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 69.76 % | 70.96 % | +1.20 pp | 70.07 % | +0.31 pp |
| IPC Active | 1.21 inst/cycle | 1.16 inst/cycle | -0.05 | 1.19 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  47360 ns  (1.000×)

Final best: 83840 ns  (0.5649×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_21(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -89,8 +99,6 @@

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
