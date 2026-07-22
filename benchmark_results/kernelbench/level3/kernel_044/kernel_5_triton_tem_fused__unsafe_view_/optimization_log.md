# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8030 ns — **Best latency**: 8030 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_5_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.12
- **Registers/thread**: 124
- **Duration**: 8030 ns
- **SM Throughput**: 15.04 % | **Memory**: 15.10 % | **DRAM**: 1.07 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 57.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8030 ns | 8130 ns | +100 ns (+1.2%) |
| SM Throughput (%) | 15.04 % | 15.25 % | +0.21 pp |
| Memory Throughput (%) | 15.10 % | 15.32 % | +0.22 pp |
| DRAM Throughput (%) | 1.07 % | 1.05 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 |
| No Eligible (%) | 57.31 % | 56.15 % | -1.16 pp |
| IPC Active | 1.68 inst/cycle | 1.73 inst/cycle | +0.05 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_batch_dimension_minimal_surgical | 0.996× | 0.984× | ✓ |  |
| D | fix_grid_launch_batch_dimension_only | 0.987× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_batch_di... | Δ | fix_grid_launch_b... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8030 ns | 8160 ns | +130 ns (+1.6%) | 8030 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 15.04 % | 14.94 % | -0.10 pp | 14.95 % | -0.09 pp |
| Memory Throughput (%) | 15.10 % | 15.01 % | -0.09 pp | 15.02 % | -0.08 pp |
| DRAM Throughput (%) | 1.07 % | 1.05 % | -0.02 pp | 1.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 57.31 % | 56.42 % | -0.89 pp | 56.28 % | -1.03 pp |
| IPC Active | 1.68 inst/cycle | 1.71 inst/cycle | +0.03 | 1.68 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  8030 ns  (1.000×)

Final best: 8030 ns  (1.0000×)
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
 def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_5(arg_A, arg_B, out_ptr0):
@@ -93,7 +94,6 @@

     xindex = idx_n + 512*idx_m + 262144*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
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
