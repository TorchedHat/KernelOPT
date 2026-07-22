# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_38` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 36580 ns — **Best latency**: 36580 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_38`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_38_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (72,1,1), Waves/SM: 0.18
- **Registers/thread**: 168
- **Duration**: 36580 ns
- **SM Throughput**: 14.24 % | **Memory**: 31.84 % | **DRAM**: 0.73 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 70.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.992× | 1.002× | ✓ |  |
| B | reduce_register_pressure_via_autotune_and_maxnreg | 0.840× | 0.833× | ✓ |  |
| C | eviction_policies_streaming_loads | 0.993× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 36580 ns | 36510 ns | -70 ns (-0.2%) | 43940 ns | +7360 ns (+20.1%) | 36540 ns | -40 ns (-0.1%) |
| SM Throughput (%) | 14.24 % | 14.23 % | -0.01 pp | 13.14 % | -1.10 pp | 14.28 % | +0.04 pp |
| Memory Throughput (%) | 31.84 % | 31.80 % | -0.04 pp | 28.21 % | -3.63 pp | 31.92 % | +0.08 pp |
| DRAM Throughput (%) | 0.73 % | 0.73 % | +0.00 pp | 0.61 % | -0.12 pp | 0.73 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.14 | -0.04 | 0.18 | +0.00 |
| No Eligible (%) | 70.53 % | 70.44 % | -0.09 pp | 72.16 % | +1.63 pp | 70.48 % | -0.05 pp |
| IPC Active | 1.18 inst/cycle | 1.18 inst/cycle | +0.00 | 1.11 inst/cycle | -0.07 | 1.18 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_smem_staging_with_padding | 0.993× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 36580 ns | 36510 ns | -70 ns (-0.2%) |
| SM Throughput (%) | 14.24 % | 14.26 % | +0.02 pp |
| Memory Throughput (%) | 31.84 % | 31.87 % | +0.03 pp |
| DRAM Throughput (%) | 0.73 % | 0.73 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 |
| No Eligible (%) | 70.53 % | 70.50 % | -0.03 pp |
| IPC Active | 1.18 inst/cycle | 1.18 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  36580 ns  (1.000×)

Final best: 36580 ns  (1.0000×)
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
 def triton_tem_fused_convolution_38(arg_A, arg_B, out_ptr0):
@@ -87,8 +88,6 @@

     xindex = idx_n + 1152*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
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
