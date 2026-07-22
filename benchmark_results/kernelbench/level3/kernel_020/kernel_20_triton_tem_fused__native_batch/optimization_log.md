# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31620 ns — **Best latency**: 31620 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 31620 ns
- **SM Throughput**: 19.38 % | **Memory**: 74.13 % | **DRAM**: 4.89 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.91 inst/cycle | **No Eligible**: 77.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.992× | 1.003× | ✓ |  |
| C | eviction_policies_streaming_loads | 0.990× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 31620 ns | 31520 ns | -100 ns (-0.3%) | 31620 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 19.38 % | 19.45 % | +0.07 pp | 19.46 % | +0.08 pp |
| Memory Throughput (%) | 74.13 % | 74.39 % | +0.26 pp | 74.41 % | +0.28 pp |
| DRAM Throughput (%) | 4.89 % | 4.91 % | +0.02 pp | 4.89 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.40 % | 77.29 % | -0.11 pp | 77.29 % | -0.11 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_replace_scalar_pointer_arithmetic | 0.998× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_replace... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 31620 ns | 31260 ns | -360 ns (-1.1%) |
| SM Throughput (%) | 19.38 % | 19.47 % | +0.09 pp |
| Memory Throughput (%) | 74.13 % | 75.23 % | +1.10 pp |
| DRAM Throughput (%) | 4.89 % | 4.95 % | +0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.40 % | 77.32 % | -0.08 pp |
| IPC Active | 0.91 inst/cycle | 0.90 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  31620 ns  (1.000×)

Final best: 31620 ns  (1.0000×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
@@ -90,8 +91,6 @@

     xindex = idx_n + 40*idx_m
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
