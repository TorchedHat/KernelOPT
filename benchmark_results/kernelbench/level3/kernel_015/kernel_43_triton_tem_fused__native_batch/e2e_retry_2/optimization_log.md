# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 133440 ns — **Best latency**: 133440 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_43`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_43_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.62
- **Registers/thread**: 168
- **Duration**: 133440 ns
- **SM Throughput**: 34.55 % | **Memory**: 78.06 % | **DRAM**: 2.56 %
- **Achieved Occupancy**: 11.94 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 62.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 133440 ns | 133310 ns | -130 ns (-0.1%) |
| SM Throughput (%) | 34.55 % | 34.62 % | +0.07 pp |
| Memory Throughput (%) | 78.06 % | 78.20 % | +0.14 pp |
| DRAM Throughput (%) | 2.56 % | 2.57 % | +0.01 pp |
| Achieved Occupancy (%) | 11.94 % | 11.94 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 |
| No Eligible (%) | 62.56 % | 62.58 % | +0.02 pp |
| IPC Active | 1.50 inst/cycle | 1.50 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  133440 ns  (1.000×)

Final best: 133440 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
