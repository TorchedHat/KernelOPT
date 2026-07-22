# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 36540 ns — **Best latency**: 36540 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_31_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (90,1,1), Waves/SM: 0.23
- **Registers/thread**: 168
- **Duration**: 36540 ns
- **SM Throughput**: 17.71 % | **Memory**: 39.76 % | **DRAM**: 0.79 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 70.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.994× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 36540 ns | 36700 ns | +160 ns (+0.4%) |
| SM Throughput (%) | 17.71 % | 17.67 % | -0.04 pp |
| Memory Throughput (%) | 39.76 % | 39.67 % | -0.09 pp |
| DRAM Throughput (%) | 0.79 % | 0.79 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 70.64 % | 70.62 % | -0.02 pp |
| IPC Active | 1.17 inst/cycle | 1.18 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  36540 ns  (1.000×)

Final best: 36540 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
