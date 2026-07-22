# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23740 ns — **Best latency**: 23740 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_15_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 23740 ns
- **SM Throughput**: 10.61 % | **Memory**: 74.17 % | **DRAM**: 5.20 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 23740 ns | 23740 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 10.61 % | 10.56 % | -0.05 pp |
| Memory Throughput (%) | 74.17 % | 73.79 % | -0.38 pp |
| DRAM Throughput (%) | 5.20 % | 5.21 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 |
| No Eligible (%) | 87.44 % | 87.44 % | +0.00 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  23740 ns  (1.000×)

Final best: 23740 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
