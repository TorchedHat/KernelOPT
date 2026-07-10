# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286240 ns — **Best latency**: 286240 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286240 ns
- **SM Throughput**: 19.33 % | **Memory**: 78.73 % | **DRAM**: 72.09 %
- **Achieved Occupancy**: 94.06 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.999× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286240 ns | 285730 ns | -510 ns (-0.2%) |
| SM Throughput (%) | 19.33 % | 19.56 % | +0.23 pp |
| Memory Throughput (%) | 78.73 % | 78.84 % | +0.11 pp |
| DRAM Throughput (%) | 72.09 % | 72.25 % | +0.16 pp |
| Achieved Occupancy (%) | 94.06 % | 94.03 % | -0.03 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.17 % | 80.19 % | +0.02 pp |
| IPC Active | 0.78 inst/cycle | 0.79 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  286240 ns  (1.000×)

Final best: 286240 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
