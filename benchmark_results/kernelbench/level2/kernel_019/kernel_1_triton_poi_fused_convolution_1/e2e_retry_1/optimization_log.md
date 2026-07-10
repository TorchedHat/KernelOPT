# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285820 ns — **Best latency**: 285820 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285820 ns
- **SM Throughput**: 19.45 % | **Memory**: 78.90 % | **DRAM**: 72.21 %
- **Achieved Occupancy**: 94.02 %
- **IPC Active**: 0.80 inst/cycle | **No Eligible**: 79.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285820 ns | 285730 ns | -90 ns (-0.0%) |
| SM Throughput (%) | 19.45 % | 19.44 % | -0.01 pp |
| Memory Throughput (%) | 78.90 % | 78.89 % | -0.01 pp |
| DRAM Throughput (%) | 72.21 % | 72.26 % | +0.05 pp |
| Achieved Occupancy (%) | 94.02 % | 94.03 % | +0.01 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.98 % | 80.48 % | +0.50 pp |
| IPC Active | 0.80 inst/cycle | 0.79 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  285820 ns  (1.000×)

Final best: 285820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
