# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176130 ns — **Best latency**: 176130 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_062/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 176130 ns
- **SM Throughput**: 28.60 % | **Memory**: 80.96 % | **DRAM**: 66.67 %
- **Achieved Occupancy**: 94.23 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.999× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 176130 ns | 174370 ns | -1760 ns (-1.0%) |
| SM Throughput (%) | 28.60 % | 27.95 % | -0.65 pp |
| Memory Throughput (%) | 80.96 % | 81.76 % | +0.80 pp |
| DRAM Throughput (%) | 66.67 % | 67.36 % | +0.69 pp |
| Achieved Occupancy (%) | 94.23 % | 94.27 % | +0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.00 % | 70.84 % | -0.16 pp |
| IPC Active | 1.12 inst/cycle | 1.16 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  176130 ns  (1.000×)

Final best: 176130 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
