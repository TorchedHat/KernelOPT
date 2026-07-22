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
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285820 ns
- **SM Throughput**: 19.81 % | **Memory**: 78.90 % | **DRAM**: 72.22 %
- **Achieved Occupancy**: 94.07 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.999× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285820 ns | 285470 ns | -350 ns (-0.1%) |
| SM Throughput (%) | 19.81 % | 19.36 % | -0.45 pp |
| Memory Throughput (%) | 78.90 % | 79.01 % | +0.11 pp |
| DRAM Throughput (%) | 72.22 % | 72.29 % | +0.07 pp |
| Achieved Occupancy (%) | 94.07 % | 94.03 % | -0.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.13 % | 80.37 % | +0.24 pp |
| IPC Active | 0.78 inst/cycle | 0.80 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  285820 ns  (1.000×)

Final best: 285820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
