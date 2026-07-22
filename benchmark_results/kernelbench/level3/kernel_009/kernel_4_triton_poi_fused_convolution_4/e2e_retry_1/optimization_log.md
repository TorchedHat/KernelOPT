# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4670 ns — **Best latency**: 4670 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_4_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 1.45
- **Registers/thread**: 28
- **Duration**: 4670 ns
- **SM Throughput**: 33.42 % | **Memory**: 38.27 % | **DRAM**: 10.80 %
- **Achieved Occupancy**: 70.98 %
- **IPC Active**: 2.41 inst/cycle | **No Eligible**: 37.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.974× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4670 ns | 4830 ns | +160 ns (+3.4%) |
| SM Throughput (%) | 33.42 % | 34.85 % | +1.43 pp |
| Memory Throughput (%) | 38.27 % | 39.91 % | +1.64 pp |
| DRAM Throughput (%) | 10.80 % | 10.46 % | -0.34 pp |
| Achieved Occupancy (%) | 70.98 % | 71.59 % | +0.61 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 1.45 | 1.45 | +0.00 |
| No Eligible (%) | 37.30 % | 37.06 % | -0.24 pp |
| IPC Active | 2.41 inst/cycle | 2.42 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  4670 ns  (1.000×)

Final best: 4670 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
