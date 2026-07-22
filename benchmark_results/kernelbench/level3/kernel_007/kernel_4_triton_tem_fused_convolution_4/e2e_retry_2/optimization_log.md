# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1710000 ns — **Best latency**: 1710000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_4_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 9.90
- **Registers/thread**: 168
- **Duration**: 1710000 ns
- **SM Throughput**: 40.41 % | **Memory**: 91.43 % | **DRAM**: 12.93 %
- **Achieved Occupancy**: 18.40 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1710000 ns | 1720000 ns | +10000 ns (+0.6%) |
| SM Throughput (%) | 40.41 % | 40.42 % | +0.01 pp |
| Memory Throughput (%) | 91.43 % | 91.45 % | +0.02 pp |
| DRAM Throughput (%) | 12.93 % | 12.81 % | -0.12 pp |
| Achieved Occupancy (%) | 18.40 % | 18.41 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 |
| No Eligible (%) | 58.95 % | 58.96 % | +0.01 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1710000 ns  (1.000×)

Final best: 1710000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
