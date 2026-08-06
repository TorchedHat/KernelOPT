# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3410000 ns — **Best latency**: 3410000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_5_triton_tem_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 14.85
- **Registers/thread**: 255
- **Duration**: 3410000 ns
- **SM Throughput**: 40.74 % | **Memory**: 59.51 % | **DRAM**: 7.09 %
- **Achieved Occupancy**: 12.44 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3410000 ns | 3390000 ns | -20000 ns (-0.6%) |
| SM Throughput (%) | 40.74 % | 40.71 % | -0.03 pp |
| Memory Throughput (%) | 59.51 % | 59.46 % | -0.05 pp |
| DRAM Throughput (%) | 7.09 % | 7.12 % | +0.03 pp |
| Achieved Occupancy (%) | 12.44 % | 12.42 % | -0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 |
| No Eligible (%) | 58.51 % | 58.53 % | +0.02 pp |
| IPC Active | 1.66 inst/cycle | 1.65 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  3410000 ns  (1.000×)

Final best: 3410000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
