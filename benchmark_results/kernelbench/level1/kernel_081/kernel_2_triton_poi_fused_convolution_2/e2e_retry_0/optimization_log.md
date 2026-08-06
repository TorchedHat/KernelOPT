# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 175550 ns — **Best latency**: 175550 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 175550 ns
- **SM Throughput**: 28.07 % | **Memory**: 80.95 % | **DRAM**: 66.91 %
- **Achieved Occupancy**: 94.19 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.999× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 175550 ns | 174110 ns | -1440 ns (-0.8%) |
| SM Throughput (%) | 28.07 % | 27.61 % | -0.46 pp |
| Memory Throughput (%) | 80.95 % | 81.64 % | +0.69 pp |
| DRAM Throughput (%) | 66.91 % | 67.49 % | +0.58 pp |
| Achieved Occupancy (%) | 94.19 % | 94.18 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 70.95 % | 70.97 % | +0.02 pp |
| IPC Active | 1.16 inst/cycle | 1.16 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  175550 ns  (1.000×)

Final best: 175550 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
