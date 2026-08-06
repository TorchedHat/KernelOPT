# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_59` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2560 ns — **Best latency**: 2560 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_59`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_59_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 28
- **Duration**: 2560 ns
- **SM Throughput**: 5.02 % | **Memory**: 5.45 % | **DRAM**: 1.69 %
- **Achieved Occupancy**: 10.89 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 85.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.992× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2560 ns | 2690 ns | +130 ns (+5.1%) |
| SM Throughput (%) | 5.02 % | 1.77 % | -3.25 pp |
| Memory Throughput (%) | 5.45 % | 2.70 % | -2.75 pp |
| DRAM Throughput (%) | 1.69 % | 1.62 % | -0.07 pp |
| Achieved Occupancy (%) | 10.89 % | 8.10 % | -2.79 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 |
| Waves/SM | 0.12 | 0.12 | +0.00 |
| No Eligible (%) | 85.23 % | 91.35 % | +6.12 pp |
| IPC Active | 0.51 inst/cycle | 0.30 inst/cycle | -0.21 |

---

## Cumulative Progress

```
Baseline:  2560 ns  (1.000×)

Final best: 2560 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
