# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_78` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 49730 ns — **Best latency**: 49730 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_78`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_78_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 49730 ns
- **SM Throughput**: 7.93 % | **Memory**: 53.20 % | **DRAM**: 0.90 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.484× | 0.468× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 49730 ns | 106370 ns | +56640 ns (+113.9%) |
| SM Throughput (%) | 7.93 % | 11.78 % | +3.85 pp |
| Memory Throughput (%) | 53.20 % | 46.01 % | -7.19 pp |
| DRAM Throughput (%) | 0.90 % | 0.47 % | -0.43 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.20 | +0.00 |
| No Eligible (%) | 86.34 % | 79.68 % | -6.66 pp |
| IPC Active | 0.55 inst/cycle | 0.81 inst/cycle | +0.26 |

---

## Cumulative Progress

```
Baseline:  49730 ns  (1.000×)

Final best: 49730 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
