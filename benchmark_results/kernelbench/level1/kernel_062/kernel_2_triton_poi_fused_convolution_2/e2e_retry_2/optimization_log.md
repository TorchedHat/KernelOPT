# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174340 ns — **Best latency**: 174340 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_062/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174340 ns
- **SM Throughput**: 27.92 % | **Memory**: 81.82 % | **DRAM**: 67.36 %
- **Achieved Occupancy**: 94.16 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174340 ns | 174430 ns | +90 ns (+0.1%) |
| SM Throughput (%) | 27.92 % | 28.68 % | +0.76 pp |
| Memory Throughput (%) | 81.82 % | 81.62 % | -0.20 pp |
| DRAM Throughput (%) | 67.36 % | 67.33 % | -0.03 pp |
| Achieved Occupancy (%) | 94.16 % | 94.20 % | +0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 70.91 % | 71.81 % | +0.90 pp |
| IPC Active | 1.16 inst/cycle | 1.14 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  174340 ns  (1.000×)

Final best: 174340 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
