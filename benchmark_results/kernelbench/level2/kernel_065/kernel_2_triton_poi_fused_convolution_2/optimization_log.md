# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174370 ns — **Best latency**: 174370 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174370 ns
- **SM Throughput**: 27.92 % | **Memory**: 81.76 % | **DRAM**: 67.36 %
- **Achieved Occupancy**: 94.20 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_transpose_tile_for_coalesced_io | 0.820× | 0.829× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174370 ns | 210400 ns | +36030 ns (+20.7%) |
| SM Throughput (%) | 27.92 % | 22.52 % | -5.40 pp |
| Memory Throughput (%) | 81.76 % | 69.96 % | -11.80 pp |
| DRAM Throughput (%) | 67.36 % | 60.23 % | -7.13 pp |
| Achieved Occupancy (%) | 94.20 % | 94.37 % | +0.17 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.08 % | 77.23 % | +6.15 pp |
| IPC Active | 1.16 inst/cycle | 0.93 inst/cycle | -0.23 |

---

## Cumulative Progress

```
Baseline:  174370 ns  (1.000×)

Final best: 174370 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
