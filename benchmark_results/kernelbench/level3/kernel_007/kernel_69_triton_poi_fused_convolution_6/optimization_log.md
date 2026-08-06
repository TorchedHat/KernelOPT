# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_69` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2820 ns — **Best latency**: 2820 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_69`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_69_triton_poi_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 28
- **Duration**: 2820 ns
- **SM Throughput**: 9.44 % | **Memory**: 10.50 % | **DRAM**: 3.03 %
- **Achieved Occupancy**: 20.93 %
- **IPC Active**: 0.99 inst/cycle | **No Eligible**: 74.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.016× | 0.008× | ✓ |  |
| B | fix_tile_sizes_for_grid_parallelism_manual | 0.956× | 0.937× | ✓ |  |
| C | fix_tile_sizes_to_fill_sm_grid | 0.958× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_tile_sizes_fo... | Δ | fix_tile_sizes_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2820 ns | 355940 ns | +353120 ns (+12522.0%) | 3010 ns | +190 ns (+6.7%) | 2850 ns | +30 ns (+1.1%) |
| SM Throughput (%) | 9.44 % | 27.46 % | +18.02 pp | 9.10 % | -0.34 pp | 9.10 % | -0.34 pp |
| Memory Throughput (%) | 10.50 % | 39.34 % | +28.84 pp | 10.13 % | -0.37 pp | 10.07 % | -0.43 pp |
| DRAM Throughput (%) | 3.03 % | 0.03 % | -3.00 pp | 2.84 % | -0.19 pp | 2.99 % | -0.04 pp |
| Achieved Occupancy (%) | 20.93 % | 24.24 % | +3.31 pp | 21.38 % | +0.45 pp | 20.96 % | +0.03 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 74.74 % | 70.31 % | -4.43 pp | 72.13 % | -2.61 pp | 73.50 % | -1.24 pp |
| IPC Active | 0.99 inst/cycle | 1.15 inst/cycle | +0.16 | 0.82 inst/cycle | -0.17 | 1.04 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  2820 ns  (1.000×)

Final best: 2820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
