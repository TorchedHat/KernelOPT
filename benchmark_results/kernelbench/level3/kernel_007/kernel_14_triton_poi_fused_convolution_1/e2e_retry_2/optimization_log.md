# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2590 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_14`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_14_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2590 ns
- **SM Throughput**: 1.37 % | **Memory**: 1.31 % | **DRAM**: 0.44 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 89.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 0.978× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy | 0.982× | 1.066× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2590 ns | +0 ns (+0.0%) | 2430 ns | -160 ns (-6.2%) |
| SM Throughput (%) | 1.37 % | 0.45 % | -0.92 pp | 0.97 % | -0.40 pp |
| Memory Throughput (%) | 1.31 % | 0.58 % | -0.73 pp | 0.98 % | -0.33 pp |
| DRAM Throughput (%) | 0.44 % | 0.31 % | -0.13 pp | 0.47 % | +0.03 pp |
| Achieved Occupancy (%) | 6.10 % | 1.56 % | -4.54 pp | 6.19 % | +0.09 pp |
| Registers/Thread | 28.00 register/thread | 44.00 register/thread | +16 | 20.00 register/thread | -8 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.03 | +0.00 |
| No Eligible (%) | 89.70 % | 74.64 % | -15.06 pp | 93.02 % | +3.32 pp |
| IPC Active | 0.37 inst/cycle | 0.24 inst/cycle | -0.13 | 0.35 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)

Final best: 2590 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
