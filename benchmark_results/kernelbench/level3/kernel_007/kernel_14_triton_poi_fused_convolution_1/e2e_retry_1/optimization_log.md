# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2400 ns — **Best latency**: 2400 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_14_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2400 ns
- **SM Throughput**: 1.38 % | **Memory**: 1.32 % | **DRAM**: 0.48 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 93.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 0.999× | 1.057× | ✓ |  |
| B | autotune_tile_sizes_for_grid_coverage | 0.984× | 0.938× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2270 ns | -130 ns (-5.4%) | 2560 ns | +160 ns (+6.7%) |
| SM Throughput (%) | 1.38 % | 0.20 % | -1.18 pp | 0.43 % | -0.95 pp |
| Memory Throughput (%) | 1.32 % | 0.48 % | -0.84 pp | 0.61 % | -0.71 pp |
| DRAM Throughput (%) | 0.48 % | 0.07 % | -0.41 pp | 0.30 % | -0.18 pp |
| Achieved Occupancy (%) | 6.10 % | 1.56 % | -4.54 pp | 5.95 % | -0.15 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.03 | +0.00 |
| No Eligible (%) | 93.15 % | 96.84 % | +3.69 pp | 91.16 % | -1.99 pp |
| IPC Active | 0.37 inst/cycle | 0.03 inst/cycle | -0.34 | 0.27 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  2400 ns  (1.000×)

Final best: 2400 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
