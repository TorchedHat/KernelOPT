# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4220 ns — **Best latency**: 4220 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2450,1,1), Waves/SM: 1.16
- **Registers/thread**: 22
- **Duration**: 4220 ns
- **SM Throughput**: 24.82 % | **Memory**: 13.19 % | **DRAM**: 12.22 %
- **Achieved Occupancy**: 62.47 %
- **IPC Active**: 1.89 inst/cycle | **No Eligible**: 54.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_tail | 0.970× | 0.909× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.973× | 0.868× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4220 ns | 4640 ns | +420 ns (+10.0%) | 4860 ns | +640 ns (+15.2%) |
| SM Throughput (%) | 24.82 % | 43.44 % | +18.62 pp | 42.95 % | +18.13 pp |
| Memory Throughput (%) | 13.19 % | 15.94 % | +2.75 pp | 15.76 % | +2.57 pp |
| DRAM Throughput (%) | 12.22 % | 11.09 % | -1.13 pp | 10.59 % | -1.63 pp |
| Achieved Occupancy (%) | 62.47 % | 72.14 % | +9.67 pp | 69.99 % | +7.52 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 54.19 % | 24.56 % | -29.63 pp | 24.46 % | -29.73 pp |
| IPC Active | 1.89 inst/cycle | 2.77 inst/cycle | +0.88 | 2.97 inst/cycle | +1.08 |

---

## Cumulative Progress

```
Baseline:  4220 ns  (1.000×)

Final best: 4220 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
