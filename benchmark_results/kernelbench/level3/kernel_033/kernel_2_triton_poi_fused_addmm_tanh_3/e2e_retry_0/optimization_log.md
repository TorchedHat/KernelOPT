# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22720 ns — **Best latency**: 22720 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_tanh_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_2_triton_poi_fused_addmm_tanh_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22720 ns
- **SM Throughput**: 37.21 % | **Memory**: 20.06 % | **DRAM**: 15.11 %
- **Achieved Occupancy**: 45.76 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_for_persistent_tile_efficiency | 0.717× | 0.197× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.890× | 0.921× | ✓ |  |
| C | add_triton_autotune | 0.997× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | increase_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 115300 ns | +92580 ns (+407.5%) | 24670 ns | +1950 ns (+8.6%) | 22720 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.21 % | 93.07 % | +55.86 pp | 65.37 % | +28.16 pp | 39.62 % | +2.41 pp |
| Memory Throughput (%) | 20.06 % | 15.62 % | -4.44 pp | 17.89 % | -2.17 pp | 19.99 % | -0.07 pp |
| DRAM Throughput (%) | 15.11 % | 2.97 % | -12.14 pp | 13.92 % | -1.19 pp | 15.10 % | -0.01 pp |
| Achieved Occupancy (%) | 45.76 % | 85.18 % | +39.42 pp | 47.78 % | +2.02 pp | 45.60 % | -0.16 pp |
| Registers/Thread | 16.00 register/thread | 29.00 register/thread | +13 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 62.06 | +46.54 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.14 % | 5.13 % | -53.01 pp | 26.42 % | -31.72 pp | 55.10 % | -3.04 pp |
| IPC Active | 1.66 inst/cycle | 3.79 inst/cycle | +2.13 | 2.93 inst/cycle | +1.27 | 1.79 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  22720 ns  (1.000×)

Final best: 22720 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
