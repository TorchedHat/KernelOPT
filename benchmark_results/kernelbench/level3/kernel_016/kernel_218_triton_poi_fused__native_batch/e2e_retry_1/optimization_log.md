# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6660 ns — **Best latency**: 6660 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_218_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6493,1,1), Waves/SM: 3.07
- **Registers/thread**: 22
- **Duration**: 6660 ns
- **SM Throughput**: 41.29 % | **Memory**: 15.81 % | **DRAM**: 10.29 %
- **Achieved Occupancy**: 61.10 %
- **IPC Active**: 2.52 inst/cycle | **No Eligible**: 36.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.968× | 0.938× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.934× | 0.795× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6660 ns | 7100 ns | +440 ns (+6.6%) | 8380 ns | +1720 ns (+25.8%) |
| SM Throughput (%) | 41.29 % | 51.35 % | +10.06 pp | 63.30 % | +22.01 pp |
| Memory Throughput (%) | 15.81 % | 19.05 % | +3.24 pp | 17.26 % | +1.45 pp |
| DRAM Throughput (%) | 10.29 % | 9.67 % | -0.62 pp | 8.17 % | -2.12 pp |
| Achieved Occupancy (%) | 61.10 % | 72.46 % | +11.36 pp | 79.50 % | +18.40 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 32.00 register/thread | +10 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 |
| No Eligible (%) | 36.79 % | 21.17 % | -15.62 pp | 16.94 % | -19.85 pp |
| IPC Active | 2.52 inst/cycle | 3.02 inst/cycle | +0.50 | 3.31 inst/cycle | +0.79 |

---

## Cumulative Progress

```
Baseline:  6660 ns  (1.000×)

Final best: 6660 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
