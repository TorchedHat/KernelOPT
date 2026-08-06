# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7070 ns — **Best latency**: 7070 ns (**1.0000×** vs baseline)
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
- **Duration**: 7070 ns
- **SM Throughput**: 42.24 % | **Memory**: 16.18 % | **DRAM**: 9.75 %
- **Achieved Occupancy**: 63.16 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 38.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.999× | 1.047× | ✓ |  |
| B | autotune_xblock_tile_size | 0.940× | 1.000× | ✓ |  |
| C | add_triton_autotune | 0.980× | 1.047× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7070 ns | 6750 ns | -320 ns (-4.5%) | 7070 ns | +0 ns (+0.0%) | 6750 ns | -320 ns (-4.5%) |
| SM Throughput (%) | 42.24 % | 42.17 % | -0.07 pp | 51.82 % | +9.58 pp | 40.96 % | -1.28 pp |
| Memory Throughput (%) | 16.18 % | 16.12 % | -0.06 pp | 19.39 % | +3.21 pp | 15.66 % | -0.52 pp |
| DRAM Throughput (%) | 9.75 % | 10.16 % | +0.41 pp | 9.69 % | -0.06 pp | 10.17 % | +0.42 pp |
| Achieved Occupancy (%) | 63.16 % | 62.15 % | -1.01 pp | 70.92 % | +7.76 pp | 62.04 % | -1.12 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 22.00 register/thread | +0 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 | 3.07 | +0.00 |
| No Eligible (%) | 38.78 % | 36.42 % | -2.36 pp | 24.99 % | -13.79 pp | 36.15 % | -2.63 pp |
| IPC Active | 2.48 inst/cycle | 2.54 inst/cycle | +0.06 | 3.10 inst/cycle | +0.62 | 2.54 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  7070 ns  (1.000×)

Final best: 7070 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
