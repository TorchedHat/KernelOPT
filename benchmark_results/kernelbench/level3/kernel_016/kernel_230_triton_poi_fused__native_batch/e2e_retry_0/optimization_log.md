# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7010 ns — **Best latency**: 7010 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_230_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7105,1,1), Waves/SM: 3.36
- **Registers/thread**: 22
- **Duration**: 7010 ns
- **SM Throughput**: 43.91 % | **Memory**: 16.82 % | **DRAM**: 10.68 %
- **Achieved Occupancy**: 62.25 %
- **IPC Active**: 2.58 inst/cycle | **No Eligible**: 34.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_eliminate_partial_wave | 0.902× | 0.769× | ✓ |  |
| B | autotune_xblock_tile_size | 0.977× | 0.987× | ✓ |  |
| C | add_triton_autotune | 0.998× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7010 ns | 9120 ns | +2110 ns (+30.1%) | 7100 ns | +90 ns (+1.3%) | 7040 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 43.91 % | 63.61 % | +19.70 pp | 43.98 % | +0.07 pp | 43.32 % | -0.59 pp |
| Memory Throughput (%) | 16.82 % | 17.17 % | +0.35 pp | 16.63 % | -0.19 pp | 16.59 % | -0.23 pp |
| DRAM Throughput (%) | 10.68 % | 8.22 % | -2.46 pp | 10.62 % | -0.06 pp | 10.67 % | -0.01 pp |
| Achieved Occupancy (%) | 62.25 % | 80.78 % | +18.53 pp | 64.93 % | +2.68 pp | 62.02 % | -0.23 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 34.41 % | 18.77 % | -15.64 pp | 39.38 % | +4.97 pp | 39.18 % | +4.77 pp |
| IPC Active | 2.58 inst/cycle | 3.24 inst/cycle | +0.66 | 2.59 inst/cycle | +0.01 | 2.56 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  7010 ns  (1.000×)

Final best: 7010 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
