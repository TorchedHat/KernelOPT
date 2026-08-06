# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4450 ns — **Best latency**: 4450 ns (**1.0000×** vs baseline)
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
- **Duration**: 4450 ns
- **SM Throughput**: 25.06 % | **Memory**: 13.30 % | **DRAM**: 11.57 %
- **Achieved Occupancy**: 65.25 %
- **IPC Active**: 1.85 inst/cycle | **No Eligible**: 48.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 0.926× | 0.947× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 0.920× | 0.953× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4450 ns | 4700 ns | +250 ns (+5.6%) | 4670 ns | +220 ns (+4.9%) |
| SM Throughput (%) | 25.06 % | 43.18 % | +18.12 pp | 43.33 % | +18.27 pp |
| Memory Throughput (%) | 13.30 % | 15.85 % | +2.55 pp | 15.90 % | +2.60 pp |
| DRAM Throughput (%) | 11.57 % | 11.03 % | -0.54 pp | 11.03 % | -0.54 pp |
| Achieved Occupancy (%) | 65.25 % | 70.03 % | +4.78 pp | 72.23 % | +6.98 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 48.10 % | 28.86 % | -19.24 pp | 28.76 % | -19.34 pp |
| IPC Active | 1.85 inst/cycle | 2.97 inst/cycle | +1.12 | 2.76 inst/cycle | +0.91 |

---

## Cumulative Progress

```
Baseline:  4450 ns  (1.000×)

Final best: 4450 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
