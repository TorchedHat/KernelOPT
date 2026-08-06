# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7010 ns — **Best latency**: 7010 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_26_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 18
- **Duration**: 7010 ns
- **SM Throughput**: 28.19 % | **Memory**: 17.01 % | **DRAM**: 11.02 %
- **Achieved Occupancy**: 48.54 %
- **IPC Active**: 1.72 inst/cycle | **No Eligible**: 57.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_and_improve_cache_reuse | 0.960× | 0.932× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.961× | 0.932× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7010 ns | 7520 ns | +510 ns (+7.3%) | 7520 ns | +510 ns (+7.3%) |
| SM Throughput (%) | 28.19 % | 37.12 % | +8.93 pp | 36.77 % | +8.58 pp |
| Memory Throughput (%) | 17.01 % | 23.88 % | +6.87 pp | 23.56 % | +6.55 pp |
| DRAM Throughput (%) | 11.02 % | 10.29 % | -0.73 pp | 10.24 % | -0.78 pp |
| Achieved Occupancy (%) | 48.54 % | 45.53 % | -3.01 pp | 45.13 % | -3.41 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 57.01 % | 40.37 % | -16.64 pp | 40.79 % | -16.22 pp |
| IPC Active | 1.72 inst/cycle | 2.36 inst/cycle | +0.64 | 2.35 inst/cycle | +0.63 |

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
