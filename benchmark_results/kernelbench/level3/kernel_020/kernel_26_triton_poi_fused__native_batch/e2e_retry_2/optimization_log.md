# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7200 ns — **Best latency**: 7200 ns (**1.0000×** vs baseline)
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
- **Duration**: 7200 ns
- **SM Throughput**: 28.46 % | **Memory**: 17.10 % | **DRAM**: 10.74 %
- **Achieved Occupancy**: 49.19 %
- **IPC Active**: 1.76 inst/cycle | **No Eligible**: 56.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_latency_hiding | 0.976× | 1.023× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_improve_occupancy_and_hide_l1tex_latency | 0.963× | 1.014× | ✓ |  |
| C | add_triton_autotune | 0.972× | 1.023× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7200 ns | 7040 ns | -160 ns (-2.2%) | 7100 ns | -100 ns (-1.4%) | 7040 ns | -160 ns (-2.2%) |
| SM Throughput (%) | 28.46 % | 28.49 % | +0.03 pp | 30.65 % | +2.19 pp | 27.56 % | -0.90 pp |
| Memory Throughput (%) | 17.10 % | 17.16 % | +0.06 pp | 21.99 % | +4.89 pp | 16.66 % | -0.44 pp |
| DRAM Throughput (%) | 10.74 % | 10.98 % | +0.24 pp | 10.83 % | +0.09 pp | 10.98 % | +0.24 pp |
| Achieved Occupancy (%) | 49.19 % | 49.38 % | +0.19 pp | 43.51 % | -5.68 pp | 48.82 % | -0.37 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 56.61 % | 55.95 % | -0.66 pp | 49.54 % | -7.07 pp | 56.75 % | +0.14 pp |
| IPC Active | 1.76 inst/cycle | 1.76 inst/cycle | +0.00 | 2.11 inst/cycle | +0.35 | 1.74 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  7200 ns  (1.000×)

Final best: 7200 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
