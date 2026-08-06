# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 3040 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (784,1,1), Waves/SM: 0.37
- **Registers/thread**: 18
- **Duration**: 3040 ns
- **SM Throughput**: 10.90 % | **Memory**: 4.17 % | **DRAM**: 2.77 %
- **Achieved Occupancy**: 30.51 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 68.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.992× | 0.629× | ✓ |  |
| B | autotune_xblock_and_warps_for_occupancy | 0.991× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 4830 ns | +1790 ns (+58.9%) | 3360 ns | +320 ns (+10.5%) |
| SM Throughput (%) | 10.90 % | 45.24 % | +34.34 pp | 7.75 % | -3.15 pp |
| Memory Throughput (%) | 4.17 % | 9.47 % | +5.30 pp | 3.93 % | -0.24 pp |
| DRAM Throughput (%) | 2.77 % | 1.76 % | -1.01 pp | 2.53 % | -0.24 pp |
| Achieved Occupancy (%) | 30.51 % | 46.48 % | +15.97 pp | 15.84 % | -14.67 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 24.00 register/thread | +6 |
| Waves/SM | 0.37 | 1.19 | +0.82 | 0.19 | -0.18 |
| No Eligible (%) | 68.78 % | 23.52 % | -45.26 pp | 77.55 % | +8.77 pp |
| IPC Active | 1.13 inst/cycle | 3.01 inst/cycle | +1.88 | 0.80 inst/cycle | -0.33 |

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)

Final best: 3040 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
