# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6750 ns — **Best latency**: 6750 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_33_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 20
- **Duration**: 6750 ns
- **SM Throughput**: 48.03 % | **Memory**: 14.46 % | **DRAM**: 9.20 %
- **Achieved Occupancy**: 72.32 %
- **IPC Active**: 2.68 inst/cycle | **No Eligible**: 32.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 0.998× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6750 ns | 6620 ns | -130 ns (-1.9%) |
| SM Throughput (%) | 48.03 % | 46.29 % | -1.74 pp |
| Memory Throughput (%) | 14.46 % | 14.46 % | +0.00 pp |
| DRAM Throughput (%) | 9.20 % | 9.40 % | +0.20 pp |
| Achieved Occupancy (%) | 72.32 % | 73.94 % | +1.62 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 |
| No Eligible (%) | 32.46 % | 31.00 % | -1.46 pp |
| IPC Active | 2.68 inst/cycle | 2.66 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  6750 ns  (1.000×)

Final best: 6750 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
