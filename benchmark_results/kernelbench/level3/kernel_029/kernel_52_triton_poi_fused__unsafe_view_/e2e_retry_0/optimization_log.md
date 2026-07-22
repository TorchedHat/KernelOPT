# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4380 ns — **Best latency**: 4380 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_52_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 1.39
- **Registers/thread**: 18
- **Duration**: 4380 ns
- **SM Throughput**: 25.19 % | **Memory**: 11.28 % | **DRAM**: 7.08 %
- **Achieved Occupancy**: 62.52 %
- **IPC Active**: 2.00 inst/cycle | **No Eligible**: 44.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_larger_xblock_autotune | 0.977× | 0.965× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.964× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4380 ns | 4540 ns | +160 ns (+3.7%) | 4480 ns | +100 ns (+2.3%) |
| SM Throughput (%) | 25.19 % | 29.83 % | +4.64 pp | 39.55 % | +14.36 pp |
| Memory Throughput (%) | 11.28 % | 15.14 % | +3.86 pp | 21.40 % | +10.12 pp |
| DRAM Throughput (%) | 7.08 % | 6.84 % | -0.24 pp | 6.97 % | -0.11 pp |
| Achieved Occupancy (%) | 62.52 % | 67.80 % | +5.28 pp | 66.73 % | +4.21 pp |
| Registers/Thread | 18.00 register/thread | 23.00 register/thread | +5 | 31.00 register/thread | +13 |
| Waves/SM | 1.39 | 1.39 | +0.00 | 1.39 | +0.00 |
| No Eligible (%) | 44.40 % | 31.36 % | -13.04 pp | 21.54 % | -22.86 pp |
| IPC Active | 2.00 inst/cycle | 2.67 inst/cycle | +0.67 | 3.08 inst/cycle | +1.08 |

---

## Cumulative Progress

```
Baseline:  4380 ns  (1.000×)

Final best: 4380 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
