# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6560 ns — **Best latency**: 6560 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_33_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 20
- **Duration**: 6560 ns
- **SM Throughput**: 44.98 % | **Memory**: 14.52 % | **DRAM**: 9.44 %
- **Achieved Occupancy**: 74.47 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 30.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.983× | 0.924× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave_and_improve_utilization | 0.960× | 0.899× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 7100 ns | +540 ns (+8.2%) | 7300 ns | +740 ns (+11.3%) |
| SM Throughput (%) | 44.98 % | 60.20 % | +15.22 pp | 60.33 % | +15.35 pp |
| Memory Throughput (%) | 14.52 % | 27.82 % | +13.30 pp | 27.77 % | +13.25 pp |
| DRAM Throughput (%) | 9.44 % | 8.70 % | -0.74 pp | 8.52 % | -0.92 pp |
| Achieved Occupancy (%) | 74.47 % | 76.40 % | +1.93 pp | 76.76 % | +2.29 pp |
| Registers/Thread | 20.00 register/thread | 31.00 register/thread | +11 | 31.00 register/thread | +11 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 30.33 % | 17.28 % | -13.05 pp | 17.96 % | -12.37 pp |
| IPC Active | 2.55 inst/cycle | 3.28 inst/cycle | +0.73 | 3.29 inst/cycle | +0.74 |

---

## Cumulative Progress

```
Baseline:  6560 ns  (1.000×)

Final best: 6560 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
