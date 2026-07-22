# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7490 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_3_triton_poi_fused_addmm_relu_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8192,1,1), Waves/SM: 3.88
- **Registers/thread**: 16
- **Duration**: 7490 ns
- **SM Throughput**: 21.11 % | **Memory**: 15.19 % | **DRAM**: 11.51 %
- **Achieved Occupancy**: 43.73 %
- **IPC Active**: 1.26 inst/cycle | **No Eligible**: 67.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_improve_occupancy | 0.991× | 0.983× | ✓ |  |
| B | autotune_xblock_num_warps_for_latency_hiding | 0.979× | 0.988× | ✓ |  |
| C | add_triton_autotune | 0.980× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7620 ns | +130 ns (+1.7%) | 7580 ns | +90 ns (+1.2%) | 7520 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 21.11 % | 25.26 % | +4.15 pp | 21.70 % | +0.59 pp | 21.68 % | +0.57 pp |
| Memory Throughput (%) | 15.19 % | 14.44 % | -0.75 pp | 15.01 % | -0.18 pp | 15.00 % | -0.19 pp |
| DRAM Throughput (%) | 11.51 % | 11.26 % | -0.25 pp | 11.29 % | -0.22 pp | 11.44 % | -0.07 pp |
| Achieved Occupancy (%) | 43.73 % | 36.82 % | -6.91 pp | 43.57 % | -0.16 pp | 43.23 % | -0.50 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 67.79 % | 56.80 % | -10.99 pp | 67.65 % | -0.14 pp | 67.14 % | -0.65 pp |
| IPC Active | 1.26 inst/cycle | 1.64 inst/cycle | +0.38 | 1.26 inst/cycle | +0.00 | 1.29 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)

Final best: 7490 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
