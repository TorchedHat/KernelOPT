# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7460 ns — **Best latency**: 7460 ns (**1.0000×** vs baseline)
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
- **Duration**: 7460 ns
- **SM Throughput**: 21.23 % | **Memory**: 15.09 % | **DRAM**: 11.49 %
- **Achieved Occupancy**: 45.57 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.964× | 0.988× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.966× | 0.971× | ✓ |  |
| C | add_triton_autotune | 0.995× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 7550 ns | +90 ns (+1.2%) | 7680 ns | +220 ns (+2.9%) | 7490 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 21.23 % | 24.71 % | +3.48 pp | 25.26 % | +4.03 pp | 21.64 % | +0.41 pp |
| Memory Throughput (%) | 15.09 % | 14.38 % | -0.71 pp | 14.39 % | -0.70 pp | 15.14 % | +0.05 pp |
| DRAM Throughput (%) | 11.49 % | 11.39 % | -0.10 pp | 11.19 % | -0.30 pp | 11.46 % | -0.03 pp |
| Achieved Occupancy (%) | 45.57 % | 37.22 % | -8.35 pp | 37.04 % | -8.53 pp | 42.64 % | -2.93 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 69.22 % | 56.66 % | -12.56 pp | 56.63 % | -12.59 pp | 67.88 % | -1.34 pp |
| IPC Active | 1.22 inst/cycle | 1.66 inst/cycle | +0.44 | 1.66 inst/cycle | +0.44 | 1.27 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  7460 ns  (1.000×)

Final best: 7460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
