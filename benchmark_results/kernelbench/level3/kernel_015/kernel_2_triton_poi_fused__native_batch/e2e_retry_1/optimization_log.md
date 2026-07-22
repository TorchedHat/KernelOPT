# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21660 ns — **Best latency**: 21660 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 16
- **Duration**: 21660 ns
- **SM Throughput**: 35.48 % | **Memory**: 23.39 % | **DRAM**: 15.11 %
- **Achieved Occupancy**: 40.60 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 61.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 0.909× | 0.932× | ✓ |  |
| B | autotune_xblock_num_warps_for_latency_hiding | 0.987× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 21660 ns | 23230 ns | +1570 ns (+7.2%) | 21860 ns | +200 ns (+0.9%) |
| SM Throughput (%) | 35.48 % | 58.43 % | +22.95 pp | 48.24 % | +12.76 pp |
| Memory Throughput (%) | 23.39 % | 34.65 % | +11.26 pp | 32.03 % | +8.64 pp |
| DRAM Throughput (%) | 15.11 % | 14.10 % | -1.01 pp | 15.00 % | -0.11 pp |
| Achieved Occupancy (%) | 40.60 % | 42.06 % | +1.46 pp | 33.39 % | -7.21 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 24.00 register/thread | +8 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 61.62 % | 33.01 % | -28.61 pp | 43.29 % | -18.33 pp |
| IPC Active | 1.53 inst/cycle | 2.68 inst/cycle | +1.15 | 2.25 inst/cycle | +0.72 |

---

## Cumulative Progress

```
Baseline:  21660 ns  (1.000×)

Final best: 21660 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
