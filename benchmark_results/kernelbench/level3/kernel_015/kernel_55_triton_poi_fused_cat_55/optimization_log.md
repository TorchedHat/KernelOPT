# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_55` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2560 ns — **Best latency**: 2560 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_55`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_55_triton_poi_fused_cat_55/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (490,1,1), Waves/SM: 0.23
- **Registers/thread**: 16
- **Duration**: 2560 ns
- **SM Throughput**: 4.51 % | **Memory**: 12.62 % | **DRAM**: 2.06 %
- **Achieved Occupancy**: 17.67 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 82.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  2560 ns  (1.000×)

Final best: 2560 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
