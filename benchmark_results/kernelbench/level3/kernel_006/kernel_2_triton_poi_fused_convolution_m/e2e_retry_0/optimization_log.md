# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2910000 ns — **Best latency**: 2910000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (235200,1,1), Waves/SM: 356.36
- **Registers/thread**: 94
- **Duration**: 2910000 ns
- **SM Throughput**: 24.87 % | **Memory**: 70.10 % | **DRAM**: 33.62 %
- **Achieved Occupancy**: 30.35 %
- **IPC Active**: 1.00 inst/cycle | **No Eligible**: 75.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  2910000 ns  (1.000×)

Final best: 2910000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
