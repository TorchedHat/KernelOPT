# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_div_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 531040 ns — **Best latency**: 531040 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_div_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_008/kernel_0_triton_poi_fused_convolution_d/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (861056,1,1), Waves/SM: 407.70
- **Registers/thread**: 16
- **Duration**: 531040 ns
- **SM Throughput**: 31.16 % | **Memory**: 33.15 % | **DRAM**: 33.15 %
- **Achieved Occupancy**: 42.07 %
- **IPC Active**: 1.25 inst/cycle | **No Eligible**: 68.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  531040 ns  (1.000×)

Final best: 531040 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
