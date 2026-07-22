# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5630 ns — **Best latency**: 5630 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_006/kernel_2_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2352,1,1), Waves/SM: 1.11
- **Registers/thread**: 22
- **Duration**: 5630 ns
- **SM Throughput**: 22.04 % | **Memory**: 40.10 % | **DRAM**: 40.10 %
- **Achieved Occupancy**: 67.89 %
- **IPC Active**: 1.37 inst/cycle | **No Eligible**: 64.00 %

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
Baseline:  5630 ns  (1.000×)

Final best: 5630 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
