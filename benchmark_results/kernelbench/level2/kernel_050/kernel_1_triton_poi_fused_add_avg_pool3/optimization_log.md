# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_avg_pool3d_convolution_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 246850 ns — **Best latency**: 246850 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_avg_pool3d_convolution_mul_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_050/kernel_1_triton_poi_fused_add_avg_pool3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (230640,1,1), Waves/SM: 109.20
- **Registers/thread**: 24
- **Duration**: 246850 ns
- **SM Throughput**: 61.76 % | **Memory**: 88.99 % | **DRAM**: 88.99 %
- **Achieved Occupancy**: 87.12 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 38.01 %

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
Baseline:  246850 ns  (1.000×)

Final best: 246850 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
