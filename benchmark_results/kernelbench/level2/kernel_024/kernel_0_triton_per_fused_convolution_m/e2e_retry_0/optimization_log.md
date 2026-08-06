# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_min_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 64770 ns — **Best latency**: 64770 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_min_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_024/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (21600,1,1), Waves/SM: 10.23
- **Registers/thread**: 32
- **Duration**: 64770 ns
- **SM Throughput**: 34.38 % | **Memory**: 80.01 % | **DRAM**: 80.01 %
- **Achieved Occupancy**: 95.30 %
- **IPC Active**: 1.41 inst/cycle | **No Eligible**: 65.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  64770 ns  (1.000×)

Final best: 64770 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
