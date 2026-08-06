# Kernel Optimization Log

## Summary

- **Kernel**: `conv_transpose2d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17980000 ns — **Best latency**: 17980000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv_transpose2d_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_010/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 165.49
- **Registers/thread**: 80
- **Duration**: 17980000 ns
- **SM Throughput**: 64.87 % | **Memory**: 59.65 % | **DRAM**: 4.84 %
- **Achieved Occupancy**: 37.37 %
- **IPC Active**: 2.39 inst/cycle | **No Eligible**: 40.26 %

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
Baseline:  17980000 ns  (1.000×)

Final best: 17980000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
