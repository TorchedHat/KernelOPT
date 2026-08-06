# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 826210 ns — **Best latency**: 826210 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 165.49
- **Registers/thread**: 160
- **Duration**: 826210 ns
- **SM Throughput**: 25.61 % | **Memory**: 83.27 % | **DRAM**: 6.87 %
- **Achieved Occupancy**: 18.04 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 74.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  826210 ns  (1.000×)

Final best: 826210 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
