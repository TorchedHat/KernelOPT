# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4770 ns — **Best latency**: 4770 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_4_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 1.45
- **Registers/thread**: 28
- **Duration**: 4770 ns
- **SM Throughput**: 35.02 % | **Memory**: 40.16 % | **DRAM**: 10.55 %
- **Achieved Occupancy**: 70.96 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4770 ns  (1.000×)

Final best: 4770 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
