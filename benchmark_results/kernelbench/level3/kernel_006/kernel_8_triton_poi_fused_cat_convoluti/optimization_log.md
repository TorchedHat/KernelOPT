# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_convolution_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 156130 ns — **Best latency**: 156130 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_convolution_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_8_triton_poi_fused_cat_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (250880,1,1), Waves/SM: 118.79
- **Registers/thread**: 16
- **Duration**: 156130 ns
- **SM Throughput**: 16.28 % | **Memory**: 31.07 % | **DRAM**: 31.07 %
- **Achieved Occupancy**: 39.57 %
- **IPC Active**: 0.66 inst/cycle | **No Eligible**: 83.52 %

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
Baseline:  156130 ns  (1.000×)

Final best: 156130 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
