# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_mm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6560 ns — **Best latency**: 6560 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_mm_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_006/kernel_0_triton_red_fused_mm_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 32
- **Duration**: 6560 ns
- **SM Throughput**: 2.25 % | **Memory**: 13.17 % | **DRAM**: 13.17 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.27 inst/cycle | **No Eligible**: 93.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  6560 ns  (1.000×)

Final best: 6560 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
