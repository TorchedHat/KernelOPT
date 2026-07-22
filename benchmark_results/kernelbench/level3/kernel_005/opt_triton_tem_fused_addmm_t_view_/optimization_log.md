# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22600000 ns — **Best latency**: 22600000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_005/synthesis/kernel_11_triton_tem_fused_addmm_t_view_18.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (24200,1,1), Waves/SM: 183.33
- **Registers/thread**: 230
- **Duration**: 22600000 ns
- **SM Throughput**: 41.27 % | **Memory**: 72.66 % | **DRAM**: 1.61 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.56 %

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
Baseline:  22600000 ns  (1.000×)

Final best: 22600000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
