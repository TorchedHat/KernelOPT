# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15980000 ns — **Best latency**: 15980000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_011/kernel_3_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.02
- **Registers/thread**: 32
- **Duration**: 15980000 ns
- **SM Throughput**: 1.18 % | **Memory**: 3.62 % | **DRAM**: 3.62 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.08 inst/cycle | **No Eligible**: 98.09 %

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
Baseline:  15980000 ns  (1.000×)

Final best: 15980000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
