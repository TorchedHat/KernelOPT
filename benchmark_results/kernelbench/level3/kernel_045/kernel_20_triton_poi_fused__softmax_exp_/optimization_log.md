# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5180 ns — **Best latency**: 5180 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_20_triton_poi_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.39
- **Registers/thread**: 48
- **Duration**: 5180 ns
- **SM Throughput**: 14.95 % | **Memory**: 33.24 % | **DRAM**: 33.24 %
- **Achieved Occupancy**: 21.33 %
- **IPC Active**: 1.01 inst/cycle | **No Eligible**: 74.42 %

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
Baseline:  5180 ns  (1.000×)

Final best: 5180 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
