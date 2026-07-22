# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11840 ns — **Best latency**: 11840 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_10_triton_poi_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.55
- **Registers/thread**: 48
- **Duration**: 11840 ns
- **SM Throughput**: 27.71 % | **Memory**: 61.76 % | **DRAM**: 61.76 %
- **Achieved Occupancy**: 49.90 %
- **IPC Active**: 1.36 inst/cycle | **No Eligible**: 65.68 %

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
Baseline:  11840 ns  (1.000×)

Final best: 11840 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
