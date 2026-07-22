# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21120 ns — **Best latency**: 21120 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_5_triton_poi_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 3.10
- **Registers/thread**: 48
- **Duration**: 21120 ns
- **SM Throughput**: 31.43 % | **Memory**: 72.11 % | **DRAM**: 72.11 %
- **Achieved Occupancy**: 53.96 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.21 %

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
Baseline:  21120 ns  (1.000×)

Final best: 21120 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
