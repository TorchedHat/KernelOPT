# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 655840 ns — **Best latency**: 655840 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_5_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 99.30
- **Registers/thread**: 95
- **Duration**: 655840 ns
- **SM Throughput**: 67.47 % | **Memory**: 88.53 % | **DRAM**: 33.05 %
- **Achieved Occupancy**: 30.24 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 32.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  655840 ns  (1.000×)

Final best: 655840 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
