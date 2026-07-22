# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_expand_unsqueeze_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2370 ns — **Best latency**: 2370 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_expand_unsqueeze_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_4_triton_poi_fused_cat_expand_un/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (10,1,1), Waves/SM: 0.00
- **Registers/thread**: 16
- **Duration**: 2370 ns
- **SM Throughput**: 0.08 % | **Memory**: 0.41 % | **DRAM**: 0.03 %
- **Achieved Occupancy**: 6.14 %
- **IPC Active**: 0.16 inst/cycle | **No Eligible**: 95.24 %

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
Baseline:  2370 ns  (1.000×)

Final best: 2370 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
