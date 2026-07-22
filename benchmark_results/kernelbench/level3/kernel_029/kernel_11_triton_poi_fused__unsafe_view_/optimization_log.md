# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 6080 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_11_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 16
- **Duration**: 6080 ns
- **SM Throughput**: 16.81 % | **Memory**: 13.80 % | **DRAM**: 10.17 %
- **Achieved Occupancy**: 47.81 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)

Final best: 6080 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
