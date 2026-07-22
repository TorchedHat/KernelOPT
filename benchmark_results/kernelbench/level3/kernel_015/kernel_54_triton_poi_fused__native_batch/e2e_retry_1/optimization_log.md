# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3940 ns — **Best latency**: 3940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_54_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3940 ns
- **SM Throughput**: 26.55 % | **Memory**: 25.69 % | **DRAM**: 3.11 %
- **Achieved Occupancy**: 66.32 %
- **IPC Active**: 2.29 inst/cycle | **No Eligible**: 42.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 0.980× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3940 ns | 3900 ns | -40 ns (-1.0%) |
| SM Throughput (%) | 26.55 % | 26.49 % | -0.06 pp |
| Memory Throughput (%) | 25.69 % | 25.67 % | -0.02 pp |
| DRAM Throughput (%) | 3.11 % | 3.12 % | +0.01 pp |
| Achieved Occupancy (%) | 66.32 % | 66.40 % | +0.08 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 42.90 % | 45.79 % | +2.89 pp |
| IPC Active | 2.29 inst/cycle | 2.19 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  3940 ns  (1.000×)

Final best: 3940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
