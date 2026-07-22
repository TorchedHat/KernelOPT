# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 53630 ns — **Best latency**: 53630 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 3.71
- **Registers/thread**: 214
- **Duration**: 53630 ns
- **SM Throughput**: 23.62 % | **Memory**: 59.71 % | **DRAM**: 8.71 %
- **Achieved Occupancy**: 12.39 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_reduce_register_pressure | 0.999× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_reduce_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 53630 ns | 52580 ns | -1050 ns (-2.0%) |
| SM Throughput (%) | 23.62 % | 23.60 % | -0.02 pp |
| Memory Throughput (%) | 59.71 % | 59.69 % | -0.02 pp |
| DRAM Throughput (%) | 8.71 % | 8.78 % | +0.07 pp |
| Achieved Occupancy (%) | 12.39 % | 12.41 % | +0.02 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 |
| No Eligible (%) | 72.90 % | 73.12 % | +0.22 pp |
| IPC Active | 1.08 inst/cycle | 1.08 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  53630 ns  (1.000×)

Final best: 53630 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
