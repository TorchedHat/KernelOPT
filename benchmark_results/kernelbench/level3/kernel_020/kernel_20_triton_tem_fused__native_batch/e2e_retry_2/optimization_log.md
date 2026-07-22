# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31520 ns — **Best latency**: 31520 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 31520 ns
- **SM Throughput**: 19.37 % | **Memory**: 74.09 % | **DRAM**: 4.91 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.90 inst/cycle | **No Eligible**: 77.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 31520 ns | 31490 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 19.37 % | 19.48 % | +0.11 pp |
| Memory Throughput (%) | 74.09 % | 74.51 % | +0.42 pp |
| DRAM Throughput (%) | 4.91 % | 4.91 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.28 % | 77.42 % | +0.14 pp |
| IPC Active | 0.90 inst/cycle | 0.91 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  31520 ns  (1.000×)

Final best: 31520 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
