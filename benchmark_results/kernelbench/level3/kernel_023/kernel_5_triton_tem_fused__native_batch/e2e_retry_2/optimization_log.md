# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45310 ns — **Best latency**: 45310 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_5_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45310 ns
- **SM Throughput**: 30.63 % | **Memory**: 77.30 % | **DRAM**: 13.26 %
- **Achieved Occupancy**: 17.91 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45310 ns | 45570 ns | +260 ns (+0.6%) |
| SM Throughput (%) | 30.63 % | 30.31 % | -0.32 pp |
| Memory Throughput (%) | 77.30 % | 76.48 % | -0.82 pp |
| DRAM Throughput (%) | 13.26 % | 13.15 % | -0.11 pp |
| Achieved Occupancy (%) | 17.91 % | 17.93 % | +0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 |
| No Eligible (%) | 66.74 % | 66.70 % | -0.04 pp |
| IPC Active | 1.33 inst/cycle | 1.32 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  45310 ns  (1.000×)

Final best: 45310 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
