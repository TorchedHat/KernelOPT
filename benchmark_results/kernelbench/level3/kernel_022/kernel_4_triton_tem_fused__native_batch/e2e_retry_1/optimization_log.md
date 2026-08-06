# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1860000 ns — **Best latency**: 1860000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_4_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 7.42
- **Registers/thread**: 168
- **Duration**: 1860000 ns
- **SM Throughput**: 38.63 % | **Memory**: 87.20 % | **DRAM**: 4.71 %
- **Achieved Occupancy**: 18.22 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 60.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1860000 ns | 1870000 ns | +10000 ns (+0.5%) |
| SM Throughput (%) | 38.63 % | 38.59 % | -0.04 pp |
| Memory Throughput (%) | 87.20 % | 87.12 % | -0.08 pp |
| DRAM Throughput (%) | 4.71 % | 4.70 % | -0.01 pp |
| Achieved Occupancy (%) | 18.22 % | 18.22 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 7.42 | 7.42 | +0.00 |
| No Eligible (%) | 60.09 % | 60.10 % | +0.01 pp |
| IPC Active | 1.60 inst/cycle | 1.60 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1860000 ns  (1.000×)

Final best: 1860000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
