# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21340 ns — **Best latency**: 21340 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_16_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 21340 ns
- **SM Throughput**: 18.58 % | **Memory**: 70.12 % | **DRAM**: 4.36 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.89 inst/cycle | **No Eligible**: 77.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.991× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 21340 ns | 21470 ns | +130 ns (+0.6%) |
| SM Throughput (%) | 18.58 % | 18.61 % | +0.03 pp |
| Memory Throughput (%) | 70.12 % | 70.25 % | +0.13 pp |
| DRAM Throughput (%) | 4.36 % | 4.33 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.58 % | 77.59 % | +0.01 pp |
| IPC Active | 0.89 inst/cycle | 0.90 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  21340 ns  (1.000×)

Final best: 21340 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
