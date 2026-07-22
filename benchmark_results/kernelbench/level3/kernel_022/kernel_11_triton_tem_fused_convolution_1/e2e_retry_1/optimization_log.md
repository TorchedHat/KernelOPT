# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 37060 ns — **Best latency**: 37060 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_11_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 168
- **Duration**: 37060 ns
- **SM Throughput**: 24.22 % | **Memory**: 53.83 % | **DRAM**: 3.38 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 70.36 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 37060 ns | 37180 ns | +120 ns (+0.3%) |
| SM Throughput (%) | 24.22 % | 24.27 % | +0.05 pp |
| Memory Throughput (%) | 53.83 % | 53.92 % | +0.09 pp |
| DRAM Throughput (%) | 3.38 % | 3.37 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 |
| No Eligible (%) | 70.36 % | 70.43 % | +0.07 pp |
| IPC Active | 1.18 inst/cycle | 1.18 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  37060 ns  (1.000×)

Final best: 37060 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
