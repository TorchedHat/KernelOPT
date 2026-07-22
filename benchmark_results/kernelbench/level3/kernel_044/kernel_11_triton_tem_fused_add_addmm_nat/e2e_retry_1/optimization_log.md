# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_native_layer_norm_t_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16530000 ns — **Best latency**: 16530000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_add_addmm_native_layer_norm_t_view_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_11_triton_tem_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12288,1,1), Waves/SM: 46.55
- **Registers/thread**: 255
- **Duration**: 16530000 ns
- **SM Throughput**: 41.59 % | **Memory**: 60.71 % | **DRAM**: 1.75 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 58.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16530000 ns | 16520000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 41.59 % | 41.53 % | -0.06 pp |
| Memory Throughput (%) | 60.71 % | 60.63 % | -0.08 pp |
| DRAM Throughput (%) | 1.75 % | 1.75 % | +0.00 pp |
| Achieved Occupancy (%) | 12.46 % | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 |
| No Eligible (%) | 58.15 % | 58.13 % | -0.02 pp |
| IPC Active | 1.68 inst/cycle | 1.67 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  16530000 ns  (1.000×)

Final best: 16530000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
