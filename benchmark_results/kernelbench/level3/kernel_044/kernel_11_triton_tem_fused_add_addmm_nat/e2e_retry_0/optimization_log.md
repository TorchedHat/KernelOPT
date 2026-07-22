# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_native_layer_norm_t_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16460000 ns — **Best latency**: 16460000 ns (**1.0000×** vs baseline)
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
- **Duration**: 16460000 ns
- **SM Throughput**: 41.56 % | **Memory**: 60.65 % | **DRAM**: 1.76 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 58.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.158× | 0.159× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16460000 ns | 103470000 ns | +87010000 ns (+528.6%) |
| SM Throughput (%) | 41.56 % | 8.07 % | -33.49 pp |
| Memory Throughput (%) | 60.65 % | 16.49 % | -44.16 pp |
| DRAM Throughput (%) | 1.76 % | 12.95 % | +11.19 pp |
| Achieved Occupancy (%) | 12.46 % | 6.25 % | -6.21 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 46.55 | 93.09 | +46.54 |
| No Eligible (%) | 58.14 % | 91.87 % | +33.73 pp |
| IPC Active | 1.68 inst/cycle | 0.32 inst/cycle | -1.36 |

---

## Cumulative Progress

```
Baseline:  16460000 ns  (1.000×)

Final best: 16460000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
