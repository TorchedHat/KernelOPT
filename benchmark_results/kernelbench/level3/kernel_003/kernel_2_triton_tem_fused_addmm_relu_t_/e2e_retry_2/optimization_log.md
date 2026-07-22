# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4030000 ns — **Best latency**: 4030000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_2_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.65
- **Registers/thread**: 168
- **Duration**: 4030000 ns
- **SM Throughput**: 37.45 % | **Memory**: 84.49 % | **DRAM**: 5.52 %
- **Achieved Occupancy**: 12.26 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_block_ptr_gemm | 0.081× | 0.118× | ✓ |  |
| C | add_triton_autotune | 1.000× | 0.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_block_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4030000 ns | 34180000 ns | +30150000 ns (+748.1%) | 4370000 ns | +340000 ns (+8.4%) |
| SM Throughput (%) | 37.45 % | 10.56 % | -26.89 pp | 34.28 % | -3.17 pp |
| Memory Throughput (%) | 84.49 % | 21.23 % | -63.26 pp | 77.96 % | -6.53 pp |
| DRAM Throughput (%) | 5.52 % | 15.10 % | +9.58 pp | 5.09 % | -0.43 pp |
| Achieved Occupancy (%) | 12.26 % | 6.25 % | -6.01 pp | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 162.00 register/thread | -6 |
| Waves/SM | 0.65 | 1.94 | +1.29 | 0.65 | +0.00 |
| No Eligible (%) | 61.23 % | 89.28 % | +28.05 pp | 64.59 % | +3.36 pp |
| IPC Active | 1.55 inst/cycle | 0.43 inst/cycle | -1.12 | 1.42 inst/cycle | -0.13 |

---

## Cumulative Progress

```
Baseline:  4030000 ns  (1.000×)

Final best: 4030000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
