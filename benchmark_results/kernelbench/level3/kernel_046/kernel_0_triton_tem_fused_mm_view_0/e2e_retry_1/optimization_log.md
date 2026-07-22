# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 406180 ns — **Best latency**: 406180 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_0_triton_tem_fused_mm_view_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3200,1,1), Waves/SM: 4.85
- **Registers/thread**: 96
- **Duration**: 406180 ns
- **SM Throughput**: 76.55 % | **Memory**: 70.54 % | **DRAM**: 22.76 %
- **Achieved Occupancy**: 29.76 %
- **IPC Active**: 3.16 inst/cycle | **No Eligible**: 20.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_and_pipeline_stages | 0.772× | 0.763× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 406180 ns | 532510 ns | +126330 ns (+31.1%) |
| SM Throughput (%) | 76.55 % | 59.24 % | -17.31 pp |
| Memory Throughput (%) | 70.54 % | 79.75 % | +9.21 pp |
| DRAM Throughput (%) | 22.76 % | 17.49 % | -5.27 pp |
| Achieved Occupancy (%) | 29.76 % | 12.45 % | -17.31 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 |
| Waves/SM | 4.85 | 12.12 | +7.27 |
| No Eligible (%) | 20.84 % | 37.83 % | +16.99 pp |
| IPC Active | 3.16 inst/cycle | 2.49 inst/cycle | -0.67 |

---

## Cumulative Progress

```
Baseline:  406180 ns  (1.000×)

Final best: 406180 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
