# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42500 ns — **Best latency**: 42500 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_0_triton_poi_fused_cat_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 16
- **Duration**: 42500 ns
- **SM Throughput**: 26.41 % | **Memory**: 23.38 % | **DRAM**: 23.38 %
- **Achieved Occupancy**: 40.07 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_latency_hiding | 0.909× | 0.927× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 42500 ns | 45860 ns | +3360 ns (+7.9%) |
| SM Throughput (%) | 26.41 % | 30.10 % | +3.69 pp |
| Memory Throughput (%) | 23.38 % | 20.61 % | -2.77 pp |
| DRAM Throughput (%) | 23.38 % | 20.61 % | -2.77 pp |
| Achieved Occupancy (%) | 40.07 % | 34.85 % | -5.22 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 |
| No Eligible (%) | 71.94 % | 64.97 % | -6.97 pp |
| IPC Active | 1.12 inst/cycle | 1.37 inst/cycle | +0.25 |

---

## Cumulative Progress

```
Baseline:  42500 ns  (1.000×)

Final best: 42500 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
