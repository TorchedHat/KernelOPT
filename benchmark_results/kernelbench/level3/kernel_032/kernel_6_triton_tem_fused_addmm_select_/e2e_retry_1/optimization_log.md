# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_select_t_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9500 ns — **Best latency**: 9500 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_select_t_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_6_triton_tem_fused_addmm_select_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.03
- **Registers/thread**: 38
- **Duration**: 9500 ns
- **SM Throughput**: 1.56 % | **Memory**: 15.19 % | **DRAM**: 1.16 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.32 inst/cycle | **No Eligible**: 91.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_with_autotune_tile_sizes | 0.683× | 0.602× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 9500 ns | 15780 ns | +6280 ns (+66.1%) |
| SM Throughput (%) | 1.56 % | 1.58 % | +0.02 pp |
| Memory Throughput (%) | 15.19 % | 17.65 % | +2.46 pp |
| DRAM Throughput (%) | 1.16 % | 0.69 % | -0.47 pp |
| Achieved Occupancy (%) | 6.23 % | 6.25 % | +0.02 pp |
| Registers/Thread | 38.00 register/thread | 42.00 register/thread | +4 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 91.92 % | 92.35 % | +0.43 pp |
| IPC Active | 0.32 inst/cycle | 0.30 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  9500 ns  (1.000×)

Final best: 9500 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
