# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23740 ns — **Best latency**: 23740 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_15_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 23740 ns
- **SM Throughput**: 10.62 % | **Memory**: 74.28 % | **DRAM**: 5.21 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_pipeline_stages_to_fix_bank_conflicts_and_occupancy | 0.906× | 0.886× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 23740 ns | 26780 ns | +3040 ns (+12.8%) |
| SM Throughput (%) | 10.62 % | 17.73 % | +7.11 pp |
| Memory Throughput (%) | 74.28 % | 66.28 % | -8.00 pp |
| DRAM Throughput (%) | 5.21 % | 4.62 % | -0.59 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp |
| Registers/Thread | 80.00 register/thread | 96.00 register/thread | +16 |
| Waves/SM | 0.16 | 0.19 | +0.03 |
| No Eligible (%) | 87.45 % | 78.77 % | -8.68 pp |
| IPC Active | 0.50 inst/cycle | 0.85 inst/cycle | +0.35 |

---

## Cumulative Progress

```
Baseline:  23740 ns  (1.000×)

Final best: 23740 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
