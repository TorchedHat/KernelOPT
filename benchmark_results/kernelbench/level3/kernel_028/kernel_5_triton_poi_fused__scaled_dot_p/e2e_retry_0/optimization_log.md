# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3390 ns — **Best latency**: 3390 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_5_triton_poi_fused__scaled_dot_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3390 ns
- **SM Throughput**: 10.59 % | **Memory**: 7.16 % | **DRAM**: 4.92 %
- **Achieved Occupancy**: 52.75 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 67.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_for_persistent_tile_scheduling | 0.994× | 1.093× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.997× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3100 ns | -290 ns (-8.6%) | 3550 ns | +160 ns (+4.7%) |
| SM Throughput (%) | 10.59 % | 13.02 % | +2.43 pp | 10.20 % | -0.39 pp |
| Memory Throughput (%) | 7.16 % | 7.31 % | +0.15 pp | 6.98 % | -0.18 pp |
| DRAM Throughput (%) | 4.92 % | 5.35 % | +0.43 pp | 4.69 % | -0.23 pp |
| Achieved Occupancy (%) | 52.75 % | 45.59 % | -7.16 pp | 52.41 % | -0.34 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 67.75 % | 56.65 % | -11.10 pp | 73.59 % | +5.84 pp |
| IPC Active | 1.16 inst/cycle | 1.53 inst/cycle | +0.37 | 1.11 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  3390 ns  (1.000×)

Final best: 3390 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
