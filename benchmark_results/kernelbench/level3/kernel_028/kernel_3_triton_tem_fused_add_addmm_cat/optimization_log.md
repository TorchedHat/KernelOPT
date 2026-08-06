# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_cat_expand_t_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 90140 ns — **Best latency**: 90140 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_add_addmm_cat_expand_t_view_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_3_triton_tem_fused_add_addmm_cat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (84,1,1), Waves/SM: 0.21
- **Registers/thread**: 168
- **Duration**: 90140 ns
- **SM Throughput**: 17.51 % | **Memory**: 39.33 % | **DRAM**: 0.90 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.21 inst/cycle | **No Eligible**: 69.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_fix_b_matrix_addressing | 0.984× | 0.985× | ✓ |  |
| C | annotate_loads_stores_with_eviction_policies | 0.997× | 1.003× | ✓ |  |
| D | increase_tile_sizes_and_fix_b_matrix_load_correctness | 0.141× | 0.140× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_fix_b_m... | Δ | annotate_loads_st... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 90140 ns | 91520 ns | +1380 ns (+1.5%) | 89890 ns | -250 ns (-0.3%) | 643840 ns | +553700 ns (+614.3%) |
| SM Throughput (%) | 17.51 % | 17.27 % | -0.24 pp | 17.54 % | +0.03 pp | 5.98 % | -11.53 pp |
| Memory Throughput (%) | 39.33 % | 38.74 % | -0.59 pp | 39.39 % | +0.06 pp | 12.29 % | -27.04 pp |
| DRAM Throughput (%) | 0.90 % | 0.89 % | -0.01 pp | 0.91 % | +0.01 pp | 4.53 % | +3.63 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.21 | 0.21 | +0.00 | 0.21 | +0.00 | 0.64 | +0.43 |
| No Eligible (%) | 69.94 % | 69.96 % | +0.02 pp | 69.86 % | -0.08 pp | 89.62 % | +19.68 pp |
| IPC Active | 1.21 inst/cycle | 1.20 inst/cycle | -0.01 | 1.21 inst/cycle | +0.00 | 0.42 inst/cycle | -0.79 |

---

## Cumulative Progress

```
Baseline:  90140 ns  (1.000×)

Final best: 90140 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
