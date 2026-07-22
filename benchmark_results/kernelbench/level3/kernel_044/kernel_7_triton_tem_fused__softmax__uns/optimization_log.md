# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24130 ns — **Best latency**: 24130 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_7_triton_tem_fused__softmax__uns/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.03
- **Registers/thread**: 123
- **Duration**: 24130 ns
- **SM Throughput**: 6.35 % | **Memory**: 5.80 % | **DRAM**: 1.07 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 2.27 inst/cycle | **No Eligible**: 42.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | expand_batch_grid_and_autotune_tile_sizes | 1.000× | 0.979× | ✓ |  |
| C | fix_grid_and_autotune_bmm_tile_parameters | 0.999× | 0.988× | ✓ |  |
| D | fix_grid_and_vectorize_loads_with_autotune | 0.958× | 0.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_batch_grid... | Δ | fix_grid_and_auto... | Δ | fix_grid_and_vect... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24130 ns | 24640 ns | +510 ns (+2.1%) | 24420 ns | +290 ns (+1.2%) | 29440 ns | +5310 ns (+22.0%) |
| SM Throughput (%) | 6.35 % | 6.35 % | +0.00 pp | 6.28 % | -0.07 pp | 5.18 % | -1.17 pp |
| Memory Throughput (%) | 5.80 % | 5.80 % | +0.00 pp | 5.74 % | -0.06 pp | 4.78 % | -1.02 pp |
| DRAM Throughput (%) | 1.07 % | 1.05 % | -0.02 pp | 1.06 % | -0.01 pp | 0.89 % | -0.18 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 123.00 register/thread | 123.00 register/thread | +0 | 123.00 register/thread | +0 | 128.00 register/thread | +5 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.04 | +0.01 |
| No Eligible (%) | 42.83 % | 43.25 % | +0.42 pp | 42.94 % | +0.11 pp | 49.89 % | +7.06 pp |
| IPC Active | 2.27 inst/cycle | 2.27 inst/cycle | +0.00 | 2.28 inst/cycle | +0.01 | 2.00 inst/cycle | -0.27 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  24130 ns  (1.000×)

Final best: 24130 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
