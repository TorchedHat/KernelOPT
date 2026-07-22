# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55900 ns — **Best latency**: 55900 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_35_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 55900 ns
- **SM Throughput**: 7.97 % | **Memory**: 53.93 % | **DRAM**: 0.92 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_warps_to_fix_occupancy_and_bank_conflicts | 0.551× | 0.534× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 55900 ns | 104580 ns | +48680 ns (+87.1%) |
| SM Throughput (%) | 7.97 % | 13.94 % | +5.97 pp |
| Memory Throughput (%) | 53.93 % | 54.17 % | +0.24 pp |
| DRAM Throughput (%) | 0.92 % | 0.49 % | -0.43 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.15 | -0.05 |
| No Eligible (%) | 86.34 % | 76.42 % | -9.92 pp |
| IPC Active | 0.55 inst/cycle | 0.94 inst/cycle | +0.39 |


### Key Learnings

#### ⚠️ Plan B: `autotune_tile_sizes_and_warps_to_fix_occupancy_and_bank_conflicts` (0.534×)

Changing BLOCK_M/BLOCK_N from 32x32 to 64x64 while keeping the launch grid fixed at (80,1,1) causes each block to process 4x more output elements, inflating executed instructions from 3.5M to 11.5M (3.3x) and doubling kernel duration from 55.9 µs to 104.6 µs. The grid must be updated as a lambda of the tile sizes (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)) whenever tile dimensions change, otherwise blocks do redundant or overlapping work. Additionally, the bank conflict rate (77.7%) and achieved occupancy (6.25%) were completely unchanged, meaning the tile change alone did not address the root cause — the access pattern driving bank conflicts must also be fixed.

> **Applicable when**: Any Triton GEMM or tiled kernel where BLOCK_M or BLOCK_N are changed from their original values and the launch grid was previously hardcoded to match the old tile sizes.

> **Avoid when**: Grid is already a lambda or computed dynamically from tile parameters; tile sizes are not changed.

---

## Cumulative Progress

```
Baseline:  55900 ns  (1.000×)

Final best: 55900 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
