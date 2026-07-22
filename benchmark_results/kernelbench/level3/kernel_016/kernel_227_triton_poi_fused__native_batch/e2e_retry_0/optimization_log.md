# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7040 ns — **Best latency**: 7040 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_227_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6983,1,1), Waves/SM: 3.31
- **Registers/thread**: 22
- **Duration**: 7040 ns
- **SM Throughput**: 43.24 % | **Memory**: 16.54 % | **DRAM**: 10.47 %
- **Achieved Occupancy**: 59.32 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 34.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_reduce_partial_wave_and_improve_cache_reuse | 0.946× | 0.783× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.941× | 0.777× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7040 ns | 8990 ns | +1950 ns (+27.7%) | 9060 ns | +2020 ns (+28.7%) |
| SM Throughput (%) | 43.24 % | 63.43 % | +20.19 pp | 63.42 % | +20.18 pp |
| Memory Throughput (%) | 16.54 % | 17.12 % | +0.58 pp | 17.11 % | +0.57 pp |
| DRAM Throughput (%) | 10.47 % | 8.18 % | -2.29 pp | 8.14 % | -2.33 pp |
| Achieved Occupancy (%) | 59.32 % | 80.66 % | +21.34 pp | 80.79 % | +21.47 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.31 | 3.31 | +0.00 | 3.31 | +0.00 |
| No Eligible (%) | 34.80 % | 18.17 % | -16.63 pp | 17.51 % | -17.29 pp |
| IPC Active | 2.48 inst/cycle | 3.26 inst/cycle | +0.78 | 3.26 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_eliminate_partial_wave` (0.777×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) increased achieved occupancy from 59% to 81% and compute SM throughput from 43% to 63%, but the kernel slowed by 22% (7.04 µs → 9.06 µs). The root cause is that the autotune selected a config that nearly doubled registers per thread (22 → 32) and almost doubled total executed instructions (2.37M → 4.50M), indicating the larger block size caused the compiler to generate significantly more instructions (likely due to unrolling or spill-fill overhead), overwhelming the gains from better occupancy. When the grid size and waves-per-SM are unchanged (both before and after show grid=6983, waves=3.31), increasing XBLOCK does not reduce the partial wave and instead only adds instruction overhead.

> **Applicable when**: Kernel has a large partial wave AND increasing XBLOCK actually reduces grid size (i.e., grid lambda uses meta['XBLOCK']). Registers per thread remain low after the change. Instruction count does not increase significantly.

> **Avoid when**: The grid is hardcoded and does not adapt to the autotuned XBLOCK (as happened here — grid=(cdiv(xnumel,128),) was kept fixed, so larger XBLOCK configs still ran 6983 blocks but with more instructions per block). Also avoid when the kernel body causes register pressure to increase significantly with larger tile sizes, as the extra instructions and register spills negate occupancy gains.

---

## Cumulative Progress

```
Baseline:  7040 ns  (1.000×)

Final best: 7040 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
