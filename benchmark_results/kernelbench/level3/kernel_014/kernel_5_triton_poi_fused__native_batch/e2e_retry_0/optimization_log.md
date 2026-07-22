# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12290 ns — **Best latency**: 12290 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12290 ns
- **SM Throughput**: 47.50 % | **Memory**: 20.98 % | **DRAM**: 13.33 %
- **Achieved Occupancy**: 57.00 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 42.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 0.924× | 0.923× | ✓ |  |
| B | autotune_xblock_num_warps_num_stages | 0.984× | 0.241× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 13310 ns | +1020 ns (+8.3%) | 51040 ns | +38750 ns (+315.3%) |
| SM Throughput (%) | 47.50 % | 66.97 % | +19.47 pp | 85.00 % | +37.50 pp |
| Memory Throughput (%) | 20.98 % | 25.63 % | +4.65 pp | 18.08 % | -2.90 pp |
| DRAM Throughput (%) | 13.33 % | 12.30 % | -1.03 pp | 3.21 % | -10.12 pp |
| Achieved Occupancy (%) | 57.00 % | 68.60 % | +11.60 pp | 55.21 % | -1.79 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 48.00 register/thread | +31 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 23.76 | +16.34 |
| No Eligible (%) | 42.90 % | 15.71 % | -27.19 pp | 11.40 % | -31.50 pp |
| IPC Active | 2.31 inst/cycle | 3.35 inst/cycle | +1.04 | 3.54 inst/cycle | +1.23 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_num_warps_num_stages` (0.241×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the compiler to select a configuration that increased registers per thread from 17 to 48, reducing Block Limit Registers from 21 to 5 and capping theoretical occupancy at 62.5%. More critically, executed instructions exploded from ~4.6M to ~34.2M (7.5x), indicating the larger tile size triggered significant register pressure and/or loop unrolling overhead that far outweighed the latency-hiding benefit. The kernel duration increased from 12.29 µs to 51.04 µs (4.15x slower) despite compute SM throughput improving from 47.5% to 85%, meaning the kernel is now doing far more work per element rather than hiding memory latency.

> **Applicable when**: Never: this specific pattern (autotune selecting larger XBLOCK for a simple pointwise kernel with very low register count) caused severe regression. Do not apply autotune with large XBLOCK configs when the baseline kernel already has very low register count (≤20) and the arithmetic is simple (BN+ReLU), as the compiler may select configs that dramatically increase register usage and instruction count.

> **Avoid when**: Pointwise elementwise kernels with simple arithmetic (≤10 ops per element) and low baseline register count (≤20 registers/thread). When the grid is already large (>10K CTAs) and the bottleneck is L1TEX scoreboard stalls, adding autotune with large XBLOCK may cause the compiler to unroll aggressively, spiking register pressure and instruction count. Also avoid when xnumel is not cleanly divisible by all candidate XBLOCK values, as the added masking overhead compounds the regression.

---

## Cumulative Progress

```
Baseline:  12290 ns  (1.000×)

Final best: 12290 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
