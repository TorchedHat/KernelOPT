# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 126300 ns — **Best latency**: 126300 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (200704,1,1), Waves/SM: 95.03
- **Registers/thread**: 19
- **Duration**: 126300 ns
- **SM Throughput**: 62.77 % | **Memory**: 28.88 % | **DRAM**: 28.82 %
- **Achieved Occupancy**: 60.72 %
- **IPC Active**: 2.56 inst/cycle | **No Eligible**: 35.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.628× | 0.620× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.628× | 0.099× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 126300 ns | 203620 ns | +77320 ns (+61.2%) | 1270000 ns | +1143700 ns (+905.5%) |
| SM Throughput (%) | 62.77 % | 87.18 % | +24.41 pp | 89.05 % | +26.28 pp |
| Memory Throughput (%) | 28.88 % | 27.35 % | -1.53 pp | 18.52 % | -10.36 pp |
| DRAM Throughput (%) | 28.82 % | 17.87 % | -10.95 pp | 2.86 % | -25.96 pp |
| Achieved Occupancy (%) | 60.72 % | 87.99 % | +27.27 pp | 34.41 % | -26.31 pp |
| Registers/Thread | 19.00 register/thread | 32.00 register/thread | +13 | 80.00 register/thread | +61 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 506.83 | +411.80 |
| No Eligible (%) | 35.80 % | 11.90 % | -23.90 pp | 10.72 % | -25.08 pp |
| IPC Active | 2.56 inst/cycle | 3.51 inst/cycle | +0.95 | 3.57 inst/cycle | +1.01 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.620×)

Adding @triton.autotune with larger XBLOCK values (512–2048) to a pointwise Triton kernel caused a 61% regression despite improving achieved occupancy from 60.7% to 88.0%. The root cause is that the larger tile sizes caused registers per thread to jump from 19 to 32, and the total executed instruction count grew by 2.2x (62M → 139M), indicating the autotuner selected a configuration that dramatically increased per-thread work without a proportional reduction in grid size (the grid was still computed with the hardcoded XBLOCK=128 in kernel_function, not updated to use meta['XBLOCK']). The implementation flaw of keeping a static grid=(cdiv(xnumel, 128),) while removing the explicit XBLOCK argument meant the kernel was called with a mismatched grid, causing massive redundant computation.

> **Applicable when**: Pointwise kernels with low achieved occupancy (<70%), high L1TEX stall rates (>40% of CPI), and low register pressure (<30 registers/thread) where increasing tile size could improve ILP and cache reuse.

> **Avoid when**: When the grid launch site is not updated to use a lambda with meta['XBLOCK'] — a static grid computed with the old XBLOCK while the kernel uses a larger autotuned XBLOCK causes massive redundant work and severe regression. Also avoid when register count is already near the block limit register threshold.

#### ❌ Plan B: `autotune_xblock_increase_tile_size` (0.099×)

Adding @triton.autotune with large XBLOCK values (512–4096) caused registers_per_thread to explode from 19 to 80, collapsing theoretical occupancy from 100% to 37.5% (Block_Limit_Registers dropped from 21 to 3 blocks/SM). Although warp cycles per instruction improved (14.9→6.1) and compute throughput rose (63%→89%), the 14x increase in executed instructions (62M→896M) and 10x longer duration (126µs→1.27ms) indicate the autotuner selected a very large tile that serialized work massively. The anti-pattern is applying large-tile autotune to a kernel where the body already has high register pressure; the larger XBLOCK forces the compiler to keep more live values simultaneously, causing register spill and occupancy collapse that far outweighs any latency-hiding benefit.

> **Applicable when**: Never applicable as described. Large-XBLOCK autotune is only safe when the kernel body uses very few registers (<<32/thread) so that Block_Limit_Registers stays high (>=16 blocks/SM) even at large tile sizes.

> **Avoid when**: Kernel has moderate-to-high register usage (>20 registers/thread at baseline). When adding autotune with large XBLOCK configs, the compiler must hold more live values across the larger arange, causing register count to scale with XBLOCK and collapsing occupancy. Also avoid when the grid is already large (>100K CTAs) but the kernel body is compute-light — the bottleneck may not be tile size at all.

---

## Cumulative Progress

```
Baseline:  126300 ns  (1.000×)

Final best: 126300 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
