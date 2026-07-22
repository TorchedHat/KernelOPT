# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 126210 ns — **Best latency**: 126210 ns (**1.0000×** vs baseline)
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
- **Duration**: 126210 ns
- **SM Throughput**: 63.35 % | **Memory**: 29.14 % | **DRAM**: 28.86 %
- **Achieved Occupancy**: 60.77 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 35.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.876× | 0.171× | ✓ |  |
| B | increase_xblock_with_autotune_to_reduce_l1tex_stalls_and_improve_occupancy | 0.628× | 0.618× | ✓ |  |
| C | add_triton_autotune | 0.997× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126210 ns | 739490 ns | +613280 ns (+485.9%) | 204130 ns | +77920 ns (+61.7%) | 126080 ns | -130 ns (-0.1%) |
| SM Throughput (%) | 63.35 % | 89.03 % | +25.68 pp | 87.14 % | +23.79 pp | 76.49 % | +13.14 pp |
| Memory Throughput (%) | 29.14 % | 18.95 % | -10.19 pp | 27.30 % | -1.84 pp | 29.02 % | -0.12 pp |
| DRAM Throughput (%) | 28.86 % | 4.92 % | -23.94 pp | 17.81 % | -11.05 pp | 28.85 % | -0.01 pp |
| Achieved Occupancy (%) | 60.77 % | 87.95 % | +27.18 pp | 87.39 % | +26.62 pp | 66.68 % | +5.91 pp |
| Registers/Thread | 19.00 register/thread | 32.00 register/thread | +13 | 32.00 register/thread | +13 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 380.12 | +285.09 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.75 % | 10.69 % | -25.06 pp | 12.11 % | -23.64 pp | 21.68 % | -14.07 pp |
| IPC Active | 2.55 inst/cycle | 3.57 inst/cycle | +1.02 | 3.52 inst/cycle | +0.97 | 3.13 inst/cycle | +0.58 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.171×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a kernel that was already running at 95 waves/SM caused the autotune to select XBLOCK=512, which increased registers/thread from 19 to 32 and exploded executed instructions from 62M to 519M (8.3x). The grid lambda was not updated to use meta['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128)=200704 CTAs while the kernel body now processed 512 elements per thread — causing each element to be processed ~4x redundantly. The duration increased from 126 µs to 739 µs (5.9x regression). When the grid is not updated to match the new XBLOCK, autotune silently selects a larger block size that over-processes the input.

> **Applicable when**: Never applicable in this form. The correct pattern requires grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),) when using @triton.autotune with variable XBLOCK.

> **Avoid when**: Do NOT add @triton.autotune with variable XBLOCK configs without simultaneously updating the grid to use a lambda that reads meta['XBLOCK']. If the grid remains hardcoded to a fixed XBLOCK value, autotune will select a larger block size causing each element to be processed multiple times, exploding instruction count and duration. Also avoid when the kernel already has high waves/SM (>80) as the occupancy headroom is already limited.

---

## Cumulative Progress

```
Baseline:  126210 ns  (1.000×)

Final best: 126210 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
