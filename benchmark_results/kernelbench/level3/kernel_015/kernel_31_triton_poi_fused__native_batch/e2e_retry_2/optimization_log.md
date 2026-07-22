# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7490 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_31_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7490 ns
- **SM Throughput**: 39.09 % | **Memory**: 17.16 % | **DRAM**: 10.97 %
- **Achieved Occupancy**: 57.62 %
- **IPC Active**: 2.23 inst/cycle | **No Eligible**: 43.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.981× | 1.004× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.917× | 0.785× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7460 ns | -30 ns (-0.4%) | 9540 ns | +2050 ns (+27.4%) |
| SM Throughput (%) | 39.09 % | 45.43 % | +6.34 pp | 66.47 % | +27.38 pp |
| Memory Throughput (%) | 17.16 % | 16.80 % | -0.36 pp | 18.12 % | +0.96 pp |
| DRAM Throughput (%) | 10.97 % | 11.01 % | +0.04 pp | 8.63 % | -2.34 pp |
| Achieved Occupancy (%) | 57.62 % | 60.00 % | +2.38 pp | 81.76 % | +24.14 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 43.19 % | 34.16 % | -9.03 pp | 15.74 % | -27.45 pp |
| IPC Active | 2.23 inst/cycle | 2.62 inst/cycle | +0.39 | 3.29 inst/cycle | +1.06 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.785×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) improved achieved occupancy (57.6%→81.8%) and eligible warps per scheduler (1.39→6.44), but the selected config doubled register usage per thread (17→32), which more than doubled total executed instructions (2.29M→5.03M) and increased kernel duration by 27% (7.49→9.54 µs). The warp-cycles-per-issued-instruction metric barely changed (15.89→15.46), confirming the L1TEX stall was not the binding bottleneck—the extra instruction count from the larger tile was the dominant cost. When the kernel is already short (< 10 µs) and the instruction count is low, expanding XBLOCK via autotune can cause the autotuner to select a config that inflates register pressure and instruction count faster than it hides latency.

> **Applicable when**: Never apply this specific pattern (autotuning XBLOCK upward from a small baseline) when the kernel is already short-running (< 10 µs), has low register count (< 20 regs/thread at baseline), and the instruction count is dominated by compute rather than memory stalls.

> **Avoid when**: Do not apply large-XBLOCK autotuning when: (1) baseline registers/thread are low (≤ 20) and the autotuner may select configs that double register usage; (2) the kernel duration is already under 10 µs and instruction-count inflation will dominate any latency-hiding benefit; (3) warp-cycles-per-issued-instruction does not decrease meaningfully after the change, indicating the stall was not the true bottleneck.

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)

Final best: 7490 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
