# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7100 ns — **Best latency**: 7100 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_46_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4900,1,1), Waves/SM: 2.32
- **Registers/thread**: 29
- **Duration**: 7100 ns
- **SM Throughput**: 52.00 % | **Memory**: 24.54 % | **DRAM**: 24.54 %
- **Achieved Occupancy**: 75.98 %
- **IPC Active**: 2.78 inst/cycle | **No Eligible**: 27.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 0.846× | 0.718× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.845× | 0.718× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7100 ns | 9890 ns | +2790 ns (+39.3%) | 9890 ns | +2790 ns (+39.3%) |
| SM Throughput (%) | 52.00 % | 61.33 % | +9.33 pp | 61.56 % | +9.56 pp |
| Memory Throughput (%) | 24.54 % | 17.64 % | -6.90 pp | 17.64 % | -6.90 pp |
| DRAM Throughput (%) | 24.54 % | 17.64 % | -6.90 pp | 17.64 % | -6.90 pp |
| Achieved Occupancy (%) | 75.98 % | 61.36 % | -14.62 pp | 61.09 % | -14.89 pp |
| Registers/Thread | 29.00 register/thread | 38.00 register/thread | +9 | 38.00 register/thread | +9 |
| Waves/SM | 2.32 | 3.09 | +0.77 | 3.09 | +0.77 |
| No Eligible (%) | 27.74 % | 22.32 % | -5.42 pp | 22.20 % | -5.54 pp |
| IPC Active | 2.78 inst/cycle | 3.08 inst/cycle | +0.30 | 3.09 inst/cycle | +0.31 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_to_eliminate_partial_wave` (0.718×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024) caused the compiler to allocate more registers per thread (29 → 38), reducing the register-limited block count per SM from 16 to 12 and dropping theoretical occupancy from 100% to 75% and achieved occupancy from 76% to 61%. This occupancy loss more than offset any partial-wave tail reduction: elapsed cycles increased from 10,536 to 14,703 and kernel duration grew from 7.1 µs to 9.89 µs. The instruction count also nearly doubled (2.86M → 4.75M), suggesting the autotune selected a larger-XBLOCK config that increased per-thread work without proportional throughput gain.

> **Applicable when**: Never apply this pattern as described; it is a cautionary anti-pattern.

> **Avoid when**: Do NOT add @triton.autotune with larger XBLOCK configs when the kernel already has low register count (≤32 regs/thread) and high theoretical occupancy (≥100%). Increasing XBLOCK via autotune can cause the compiler to allocate more registers, reducing Block Limit Registers and theoretical occupancy, which outweighs any partial-wave tail savings. Also avoid when the grid function in kernel_function is not updated to use the autotune-selected tile size (mismatch between hardcoded grid and autotune XBLOCK causes incorrect launch).

---

## Cumulative Progress

```
Baseline:  7100 ns  (1.000×)

Final best: 7100 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
