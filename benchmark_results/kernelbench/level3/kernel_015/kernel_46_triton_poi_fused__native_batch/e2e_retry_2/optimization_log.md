# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7260 ns — **Best latency**: 7260 ns (**1.0000×** vs baseline)
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
- **Duration**: 7260 ns
- **SM Throughput**: 51.24 % | **Memory**: 24.04 % | **DRAM**: 24.04 %
- **Achieved Occupancy**: 75.06 %
- **IPC Active**: 2.88 inst/cycle | **No Eligible**: 25.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.830× | 0.720× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave | 0.998× | 1.013× | ✓ |  |
| C | add_triton_autotune | 0.998× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 10080 ns | +2820 ns (+38.8%) | 7170 ns | -90 ns (-1.2%) | 7230 ns | -30 ns (-0.4%) |
| SM Throughput (%) | 51.24 % | 61.59 % | +10.35 pp | 50.44 % | -0.80 pp | 50.90 % | -0.34 pp |
| Memory Throughput (%) | 24.04 % | 17.58 % | -6.46 pp | 24.33 % | +0.29 pp | 24.15 % | +0.11 pp |
| DRAM Throughput (%) | 24.04 % | 17.28 % | -6.76 pp | 24.33 % | +0.29 pp | 24.15 % | +0.11 pp |
| Achieved Occupancy (%) | 75.06 % | 61.48 % | -13.58 pp | 75.40 % | +0.34 pp | 75.89 % | +0.83 pp |
| Registers/Thread | 29.00 register/thread | 38.00 register/thread | +9 | 29.00 register/thread | +0 | 29.00 register/thread | +0 |
| Waves/SM | 2.32 | 3.09 | +0.77 | 2.32 | +0.00 | 2.32 | +0.00 |
| No Eligible (%) | 25.66 % | 22.18 % | -3.48 pp | 26.15 % | +0.49 pp | 26.79 % | +1.13 pp |
| IPC Active | 2.88 inst/cycle | 3.01 inst/cycle | +0.13 | 2.89 inst/cycle | +0.01 | 2.92 inst/cycle | +0.04 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_eliminate_partial_wave` (0.720×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) caused the compiler to allocate more registers per thread (29→38), which reduced theoretical occupancy from 100% to 75% and achieved occupancy from 75% to 61%. The increased register pressure also caused a 66% increase in executed instructions (2.86M→4.75M), likely due to register spilling or additional index computation at larger tile sizes. Despite eliminating the partial wave, the occupancy loss and instruction count increase more than offset the benefit, resulting in a 39% longer kernel duration (7.26→10.08 µs).

> **Applicable when**: Never: this specific pattern caused regression. Avoid when the kernel is already register-pressure-sensitive (registers per thread near the block limit boundary) and theoretical occupancy is at 100%.

> **Avoid when**: Kernel has low register count (≤32/thread) at baseline with 100% theoretical occupancy; increasing XBLOCK will push register count up, reduce block limit from registers, and drop theoretical occupancy below 100%, causing net regression even if partial-wave tail is eliminated. Also avoid when instruction count grows super-linearly with XBLOCK (indicates index recomputation or spill overhead).

---

## Cumulative Progress

```
Baseline:  7260 ns  (1.000×)

Final best: 7260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
