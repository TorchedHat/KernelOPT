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
- **SM Throughput**: 51.73 % | **Memory**: 23.95 % | **DRAM**: 23.95 %
- **Achieved Occupancy**: 76.47 %
- **IPC Active**: 2.94 inst/cycle | **No Eligible**: 25.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 0.847× | 0.736× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.832× | 0.732× | ✓ |  |
| C | add_triton_autotune | 1.000× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 9860 ns | +2600 ns (+35.8%) | 9920 ns | +2660 ns (+36.6%) | 7140 ns | -120 ns (-1.7%) |
| SM Throughput (%) | 51.73 % | 62.04 % | +10.31 pp | 61.84 % | +10.11 pp | 50.93 % | -0.80 pp |
| Memory Throughput (%) | 23.95 % | 17.66 % | -6.29 pp | 17.63 % | -6.32 pp | 24.54 % | +0.59 pp |
| DRAM Throughput (%) | 23.95 % | 17.65 % | -6.30 pp | 17.63 % | -6.32 pp | 24.54 % | +0.59 pp |
| Achieved Occupancy (%) | 76.47 % | 60.75 % | -15.72 pp | 61.06 % | -15.41 pp | 73.91 % | -2.56 pp |
| Registers/Thread | 29.00 register/thread | 38.00 register/thread | +9 | 38.00 register/thread | +9 | 29.00 register/thread | +0 |
| Waves/SM | 2.32 | 3.09 | +0.77 | 3.09 | +0.77 | 2.32 | +0.00 |
| No Eligible (%) | 25.98 % | 22.16 % | -3.82 pp | 22.07 % | -3.91 pp | 26.61 % | +0.63 pp |
| IPC Active | 2.94 inst/cycle | 3.08 inst/cycle | +0.14 | 3.08 inst/cycle | +0.14 | 2.86 inst/cycle | -0.08 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.732×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to eliminate a partial wave caused a regression because the larger tile sizes forced the compiler to allocate more registers per thread (29→38), which reduced Block Limit Registers from 16 to 12 and dropped theoretical occupancy from 100% to 75% and achieved occupancy from 76.5% to 61.1%. The increased register pressure also caused a 66% increase in executed instructions (2.86M→4.75M), overwhelming any benefit from eliminating the partial wave tail. When the kernel is already register-constrained near the occupancy ceiling, increasing XBLOCK inflates register pressure and causes a net slowdown despite reducing grid size.

> **Applicable when**: Kernel has a partial wave (waves_per_sm is non-integer with significant fractional part), low register usage per thread (well below 32), and theoretical occupancy is already at 100% with headroom in Block Limit Registers.

> **Avoid when**: Kernel is already near register budget limits (registers_per_thread >= 28–32); increasing XBLOCK will inflate register pressure, reduce Block Limit Registers, drop theoretical and achieved occupancy, and increase total instruction count — causing a net regression even if grid size is reduced. Also avoid when the kernel body contains many live variables that scale with XBLOCK tile size.

---

## Cumulative Progress

```
Baseline:  7260 ns  (1.000×)

Final best: 7260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
