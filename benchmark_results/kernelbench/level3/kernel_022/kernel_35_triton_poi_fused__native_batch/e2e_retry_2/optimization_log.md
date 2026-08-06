# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4290 ns — **Best latency**: 4290 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_35_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2573,1,1), Waves/SM: 1.22
- **Registers/thread**: 18
- **Duration**: 4290 ns
- **SM Throughput**: 25.97 % | **Memory**: 9.62 % | **DRAM**: 6.38 %
- **Achieved Occupancy**: 60.91 %
- **IPC Active**: 2.10 inst/cycle | **No Eligible**: 46.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.958× | 0.905× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave | 0.999× | 0.390× | ✓ |  |
| C | add_triton_autotune | 0.966× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 4740 ns | +450 ns (+10.5%) | 11010 ns | +6720 ns (+156.6%) | 4320 ns | +30 ns (+0.7%) |
| SM Throughput (%) | 25.97 % | 43.97 % | +18.00 pp | 67.85 % | +41.88 pp | 25.98 % | +0.01 pp |
| Memory Throughput (%) | 9.62 % | 12.04 % | +2.42 pp | 13.70 % | +4.08 pp | 9.48 % | -0.14 pp |
| DRAM Throughput (%) | 6.38 % | 5.77 % | -0.61 pp | 2.48 % | -3.90 pp | 6.34 % | -0.04 pp |
| Achieved Occupancy (%) | 60.91 % | 70.21 % | +9.30 pp | 52.91 % | -8.00 pp | 61.17 % | +0.26 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 48.00 register/thread | +30 | 18.00 register/thread | +0 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 3.90 | +2.68 | 1.22 | +0.00 |
| No Eligible (%) | 46.89 % | 22.43 % | -24.46 pp | 17.24 % | -29.65 pp | 47.93 % | +1.04 pp |
| IPC Active | 2.10 inst/cycle | 3.01 inst/cycle | +0.91 | 3.36 inst/cycle | +1.26 | 2.13 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_to_eliminate_partial_wave` (0.390×)

Increasing XBLOCK from 128 to 256–2048 via @triton.autotune caused registers per thread to jump from 18 to 48, capping theoretical occupancy at 62.5% (down from 100%) due to register file pressure. Simultaneously, executed instructions exploded ~6.6x (885K → 5.83M), indicating the autotuner selected a large-tile config that serialized far more work per block than the hardware could efficiently hide. Despite the scheduler's No-Eligible stall rate improving (46.89% → 17.24%), the net effect was a 2.57x increase in kernel duration (4.29 µs → 11.01 µs) because the register-pressure-induced occupancy drop and instruction count increase outweighed the partial-wave elimination benefit.

> **Applicable when**: Kernel has very low register usage (≤24 registers/thread) so that doubling XBLOCK does not push register count past the occupancy cliff; kernel is genuinely latency-bound with high No-Eligible stall rates AND the partial-wave tail is the dominant cost (waves_per_sm close to N+0.5 for small N).

> **Avoid when**: Kernel already uses ≥18 registers/thread and the body contains multiple independent memory loads with arithmetic (e.g., batch-norm with 4 broadcast pointer loads); increasing XBLOCK in such kernels causes register count to scale super-linearly, collapsing theoretical occupancy and multiplying executed instruction count, producing a net regression even when scheduler stall rates improve.

---

## Cumulative Progress

```
Baseline:  4290 ns  (1.000×)

Final best: 4290 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
