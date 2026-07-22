# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12290 ns — **Best latency**: 12290 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_22_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12290 ns
- **SM Throughput**: 47.49 % | **Memory**: 21.01 % | **DRAM**: 13.35 %
- **Achieved Occupancy**: 55.68 %
- **IPC Active**: 2.32 inst/cycle | **No Eligible**: 40.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.542× | 0.409× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency | 0.996× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 30050 ns | +17760 ns (+144.5%) | 12350 ns | +60 ns (+0.5%) |
| SM Throughput (%) | 47.49 % | 78.00 % | +30.51 pp | 55.67 % | +8.18 pp |
| Memory Throughput (%) | 21.01 % | 27.73 % | +6.72 pp | 21.03 % | +0.02 pp |
| DRAM Throughput (%) | 13.35 % | 5.44 % | -7.91 pp | 13.31 % | -0.04 pp |
| Achieved Occupancy (%) | 55.68 % | 78.51 % | +22.83 pp | 61.29 % | +5.61 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 29.70 | +22.28 | 7.42 | +0.00 |
| No Eligible (%) | 40.48 % | 16.56 % | -23.92 pp | 31.13 % | -9.35 pp |
| IPC Active | 2.32 inst/cycle | 3.34 inst/cycle | +1.02 | 2.78 inst/cycle | +0.46 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.409×)

Increasing the tile size (XBLOCK) from 128 to 512 via autotune improved scheduler eligibility (No Eligible dropped from 40.5% to 16.6%) and achieved occupancy (55.7% → 78.5%), but caused a 4x increase in total executed instructions (4.6M → 18.4M) and 2.4x longer kernel duration (12.3 µs → 30.1 µs). The regression was caused by a critical implementation flaw: the grid was not updated to use `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — it remained hardcoded at `triton.cdiv(xnumel, 128)`, so with XBLOCK=512 the grid launched 4x too many CTAs, each processing the same data redundantly. When adding autotune to change XBLOCK, the launch grid MUST be a dynamic lambda that reads `meta['XBLOCK']`, not a static expression computed with the old tile size.

> **Applicable when**: Never apply this pattern (static grid + autotune XBLOCK). Always pair autotune XBLOCK changes with a dynamic grid lambda.

> **Avoid when**: Any time the launch grid is computed statically using the old XBLOCK value — this will cause redundant work proportional to the tile size ratio and always regress performance.

---

## Cumulative Progress

```
Baseline:  12290 ns  (1.000×)

Final best: 12290 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
