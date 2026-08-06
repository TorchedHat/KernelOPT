# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6980 ns — **Best latency**: 6980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_227_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6983,1,1), Waves/SM: 3.31
- **Registers/thread**: 22
- **Duration**: 6980 ns
- **SM Throughput**: 43.23 % | **Memory**: 16.49 % | **DRAM**: 10.60 %
- **Achieved Occupancy**: 59.27 %
- **IPC Active**: 2.47 inst/cycle | **No Eligible**: 34.87 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.951× | 0.774× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.998× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 9020 ns | +2040 ns (+29.2%) | 7140 ns | +160 ns (+2.3%) |
| SM Throughput (%) | 43.23 % | 63.37 % | +20.14 pp | 42.30 % | -0.93 pp |
| Memory Throughput (%) | 16.49 % | 17.11 % | +0.62 pp | 16.18 % | -0.31 pp |
| DRAM Throughput (%) | 10.60 % | 8.18 % | -2.42 pp | 10.32 % | -0.28 pp |
| Achieved Occupancy (%) | 59.27 % | 80.97 % | +21.70 pp | 61.90 % | +2.63 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.31 | 3.31 | +0.00 | 3.31 | +0.00 |
| No Eligible (%) | 34.87 % | 17.52 % | -17.35 pp | 34.85 % | -0.02 pp |
| IPC Active | 2.47 inst/cycle | 3.21 inst/cycle | +0.74 | 2.54 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.774×)

Adding @triton.autotune over larger XBLOCK values (512/1024/2048) while leaving the launch grid hardcoded to cdiv(xnumel, 128) caused the kernel to run with a larger tile size but the same number of blocks as before, resulting in each block processing far more elements than intended and nearly doubling the executed instruction count (2.37M → 4.50M). The register count increased from 22 to 32 per thread (due to the larger tile unrolling), reducing the block limit from 21 to 16 and increasing duration from 6.98µs to 9.02µs despite higher occupancy (59% → 81%). The implementation flaw was that the grid was not updated to use `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — autotune injected the larger XBLOCK into the kernel body but the grid remained fixed at 6983 blocks, causing massive redundant work.

> **Applicable when**: Never apply this exact pattern (fixed grid + autotune XBLOCK). The corrected version (lambda grid) is applicable when: achieved occupancy < 70%, LaunchConfiguration rule fires with >15% partial-wave overhead, kernel is pure pointwise with no shared memory, and register count is low (<30/thread).

> **Avoid when**: Do not use a fixed hardcoded grid when adding @triton.autotune over XBLOCK — the grid must always be a lambda that reads meta['XBLOCK']. Also avoid increasing XBLOCK when register pressure is already high (>40/thread), as larger tiles increase register usage and reduce block-level occupancy.

---

## Cumulative Progress

```
Baseline:  6980 ns  (1.000×)

Final best: 6980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
