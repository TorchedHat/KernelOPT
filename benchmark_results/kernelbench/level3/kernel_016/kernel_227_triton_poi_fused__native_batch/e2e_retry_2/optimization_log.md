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
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_227_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6983,1,1), Waves/SM: 3.31
- **Registers/thread**: 22
- **Duration**: 6980 ns
- **SM Throughput**: 43.12 % | **Memory**: 16.47 % | **DRAM**: 10.60 %
- **Achieved Occupancy**: 64.64 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 34.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.904× | 0.268× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.928× | 0.774× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 26020 ns | +19040 ns (+272.8%) | 9020 ns | +2040 ns (+29.2%) |
| SM Throughput (%) | 43.12 % | 79.11 % | +35.99 pp | 63.34 % | +20.22 pp |
| Memory Throughput (%) | 16.47 % | 15.76 % | -0.71 pp | 17.09 % | +0.62 pp |
| DRAM Throughput (%) | 10.60 % | 2.83 % | -7.77 pp | 8.20 % | -2.40 pp |
| Achieved Occupancy (%) | 64.64 % | 54.73 % | -9.91 pp | 80.66 % | +16.02 pp |
| Registers/Thread | 22.00 register/thread | 48.00 register/thread | +26 | 32.00 register/thread | +10 |
| Waves/SM | 3.31 | 10.58 | +7.27 | 3.31 | +0.00 |
| No Eligible (%) | 34.58 % | 13.60 % | -20.98 pp | 17.76 % | -16.82 pp |
| IPC Active | 2.55 inst/cycle | 3.45 inst/cycle | +0.90 | 3.27 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_eliminate_partial_wave` (0.268×)

Adding @triton.autotune with large XBLOCK values (512–2048) to a kernel that previously used XBLOCK=128 caused registers per thread to jump from 22 to 48, collapsing theoretical occupancy from 100% to 62.5% (Block Limit Registers dropped from 21 to 5 blocks/SM) and increasing waves per SM from 3.3 to 10.6. The autotune grid was left fixed at the original small-block grid size (cdiv(xnumel, 128)) rather than being updated to a lambda using meta['XBLOCK'], so the kernel ran the same number of blocks but with far more work per block and severe register pressure, resulting in 6.8x more executed instructions and 3.7x longer duration. When increasing XBLOCK via autotune, the launch grid must be a lambda over meta['XBLOCK']; a static grid computed with the old small block size defeats the purpose and causes massive over-subscription.

> **Applicable when**: Never apply this specific pattern (static grid + autotune XBLOCK). The corrected pattern (lambda grid) is applicable when LaunchConfiguration rule fires with >15% partial-wave waste and registers per thread are low (<30), leaving headroom to increase block size without register pressure.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when the kernel already uses 22+ registers per thread, as doubling XBLOCK tends to push register count above 32–48, cutting theoretical occupancy by 37–62% and negating any partial-wave savings. Also do not use a static grid when adding autotune — always use a lambda grid.

---

## Cumulative Progress

```
Baseline:  6980 ns  (1.000×)

Final best: 6980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
