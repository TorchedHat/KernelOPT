# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7650 ns — **Best latency**: 7650 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7650 ns
- **SM Throughput**: 39.14 % | **Memory**: 17.14 % | **DRAM**: 10.78 %
- **Achieved Occupancy**: 57.13 %
- **IPC Active**: 2.28 inst/cycle | **No Eligible**: 42.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_tile_size | 0.977× | 0.262× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.992× | 0.259× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7650 ns | 29150 ns | +21500 ns (+281.0%) | 29500 ns | +21850 ns (+285.6%) |
| SM Throughput (%) | 39.14 % | 82.04 % | +42.90 pp | 81.27 % | +42.13 pp |
| Memory Throughput (%) | 17.14 % | 15.91 % | -1.23 pp | 15.66 % | -1.48 pp |
| DRAM Throughput (%) | 10.78 % | 2.81 % | -7.97 pp | 2.78 % | -8.00 pp |
| Achieved Occupancy (%) | 57.13 % | 84.16 % | +27.03 pp | 84.35 % | +27.22 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 3.71 | 14.85 | +11.14 | 14.85 | +11.14 |
| No Eligible (%) | 42.48 % | 11.88 % | -30.60 pp | 12.18 % | -30.30 pp |
| IPC Active | 2.28 inst/cycle | 3.52 inst/cycle | +1.24 | 3.49 inst/cycle | +1.21 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_and_autotune_tile_size` (0.262×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) caused autotune to select XBLOCK=512, which doubled registers per thread from 17 to 32 and increased executed instructions by 8x (2.3M to 18.7M), inflating kernel duration from 7.65 µs to 29.15 µs (3.8x slower). Although occupancy improved (57% → 84%) and scheduler idle cycles dropped (42% → 12%), the massive instruction count increase — likely from predicated masking overhead introduced by changing from `tl.full([XBLOCK], True, tl.int1)` to `xindex < xnumel` — dominated all other gains. For kernels where xnumel is exactly divisible by the original XBLOCK, replacing the unconditional mask with a bounds-check mask adds unnecessary predication overhead at larger tile sizes.

> **Applicable when**: Never — this specific combination caused severe regression. The autotune + larger XBLOCK strategy may be valid in other contexts but not when: (1) the grid is hardcoded and not updated to use lambda meta, and (2) an unconditional mask is replaced with a bounds-check mask unnecessarily.

> **Avoid when**: Do not replace `tl.full([XBLOCK], True, tl.int1)` with `xindex < xnumel` when xnumel is exactly divisible by all candidate XBLOCK values — this introduces unnecessary predication. Do not add @triton.autotune without also updating the grid launch to use `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — a fixed grid with a larger autotuned XBLOCK will launch too few blocks and process wrong data. Do not increase XBLOCK when registers per thread are already near the SM limit, as doubling block size can double register pressure and cause spilling.

---

## Cumulative Progress

```
Baseline:  7650 ns  (1.000×)

Final best: 7650 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
