# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42940 ns — **Best latency**: 42940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_039/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 42940 ns
- **SM Throughput**: 61.71 % | **Memory**: 31.17 % | **DRAM**: 19.52 %
- **Achieved Occupancy**: 57.19 %
- **IPC Active**: 2.60 inst/cycle | **No Eligible**: 33.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_occupancy_and_l1tex_reuse | 0.614× | 0.598× | ✓ |  |
| B | increase_xblock_with_autotune_for_better_occupancy_and_latency_hiding | 0.998× | 0.167× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42940 ns | 71810 ns | +28870 ns (+67.2%) | 257950 ns | +215010 ns (+500.7%) |
| SM Throughput (%) | 61.71 % | 83.12 % | +21.41 pp | 86.29 % | +24.58 pp |
| Memory Throughput (%) | 31.17 % | 28.44 % | -2.73 pp | 20.15 % | -11.02 pp |
| DRAM Throughput (%) | 19.52 % | 11.63 % | -7.89 pp | 3.22 % | -16.30 pp |
| Achieved Occupancy (%) | 57.19 % | 89.06 % | +31.87 pp | 86.33 % | +29.14 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 124.12 | +93.09 |
| No Eligible (%) | 33.69 % | 14.25 % | -19.44 pp | 12.94 % | -20.75 pp |
| IPC Active | 2.60 inst/cycle | 3.42 inst/cycle | +0.82 | 3.48 inst/cycle | +0.88 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size_for_occupancy_and_l1tex_reuse` (0.598×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) increased registers per thread from 22 to 32 and caused a 2.24x explosion in executed instructions (20.9M → 46.9M), despite improving occupancy (57% → 89%) and eligible warps per scheduler (1.39 → 7.48). The net effect was a 67% longer kernel duration (42.94 µs → 71.81 µs). The instruction count increase suggests the autotune selected a config that introduced significant overhead—likely from the added xmask guard on every load/store (previously xmask=True was elided), combined with larger tiles executing more predicated-off work. When the total element count is a power-of-2 and the original tile size divides it evenly, replacing `tl.full([XBLOCK], True, tl.int1)` with a dynamic `xindex < xnumel` mask adds unnecessary per-element predication overhead that compounds at larger XBLOCK sizes.

> **Applicable when**: Kernel has low achieved occupancy (<60%) and high L1TEX stall share (>40% of CPI), with many small CTAs doing broadcast loads indexed by a modulo pattern. Original XBLOCK divides xnumel evenly so xmask=True is valid.

> **Avoid when**: Do not add dynamic xmask (xindex < xnumel) when the original kernel uses xmask=True (i.e., xnumel is exactly divisible by XBLOCK). The mask predication overhead compounds at larger XBLOCK sizes and causes instruction count explosion. Do not autotune XBLOCK upward when the autotune overhead or register pressure increase (22→32 regs/thread) will reduce Block Limit Registers and negate occupancy gains. Avoid when the kernel is already compute-bound at the chosen tile size.

---

## Cumulative Progress

```
Baseline:  42940 ns  (1.000×)

Final best: 42940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
