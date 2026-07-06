# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 6080 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.59 % | **Memory**: 40.68 % | **DRAM**: 20.26 %
- **Achieved Occupancy**: 55.28 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy | 0.795× | 0.335× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6080 ns | 18140 ns | +12060 ns (+198.4%) |
| SM Throughput (%) | 16.59 % | 14.00 % | -2.59 pp |
| Memory Throughput (%) | 40.68 % | 36.64 % | -4.04 pp |
| DRAM Throughput (%) | 20.26 % | 6.78 % | -13.48 pp |
| Achieved Occupancy (%) | 55.28 % | 58.54 % | +3.26 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.91 | +0.17 |
| No Eligible (%) | 71.21 % | 76.37 % | +5.16 pp |
| IPC Active | 1.13 inst/cycle | 0.94 inst/cycle | -0.19 |


### Key Learnings

#### ❌ Plan B: `autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy` (0.335×)

Adding @triton.autotune with large tile configs (YBLOCK=32, XBLOCK up to 1024) caused a 3x regression. The larger tiles increased dynamic shared memory per block from 4.1 KB to 16.4 KB, cutting Block Limit Shared Mem from 26 to 13 and reducing theoretical occupancy from 100% to 81.25%. Critically, the instruction count more than doubled (771K→1.76M) and the dominant stall shifted from L1TEX latency to MIO queue saturation (34.5% of cycles), indicating the autotune overhead and larger tile footprint introduced more shared memory traffic rather than eliminating it. The uncoalesced excess sector ratio actually worsened from 23% to 52%, meaning the tile change did not fix the store coalescing problem as predicted.

> **Applicable when**: Never: this specific combination of autotune + large YBLOCK for a small-ynumel (30) layout permutation kernel caused regression. The pattern is a cautionary anti-pattern.

> **Avoid when**: Do NOT apply large YBLOCK autotune configs when: (1) ynumel is very small (e.g. 30) so YBLOCK > ynumel causes wasted threads and mask overhead; (2) the kernel already has low dynamic shared memory usage and increasing tile size would push Block Limit Shared Mem below the current block limit, reducing theoretical occupancy; (3) the store pattern has a fixed stride in the y-dimension (y0 = yindex % C) — larger YBLOCK does not fix inter-thread store stride within a warp, it only changes which y-values are co-located in a CTA; (4) the autotune overhead itself (extra smem for autotuning bookkeeping) is significant relative to kernel duration (here 6 us baseline).

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)

Final best: 6080 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
