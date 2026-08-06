# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244220 ns — **Best latency**: 244220 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_4_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244220 ns
- **SM Throughput**: 34.98 % | **Memory**: 32.02 % | **DRAM**: 32.02 %
- **Achieved Occupancy**: 43.79 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_larger_xblock_reduce_grid_overhead | 0.793× | 0.803× | ✓ |  |
| B | increase_xblock_with_autotune | 0.877× | 0.886× | ✓ |  |
| C | add_triton_autotune | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_larger_x... | Δ | increase_xblock_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244220 ns | 304100 ns | +59880 ns (+24.5%) | 275710 ns | +31490 ns (+12.9%) | 244350 ns | +130 ns (+0.1%) |
| SM Throughput (%) | 34.98 % | 55.06 % | +20.08 pp | 38.25 % | +3.27 pp | 37.66 % | +2.68 pp |
| Memory Throughput (%) | 32.02 % | 25.41 % | -6.61 pp | 28.05 % | -3.97 pp | 31.97 % | -0.05 pp |
| DRAM Throughput (%) | 32.02 % | 25.41 % | -6.61 pp | 28.05 % | -3.97 pp | 31.97 % | -0.05 pp |
| Achieved Occupancy (%) | 43.79 % | 36.12 % | -7.67 pp | 35.23 % | -8.56 pp | 44.42 % | +0.63 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.40 % | 46.21 % | -18.19 pp | 60.08 % | -4.32 pp | 62.17 % | -2.23 pp |
| IPC Active | 1.42 inst/cycle | 2.14 inst/cycle | +0.72 | 1.57 inst/cycle | +0.15 | 1.51 inst/cycle | +0.09 |


### Key Learnings

#### ❌ Plan A: `autotune_larger_xblock_reduce_grid_overhead` (0.803×)

Adding @triton.autotune with larger XBLOCK configs (1024–4096) increased registers per thread from 16 to 28, which reduced achieved occupancy from 43.8% to 36.1% (Block Limit Registers dropped from 32 to 16 blocks/SM). Although warp cycles per issued instruction improved (19.3→10.5) and scheduler idle rate fell (64.4%→46.2%), the instruction count nearly doubled (67M→126M) and duration increased 24% (244→304 µs). The net effect was a 20% regression: the larger XBLOCK forced more register pressure that cut occupancy and increased total work, outweighing the latency-hiding benefit.

> **Applicable when**: Never: this specific pattern caused regression. In general, larger XBLOCK autotune is worth trying when registers_per_thread is low (≤16) and occupancy is register-bound, not when the kernel is already at moderate occupancy with low register count.

> **Avoid when**: Kernel has low register count (≤16 registers/thread) and moderate achieved occupancy (~40%+). Increasing XBLOCK with complex index arithmetic (multiple modulo/division operations per element) inflates register pressure, cuts occupancy further, and increases total instruction count — net regression even if per-warp stall cycles improve. Also avoid when the grid is already large (>100K CTAs) and the bottleneck is L1TEX stalls rather than grid launch overhead.

---

## Cumulative Progress

```
Baseline:  244220 ns  (1.000×)

Final best: 244220 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
