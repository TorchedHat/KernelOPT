# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12260 ns — **Best latency**: 12260 ns (**1.0000×** vs baseline)
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
- **Duration**: 12260 ns
- **SM Throughput**: 46.09 % | **Memory**: 20.33 % | **DRAM**: 13.37 %
- **Achieved Occupancy**: 57.60 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 41.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.848× | 0.724× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.746× | 0.581× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12260 ns | 16930 ns | +4670 ns (+38.1%) | 21090 ns | +8830 ns (+72.0%) |
| SM Throughput (%) | 46.09 % | 73.73 % | +27.64 pp | 78.68 % | +32.59 pp |
| Memory Throughput (%) | 20.33 % | 26.94 % | +6.61 pp | 25.09 % | +4.76 pp |
| DRAM Throughput (%) | 13.37 % | 9.69 % | -3.68 pp | 7.77 % | -5.60 pp |
| Achieved Occupancy (%) | 57.60 % | 79.35 % | +21.75 pp | 82.73 % | +25.13 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 24.00 register/thread | +7 |
| Waves/SM | 7.42 | 14.85 | +7.43 | 14.85 | +7.43 |
| No Eligible (%) | 41.64 % | 16.45 % | -25.19 pp | 13.10 % | -28.54 pp |
| IPC Active | 2.31 inst/cycle | 3.33 inst/cycle | +1.02 | 3.46 inst/cycle | +1.15 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.724×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel that has a fixed xnumel exactly divisible by the original block size caused a 2.12x increase in executed instructions (4.58M → 9.72M) and a 38% increase in kernel duration (12.26 µs → 16.93 µs). The regression occurred because the autotune implementation also changed the xmask from `tl.full([XBLOCK], True, tl.int1)` (no masking) to `xindex < xnumel` (conditional masking), which introduces predicated-off threads and additional branch/mask instructions for every load and store. Although occupancy improved (57.6% → 79.4%) and no-eligible-warp cycles dropped (41.6% → 16.5%), the extra instruction overhead from masking more than cancelled the latency-hiding benefit, resulting in a net slowdown.

> **Applicable when**: Never apply this specific pattern (autotune + mask introduction on a kernel where xnumel is already divisible by the original XBLOCK). The masking overhead dominates any occupancy gains.

> **Avoid when**: Do NOT add conditional xmask (xindex < xnumel) when the original kernel uses tl.full([XBLOCK], True, tl.int1) and xnumel is exactly divisible by the original XBLOCK — this introduces unnecessary predication overhead. Also do NOT add autotune without wiring the grid to a lambda using meta['XBLOCK']; the fixed grid at the old XBLOCK means larger configs process out-of-bounds elements unnecessarily.

#### ❌ Plan B: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.581×)

Increasing XBLOCK from 128 to 256 (via autotune) improved scheduler eligibility (No Eligible dropped from 41.6% to 13.1%) and raised achieved occupancy (57.6% → 82.7%), but caused a 1.72x slowdown. The root cause is that the larger block size increased registers per thread from 17 to 24, which reduced the block limit by registers from 21 to 10, and the total executed instruction count ballooned from 4.6M to 12.9M — nearly 3x more instructions for the same work, indicating the autotune selected a config that introduced significant overhead (likely from the added xmask predication on a kernel where xnumel was already a multiple of XBLOCK, or from autotune warmup cost dominating a short kernel). The kernel duration increased from 12.26µs to 21.09µs despite better scheduler utilization, showing that instruction count inflation and register pressure outweighed latency hiding benefits.

> **Applicable when**: Kernel has very low eligible warps per scheduler (<2), high no-eligible-warp percentage (>40%), and L1TEX scoreboard stalls dominating CPI, with low registers per thread (<20) leaving headroom for larger blocks.

> **Avoid when**: The kernel is already short-running (<15µs) where autotune warmup cost dominates; when increasing block size raises registers per thread significantly (e.g., 17→24), reducing block limit by registers; when the grid lambda is not updated to match the autotune XBLOCK (causing the grid to remain fixed at the old block size while autotune injects a different XBLOCK, leading to incorrect coverage or wasted threads); when xnumel is already a multiple of the original XBLOCK making masking unnecessary overhead.

---

## Cumulative Progress

```
Baseline:  12260 ns  (1.000×)

Final best: 12260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
