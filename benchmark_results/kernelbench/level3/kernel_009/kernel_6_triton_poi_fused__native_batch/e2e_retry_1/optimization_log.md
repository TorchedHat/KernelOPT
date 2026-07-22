# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4580 ns — **Best latency**: 4580 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 22
- **Duration**: 4580 ns
- **SM Throughput**: 26.31 % | **Memory**: 14.50 % | **DRAM**: 14.50 %
- **Achieved Occupancy**: 62.67 %
- **IPC Active**: 2.08 inst/cycle | **No Eligible**: 52.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_autotune_eliminate_partial_wave | 0.900× | 0.837× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.883× | 0.809× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4580 ns | 5470 ns | +890 ns (+19.4%) | 5660 ns | +1080 ns (+23.6%) |
| SM Throughput (%) | 26.31 % | 49.07 % | +22.76 pp | 48.40 % | +22.09 pp |
| Memory Throughput (%) | 14.50 % | 14.65 % | +0.15 pp | 14.43 % | -0.07 pp |
| DRAM Throughput (%) | 14.50 % | 12.06 % | -2.44 pp | 11.63 % | -2.87 pp |
| Achieved Occupancy (%) | 62.67 % | 74.75 % | +12.08 pp | 74.58 % | +11.91 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 1.48 | +0.00 |
| No Eligible (%) | 52.38 % | 21.30 % | -31.08 pp | 21.39 % | -30.99 pp |
| IPC Active | 2.08 inst/cycle | 3.11 inst/cycle | +1.03 | 3.12 inst/cycle | +1.04 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_autotune_eliminate_partial_wave` (0.837×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) to eliminate a partial-wave tail caused a 19% regression despite improving scheduler eligibility (No Eligible: 52% → 21%) and compute throughput (26% → 49%). The regression was driven by a 2.2x increase in executed instructions (965K → 2129K) and a 10-point drop in L1 hit rate (70% → 58%), indicating that the autotune overhead and/or the introduction of xmask guards (replacing tl.full([XBLOCK], True) with xindex < xnumel) added substantial instruction count without reducing memory latency enough to compensate. Additionally, registers per thread increased from 22 to 32, reducing block-level parallelism. When the kernel is already latency-bound with a partial wave, adding autotune with masked loads can inflate instruction count more than the wave-elimination saves.

> **Applicable when**: Kernel has a genuine partial-wave tail AND the xnumel is NOT exactly divisible by the larger XBLOCK values, requiring masks. Only beneficial when instruction count does not increase significantly and register pressure stays flat.

> **Avoid when**: The original kernel uses tl.full([XBLOCK], True) masks (meaning xnumel is exactly divisible by XBLOCK), because switching to xindex < xnumel masks adds unnecessary predication overhead and inflates instruction count. Also avoid when registers per thread would increase (e.g., from 22 to 32), as this reduces block-level parallelism and can negate wave-elimination gains. Do not apply when the kernel is already latency-bound with low memory throughput (<20% DRAM), as the bottleneck is not the partial wave.

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.809×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to a pointwise kernel that already has a near-integer wave count caused a 24% regression. The autotuner selected a config that increased registers per thread from 22 to 32, which reduced Block Limit Registers from 21 to 16 blocks/SM. This caused executed instructions to balloon from ~966K to ~2.1M (2.2x more instructions), indicating the autotuner chose a suboptimal config that added overhead without proportional benefit. Although scheduler eligibility improved (no-eligible dropped from 52% to 21%), the extra instruction count and register pressure dominated, increasing duration from 4.58us to 5.66us. The grid was also not updated to use meta['XBLOCK'] in the lambda — it remained hardcoded at cdiv(xnumel, 128) — so the autotuner's XBLOCK selection had no effect on grid size, defeating the partial-wave elimination goal entirely.

> **Applicable when**: Never in this form. The correct form (grid as lambda meta) should be used when: LaunchConfiguration rule fires with >30% speedup_pct, waves_per_sm is non-integer with large fractional part, and the kernel is pure pointwise with no shared memory.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK if the grid launch is not simultaneously updated to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Without this fix, the autotuner selects a larger XBLOCK that increases register pressure and instruction count without reducing grid size, causing regression. Also avoid when registers per thread are already near the block limit register boundary (22 regs/thread with limit=21 blocks/SM), as larger configs will push registers higher and reduce occupancy.

---

## Cumulative Progress

```
Baseline:  4580 ns  (1.000×)

Final best: 4580 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
