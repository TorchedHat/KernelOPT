# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7490 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_44_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 18
- **Duration**: 7490 ns
- **SM Throughput**: 49.90 % | **Memory**: 24.60 % | **DRAM**: 24.60 %
- **Achieved Occupancy**: 76.48 %
- **IPC Active**: 2.72 inst/cycle | **No Eligible**: 29.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_larger_xblock_autotune | 0.847× | 0.775× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.848× | 0.775× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 9660 ns | +2170 ns (+29.0%) | 9660 ns | +2170 ns (+29.0%) |
| SM Throughput (%) | 49.90 % | 62.79 % | +12.89 pp | 61.90 % | +12.00 pp |
| Memory Throughput (%) | 24.60 % | 19.13 % | -5.47 pp | 19.13 % | -5.47 pp |
| DRAM Throughput (%) | 24.60 % | 19.13 % | -5.47 pp | 19.13 % | -5.47 pp |
| Achieved Occupancy (%) | 76.48 % | 78.95 % | +2.47 pp | 78.58 % | +2.10 pp |
| Registers/Thread | 18.00 register/thread | 26.00 register/thread | +8 | 26.00 register/thread | +8 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 29.76 % | 22.57 % | -7.19 pp | 22.46 % | -7.30 pp |
| IPC Active | 2.72 inst/cycle | 3.06 inst/cycle | +0.34 | 3.08 inst/cycle | +0.36 |


### Key Learnings

#### ❌ Plan A: `eliminate_partial_wave_via_larger_xblock_autotune` (0.775×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel that had a partial-wave penalty caused a 29% slowdown instead of the expected speedup. The profiling data shows that executed instructions nearly doubled (2.94M → 4.66M) and registers per thread increased from 18 to 26, indicating the autotune selected a larger XBLOCK that increased register pressure and instruction count without proportionally reducing kernel launches. The grid size remained 5880 (autotune did not override the hard-coded grid lambda), and the compute throughput rose (49.9% → 62.79%) while memory throughput fell (24.6% → 19.13%), suggesting the kernel became compute-bound with more work per thread but no net benefit from wave reduction.

> **Applicable when**: Never: this specific combination of autotune + mismatched static grid caused regression. In general, autotune for XBLOCK is only beneficial when the grid lambda is also updated to use META['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune for XBLOCK when the grid is still computed with a hard-coded block size (e.g., triton.cdiv(xnumel, 128)) rather than a lambda using META['XBLOCK']. This mismatch causes the kernel to launch with the old grid size but execute with a different XBLOCK, leading to incorrect behavior or increased instruction count. Also avoid when register count is already near limits, as larger XBLOCK configs increase register pressure (18 → 26 regs/thread observed here), which can reduce occupancy and increase stalls.

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)

Final best: 7490 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
