# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 531680 ns — **Best latency**: 531680 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_048/kernel_0_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (861056,1,1), Waves/SM: 407.70
- **Registers/thread**: 16
- **Duration**: 531680 ns
- **SM Throughput**: 54.06 % | **Memory**: 32.73 % | **DRAM**: 32.73 %
- **Achieved Occupancy**: 53.21 %
- **IPC Active**: 2.17 inst/cycle | **No Eligible**: 45.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_xblock_tile_size_with_autotune | 0.597× | 0.573× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 531680 ns | 927460 ns | +395780 ns (+74.4%) |
| SM Throughput (%) | 54.06 % | 92.59 % | +38.53 pp |
| Memory Throughput (%) | 32.73 % | 30.72 % | -2.01 pp |
| DRAM Throughput (%) | 32.73 % | 18.77 % | -13.96 pp |
| Achieved Occupancy (%) | 53.21 % | 88.15 % | +34.94 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 |
| Waves/SM | 407.70 | 407.70 | +0.00 |
| No Eligible (%) | 45.69 % | 7.30 % | -38.39 pp |
| IPC Active | 2.17 inst/cycle | 3.71 inst/cycle | +1.54 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.573×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) alongside a mask change (xmask = xindex < xnumel instead of tl.full([XBLOCK], True)) caused executed instructions to nearly triple (227M → 675M) and duration to increase 1.74x despite improving occupancy (53% → 88%) and scheduler utilization (54% → 93%). The register count per thread rose from 16 to 28, halving the Block Limit Registers from 32 to 16, which constrained occupancy gains. The instruction explosion — likely from predicated/masked loads on a tensor whose size is already a multiple of the original XBLOCK — dominated any latency-hiding benefit from larger tiles. When the original kernel uses tl.full([XBLOCK], True) (no masking needed), introducing xmask = xindex < xnumel adds unnecessary predication overhead that scales with the larger tile sizes chosen by autotune.

> **Applicable when**: Never — this specific combination is always harmful. The general direction of increasing XBLOCK may still be valid but requires correct grid lambda and no spurious mask introduction.

> **Avoid when**: When the original kernel already uses tl.full([XBLOCK], True) indicating xnumel is a multiple of XBLOCK (no boundary masking needed). Do not introduce xmask = xindex < xnumel in that case as it adds predication overhead. Also, the grid was not updated to use lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),) — the fixed grid of cdiv(xnumel, 128) blocks was kept, meaning autotune selected larger XBLOCK but the grid remained sized for XBLOCK=128, causing each block to process far more elements than intended and massively inflating instruction count.

---

## Cumulative Progress

```
Baseline:  531680 ns  (1.000×)

Final best: 531680 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
