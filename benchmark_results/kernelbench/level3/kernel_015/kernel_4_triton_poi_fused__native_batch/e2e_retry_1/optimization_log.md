# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41380 ns — **Best latency**: 41380 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 17
- **Duration**: 41380 ns
- **SM Throughput**: 56.92 % | **Memory**: 24.74 % | **DRAM**: 18.39 %
- **Achieved Occupancy**: 55.12 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_latency_hiding | 0.864× | 0.694× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.993× | 0.214× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41380 ns | 59650 ns | +18270 ns (+44.2%) | 193440 ns | +152060 ns (+367.5%) |
| SM Throughput (%) | 56.92 % | 86.02 % | +29.10 pp | 89.78 % | +32.86 pp |
| Memory Throughput (%) | 24.74 % | 25.31 % | +0.57 pp | 19.27 % | -5.47 pp |
| DRAM Throughput (%) | 18.39 % | 12.72 % | -5.67 pp | 3.92 % | -14.47 pp |
| Achieved Occupancy (%) | 55.12 % | 87.52 % | +32.40 pp | 55.29 % | +0.17 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 48.00 register/thread | +31 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 95.03 | +65.33 |
| No Eligible (%) | 39.72 % | 10.97 % | -28.75 pp | 9.21 % | -30.51 pp |
| IPC Active | 2.40 inst/cycle | 3.56 inst/cycle | +1.16 | 3.63 inst/cycle | +1.23 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_latency_hiding` (0.214×)

Adding a @triton.autotune decorator with large XBLOCK configs (512–2048) caused the compiler to select a config that increased registers per thread from 17 to 48, triggering Block Limit Registers to drop from 21 to 5 blocks/SM and capping theoretical occupancy at 62.5%. Despite eligible warps per scheduler improving (1.17→5.26) and warp cycles per instruction decreasing (14.46→9.69), the 7.5x increase in executed instructions (18M→137M) caused duration to balloon from 41 µs to 193 µs — a 4.7x regression. The autotune decorator itself, combined with the introduction of a non-trivial xmask guard (replacing tl.full True mask), appears to have caused the compiler to generate a much more instruction-heavy kernel body for the winning config.

> **Applicable when**: Never apply this exact pattern (autotune with large XBLOCK + replacing None mask with xindex<xnumel guard) when the baseline kernel uses trivial masks (None/tl.full True) because xnumel is exactly divisible by XBLOCK.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when: (1) the baseline kernel uses None/trivial masks that the compiler can optimize away — replacing them with xindex<xnumel forces bounds-check codegen that dramatically increases register pressure and instruction count; (2) register count is already low (≤20 regs/thread) and the kernel body is simple — autotune may select a config that causes register spill, collapsing Block Limit Registers and negating any latency-hiding benefit; (3) xnumel is exactly divisible by all candidate XBLOCK values, making the mask change unnecessary.

---

## Cumulative Progress

```
Baseline:  41380 ns  (1.000×)

Final best: 41380 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
