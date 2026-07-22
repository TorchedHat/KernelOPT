# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22720 ns — **Best latency**: 22720 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_tanh_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_2_triton_poi_fused_addmm_tanh_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22720 ns
- **SM Throughput**: 37.20 % | **Memory**: 20.10 % | **DRAM**: 15.11 %
- **Achieved Occupancy**: 45.00 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_tile_utilization | 0.721× | 0.713× | ✓ |  |
| B | increase_xblock_tile_size_to_reduce_latency_and_improve_cache_reuse | 1.002× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 31870 ns | +9150 ns (+40.3%) | 22720 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.20 % | 86.00 % | +48.80 pp | 36.92 % | -0.28 pp |
| Memory Throughput (%) | 20.10 % | 19.06 % | -1.04 pp | 20.01 % | -0.09 pp |
| DRAM Throughput (%) | 15.11 % | 10.78 % | -4.33 pp | 15.12 % | +0.01 pp |
| Achieved Occupancy (%) | 45.00 % | 86.26 % | +41.26 pp | 45.06 % | +0.06 pp |
| Registers/Thread | 16.00 register/thread | 27.00 register/thread | +11 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.73 % | 7.92 % | -50.81 pp | 58.13 % | -0.60 pp |
| IPC Active | 1.66 inst/cycle | 3.67 inst/cycle | +2.01 | 1.66 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_better_tile_utilization` (0.713×)

Increasing XBLOCK via autotune for a pointwise kernel with a broadcast bias pattern dramatically increased compute throughput (37% → 86%) and occupancy (45% → 86%), but the total executed instruction count exploded 3.2x (6.7M → 21.6M) and registers per thread grew from 16 to 27, causing Block Limit Registers to halve from 32 to 16. The net effect was a 40% longer kernel duration (22.7 µs → 31.9 µs) despite better scheduler utilization, because the larger tile size forced each thread to process more elements with more register pressure, and the autotune grid lambda was not actually applied — the grid remained hardcoded to cdiv(xnumel, 128), so the kernel still launched 32,768 CTAs each with a larger XBLOCK, causing redundant/out-of-bounds masked work. When the grid is not updated to match the new XBLOCK, larger tile configs produce more total work rather than less.

> **Applicable when**: Pointwise kernels with low achieved occupancy (<50%), high scheduler 'No Eligible' % (>50%), and CPIStall from L1TEX — suggesting over-decomposed tiles.

> **Avoid when**: When the grid is not updated to use `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` — leaving the grid hardcoded while increasing XBLOCK causes each CTA to process more elements than intended, exploding instruction count and register pressure without reducing CTA count. Also avoid when register pressure from larger tiles would halve Block Limit Registers.

---

## Cumulative Progress

```
Baseline:  22720 ns  (1.000×)

Final best: 22720 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_addmm_tanh_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -18,8 +19,6 @@

     tmp2 = tmp0 + tmp1
     tmp3 = libdevice.tanh(tmp2)
     tl.store(in_out_ptr0 + (x2), tmp3, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
