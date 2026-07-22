# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4830 ns — **Best latency**: 4830 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_4_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 1.45
- **Registers/thread**: 28
- **Duration**: 4830 ns
- **SM Throughput**: 34.73 % | **Memory**: 39.79 % | **DRAM**: 10.45 %
- **Achieved Occupancy**: 72.14 %
- **IPC Active**: 2.45 inst/cycle | **No Eligible**: 39.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tl_hints_and_tile_reshape | 0.824× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4830 ns | 4860 ns | +30 ns (+0.6%) |
| SM Throughput (%) | 34.73 % | 33.30 % | -1.43 pp |
| Memory Throughput (%) | 39.79 % | 37.90 % | -1.89 pp |
| DRAM Throughput (%) | 10.45 % | 10.42 % | -0.03 pp |
| Achieved Occupancy (%) | 72.14 % | 70.48 % | -1.66 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 1.45 | 1.45 | +0.00 |
| No Eligible (%) | 39.32 % | 38.25 % | -1.07 pp |
| IPC Active | 2.45 inst/cycle | 2.35 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_xblock32_yblock32_no_autotune | 0.799× | 1.000× | ✓ |  |
| B | static_tile_reshape_eliminate_partial_wave | 0.976× | 1.013× | ✓ |  |
| C | coalesced_store_via_dimension_swap | 0.997× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_xb... | Δ | static_tile_resha... | Δ | coalesced_store_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4830 ns | 4830 ns | +0 ns (+0.0%) | 4770 ns | -60 ns (-1.2%) | 4800 ns | -30 ns (-0.6%) |
| SM Throughput (%) | 34.73 % | 34.66 % | -0.07 pp | 32.38 % | -2.35 pp | 36.12 % | +1.39 pp |
| Memory Throughput (%) | 39.79 % | 39.75 % | -0.04 pp | 37.09 % | -2.70 pp | 39.50 % | -0.29 pp |
| DRAM Throughput (%) | 10.45 % | 10.43 % | -0.02 pp | 10.54 % | +0.09 pp | 10.50 % | +0.05 pp |
| Achieved Occupancy (%) | 72.14 % | 72.43 % | +0.29 pp | 71.48 % | -0.66 pp | 69.56 % | -2.58 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.45 | 1.45 | +0.00 | 1.45 | +0.00 | 1.45 | +0.00 |
| No Eligible (%) | 39.32 % | 37.09 % | -2.23 pp | 39.43 % | +0.11 pp | 39.77 % | +0.45 pp |
| IPC Active | 2.45 inst/cycle | 2.45 inst/cycle | +0.00 | 2.31 inst/cycle | -0.14 | 2.42 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  4830 ns  (1.000×)

Final best: 4830 ns  (1.0000×)
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
 def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,9 @@

     y3 = yindex
     y0 = (yindex % 96)
     y1 = yindex // 96
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 96*x2 + 2400*y1), tmp0, xmask)
-
-
+    x2_hint = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK) + xoffset, XBLOCK), XBLOCK)[None, :]
+    tmp0 = tl.load(in_ptr0 + (x2_hint + 25*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 96*x2 + 2400*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
