# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31230 ns — **Best latency**: 31230 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_7_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (47040,1,1), Waves/SM: 22.27
- **Registers/thread**: 16
- **Duration**: 31230 ns
- **SM Throughput**: 25.20 % | **Memory**: 19.43 % | **DRAM**: 16.11 %
- **Achieved Occupancy**: 36.80 %
- **IPC Active**: 1.09 inst/cycle | **No Eligible**: 72.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_pointwise | 0.986× | 0.580× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.915× | 0.948× | ✓ |  |
| C | increase_xblock_for_vectorized_memory_access | 0.912× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31230 ns | 53860 ns | +22630 ns (+72.5%) | 32960 ns | +1730 ns (+5.5%) | 33180 ns | +1950 ns (+6.2%) |
| SM Throughput (%) | 25.20 % | 71.19 % | +45.99 pp | 29.25 % | +4.05 pp | 29.12 % | +3.92 pp |
| Memory Throughput (%) | 19.43 % | 27.39 % | +7.96 pp | 18.73 % | -0.70 pp | 18.73 % | -0.70 pp |
| DRAM Throughput (%) | 16.11 % | 9.33 % | -6.78 pp | 15.27 % | -0.84 pp | 15.16 % | -0.95 pp |
| Achieved Occupancy (%) | 36.80 % | 66.73 % | +29.93 pp | 34.46 % | -2.34 pp | 34.68 % | -2.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 22.27 | 89.09 | +66.82 | 22.27 | +0.00 | 22.27 | +0.00 |
| No Eligible (%) | 72.53 % | 25.99 % | -46.54 pp | 64.98 % | -7.55 pp | 64.66 % | -7.87 pp |
| IPC Active | 1.09 inst/cycle | 2.94 inst/cycle | +1.85 | 1.37 inst/cycle | +0.28 | 1.36 inst/cycle | +0.27 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_via_autotune_pointwise` (0.580×)

Increasing XBLOCK from 128 to 512 via @triton.autotune improved compute SM throughput (25% → 71%) and occupancy (36.8% → 66.7%), but caused a 1.72x slowdown in wall-clock duration (31.23 µs → 53.86 µs). The critical flaw is that the grid was NOT updated to use a lambda — the grid remained hardcoded as `triton.cdiv(xnumel, 128)`, so the grid size stayed at 47,040 CTAs while each CTA now processed 512 elements, causing 4x more total work (executed instructions jumped from 6.2M to 30.2M). When adding autotune to a pointwise kernel, the grid computation must also be made dynamic (lambda meta: triton.cdiv(xnumel, meta['XBLOCK'])) to match the selected tile size.

> **Applicable when**: Pointwise kernels with low achieved occupancy (<50%), high L1TEX stall cycles (>70% of warp cycles), low eligible warps per scheduler (<0.5), and small block sizes (≤128) relative to available register headroom.

> **Avoid when**: When the grid is not updated to be a dynamic lambda matching the autotune XBLOCK — this causes the kernel to process each element multiple times, multiplying instruction count and causing severe regression. Also avoid when register count per thread is high (>32) and larger blocks would hit register limits.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_use_dynamic_lambda | 0.910× | 0.947× | ✓ |  |
| B | fix_autotune_grid_lambda_and_streaming_eviction_policy | 0.913× | 0.946× | ✓ |  |
| C | fix_dynamic_grid_for_autotune_pointwise | 0.986× | 0.581× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_dynamic_grid_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31230 ns | 32990 ns | +1760 ns (+5.6%) | 33020 ns | +1790 ns (+5.7%) | 53790 ns | +22560 ns (+72.2%) |
| SM Throughput (%) | 25.20 % | 29.22 % | +4.02 pp | 32.18 % | +6.98 pp | 71.16 % | +45.96 pp |
| Memory Throughput (%) | 19.43 % | 18.74 % | -0.69 pp | 18.89 % | -0.54 pp | 27.39 % | +7.96 pp |
| DRAM Throughput (%) | 16.11 % | 15.23 % | -0.88 pp | 15.26 % | -0.85 pp | 9.35 % | -6.76 pp |
| Achieved Occupancy (%) | 36.80 % | 34.63 % | -2.17 pp | 34.15 % | -2.65 pp | 66.23 % | +29.43 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 89.09 | +66.82 |
| No Eligible (%) | 72.53 % | 64.61 % | -7.92 pp | 62.02 % | -10.51 pp | 26.05 % | -46.48 pp |
| IPC Active | 1.09 inst/cycle | 1.37 inst/cycle | +0.28 | 1.48 inst/cycle | +0.39 | 2.95 inst/cycle | +1.86 |

---

## Cumulative Progress

```
Baseline:  31230 ns  (1.000×)

Final best: 31230 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
@@ -18,13 +30,12 @@

     tmp4 = tl.maximum(tmp2, 0.0)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 6021120
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_relu_7[grid](arg_0, arg_1, 6021120, XBLOCK=128)
+    triton_poi_fused_convolution_relu_7[grid](arg_0, arg_1, 6021120)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
