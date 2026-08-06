# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3330 ns (**1.7778×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.69 % | **Memory**: 41.85 % | **DRAM**: 20.83 %
- **Achieved Occupancy**: 55.34 %
- **IPC Active**: 1.09 inst/cycle | **No Eligible**: 70.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.344× | 1.778× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_fix_coalescing | 1.078× | 0.990× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.302× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3330 ns | -2590 ns (-43.8%) | 5980 ns | +60 ns (+1.0%) | 5890 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 16.69 % | 17.01 % | +0.32 pp | 16.57 % | -0.12 pp | 16.70 % | +0.01 pp |
| Memory Throughput (%) | 41.85 % | 20.59 % | -21.26 pp | 40.98 % | -0.87 pp | 41.43 % | -0.42 pp |
| DRAM Throughput (%) | 20.83 % | 4.99 % | -15.84 pp | 20.70 % | -0.13 pp | 21.04 % | +0.21 pp |
| Achieved Occupancy (%) | 55.34 % | 43.99 % | -11.35 pp | 54.92 % | -0.42 pp | 54.54 % | -0.80 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.68 % | 55.81 % | -14.87 pp | 70.67 % | -0.01 pp | 70.71 % | +0.03 pp |
| IPC Active | 1.09 inst/cycle | 1.72 inst/cycle | +0.63 | 1.14 inst/cycle | +0.05 | 1.15 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.7778× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_and_coalesced_store_via_output_index_restructure | 1.297× | 1.635× | ✓ |  |
| B | vectorized_loads_with_contiguous_hint | 1.330× | 1.794× | ✓ |  |
| C | vectorized_loads_with_contiguous_hint | 1.319× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3620 ns | -2300 ns (-38.9%) | 3300 ns | -2620 ns (-44.3%) | 5700 ns | -220 ns (-3.7%) |
| SM Throughput (%) | 16.69 % | 18.46 % | +1.77 pp | 17.15 % | +0.46 pp | 16.57 % | -0.12 pp |
| Memory Throughput (%) | 41.85 % | 22.41 % | -19.44 pp | 20.86 % | -20.99 pp | 42.80 % | +0.95 pp |
| DRAM Throughput (%) | 20.83 % | 13.78 % | -7.05 pp | 5.04 % | -15.79 pp | 21.58 % | +0.75 pp |
| Achieved Occupancy (%) | 55.34 % | 53.52 % | -1.82 pp | 43.25 % | -12.09 pp | 54.79 % | -0.55 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.68 % | 61.54 % | -9.14 pp | 53.91 % | -16.77 pp | 70.66 % | -0.02 pp |
| IPC Active | 1.09 inst/cycle | 1.32 inst/cycle | +0.23 | 1.69 inst/cycle | +0.60 | 1.15 inst/cycle | +0.06 |

---

## Iteration 3

**Best so far**: 1.7778× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_store_via_nhwc_flat_output_index | 1.340× | 1.635× | ✓ |  |
| B | fix_grid_lambda_to_enable_autotune_and_increase_waves_per_sm | 1.340× | 1.635× | ✓ |  |
| C | coalesce_store_via_nhwc_flat_output_index | 1.338× | 1.622× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_store_vi... | Δ | fix_grid_lambda_t... | Δ | coalesce_store_vi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3620 ns | -2300 ns (-38.9%) | 3620 ns | -2300 ns (-38.9%) | 3650 ns | -2270 ns (-38.3%) |
| SM Throughput (%) | 16.69 % | 18.17 % | +1.48 pp | 18.34 % | +1.65 pp | 18.41 % | +1.72 pp |
| Memory Throughput (%) | 41.85 % | 21.30 % | -20.55 pp | 21.87 % | -19.98 pp | 22.37 % | -19.48 pp |
| DRAM Throughput (%) | 20.83 % | 13.66 % | -7.17 pp | 13.78 % | -7.05 pp | 13.61 % | -7.22 pp |
| Achieved Occupancy (%) | 55.34 % | 50.59 % | -4.75 pp | 52.39 % | -2.95 pp | 50.55 % | -4.79 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 20.00 register/thread | -6 | 20.00 register/thread | -6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.68 % | 60.16 % | -10.52 pp | 61.42 % | -9.26 pp | 64.54 % | -6.14 pp |
| IPC Active | 1.09 inst/cycle | 1.49 inst/cycle | +0.40 | 1.42 inst/cycle | +0.33 | 1.49 inst/cycle | +0.40 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3330 ns  (1.778×)  — vectorized_loads_with_autotune

Final best: 3330 ns  (1.7778×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +23,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +32,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
