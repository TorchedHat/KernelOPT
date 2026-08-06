# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_clamp_convolution_div_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1210000 ns — **Best latency**: 924450 ns (**1.3089×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`coalesced_load_via_yrange_vectorization_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_clamp_convolution_div_mul_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_002/kernel_2_triton_poi_fused_add_clamp_con/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524288,1,1), Waves/SM: 248.24
- **Registers/thread**: 28
- **Duration**: 1210000 ns
- **SM Throughput**: 25.00 % | **Memory**: 74.35 % | **DRAM**: 71.69 %
- **Achieved Occupancy**: 94.39 %
- **IPC Active**: 1.00 inst/cycle | **No Eligible**: 75.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints_and_tile_swap | 2.252× | 1.311× | ✓ | **✓ best** |
| C | software_pipelining_with_autotune | 1.546× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1210000 ns | 922720 ns | -287280 ns (-23.7%) | 1230000 ns | +20000 ns (+1.7%) |
| SM Throughput (%) | 25.00 % | 68.56 % | +43.56 pp | 31.15 % | +6.15 pp |
| Memory Throughput (%) | 74.35 % | 79.28 % | +4.93 pp | 73.61 % | -0.74 pp |
| DRAM Throughput (%) | 71.69 % | 11.57 % | -60.12 pp | 70.74 % | -0.95 pp |
| Achieved Occupancy (%) | 94.39 % | 90.34 % | -4.05 pp | 95.41 % | +1.02 pp |
| Registers/Thread | 28.00 register/thread | 24.00 register/thread | -4 | 32.00 register/thread | +4 |
| Waves/SM | 248.24 | 496.48 | +248.24 | 248.24 | +0.00 |
| No Eligible (%) | 75.02 % | 31.20 % | -43.82 pp | 69.01 % | -6.01 pp |
| IPC Active | 1.00 inst/cycle | 2.75 inst/cycle | +1.75 | 1.24 inst/cycle | +0.24 |

---

## Iteration 2

**Best so far**: 1.3113× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_load_via_yrange_vectorization_hints | 3.387× | 1.309× | ✓ | **✓ best** |
| C | vectorized_loads_with_expanded_autotune_configs | 2.255× | 1.309× | ✓ |  |
| D | output_coalescing_via_contiguous_hints_and_expanded_autotune | 2.247× | 1.310× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_load_vi... | Δ | vectorized_loads_... | Δ | output_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1210000 ns | 924450 ns | -285550 ns (-23.6%) | 924670 ns | -285330 ns (-23.6%) | 923550 ns | -286450 ns (-23.7%) |
| SM Throughput (%) | 25.00 % | 68.54 % | +43.54 pp | 68.90 % | +43.90 pp | 68.51 % | +43.51 pp |
| Memory Throughput (%) | 74.35 % | 79.28 % | +4.93 pp | 79.26 % | +4.91 pp | 79.24 % | +4.89 pp |
| DRAM Throughput (%) | 71.69 % | 11.55 % | -60.14 pp | 11.49 % | -60.20 pp | 11.55 % | -60.14 pp |
| Achieved Occupancy (%) | 94.39 % | 90.33 % | -4.06 pp | 90.36 % | -4.03 pp | 90.35 % | -4.04 pp |
| Registers/Thread | 28.00 register/thread | 24.00 register/thread | -4 | 24.00 register/thread | -4 | 24.00 register/thread | -4 |
| Waves/SM | 248.24 | 496.48 | +248.24 | 496.48 | +248.24 | 496.48 | +248.24 |
| No Eligible (%) | 75.02 % | 31.23 % | -43.79 pp | 30.88 % | -44.14 pp | 31.28 % | -43.74 pp |
| IPC Active | 1.00 inst/cycle | 2.75 inst/cycle | +1.75 | 2.77 inst/cycle | +1.77 | 2.75 inst/cycle | +1.75 |

---

## Iteration 3

**Best so far**: 1.3113× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_triton_autotune | 1.305× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1210000 ns | 1230000 ns | +20000 ns (+1.7%) |
| SM Throughput (%) | 25.00 % | 31.98 % | +6.98 pp |
| Memory Throughput (%) | 74.35 % | 73.56 % | -0.79 pp |
| DRAM Throughput (%) | 71.69 % | 70.60 % | -1.09 pp |
| Achieved Occupancy (%) | 94.39 % | 95.41 % | +1.02 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 |
| Waves/SM | 248.24 | 248.24 | +0.00 |
| No Eligible (%) | 75.02 % | 67.98 % | -7.04 pp |
| IPC Active | 1.00 inst/cycle | 1.27 inst/cycle | +0.27 |

---

## Iteration 4

**Best so far**: 1.3113× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | software_pipelining_increased_num_stages | 1.304× | 0.984× | ✓ |  |
| D | fix_autotune_grid_to_use_meta_and_expand_large_xblock_configs | 2.371× | 1.309× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1210000 ns | 1230000 ns | +20000 ns (+1.7%) | 924220 ns | -285780 ns (-23.6%) |
| SM Throughput (%) | 25.00 % | 32.05 % | +7.05 pp | 68.88 % | +43.88 pp |
| Memory Throughput (%) | 74.35 % | 73.75 % | -0.60 pp | 79.23 % | +4.88 pp |
| DRAM Throughput (%) | 71.69 % | 70.81 % | -0.88 pp | 11.49 % | -60.20 pp |
| Achieved Occupancy (%) | 94.39 % | 95.38 % | +0.99 pp | 90.36 % | -4.03 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 24.00 register/thread | -4 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 496.48 | +248.24 |
| No Eligible (%) | 75.02 % | 68.25 % | -6.77 pp | 30.88 % | -44.14 pp |
| IPC Active | 1.00 inst/cycle | 1.27 inst/cycle | +0.27 | 2.76 inst/cycle | +1.76 |

---

## Cumulative Progress

```
Baseline:  1210000 ns  (1.000×)
Iter 1     922720 ns  (1.311×)  — vectorized_loads_via_contiguous_hints_and_tile_swap

Final best: 924450 ns  (1.3089×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_clamp_convolution_div_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 65536
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
+    _x_range = tl.arange(0, XBLOCK)
+    _x_range = tl.max_contiguous(tl.multiple_of(_x_range, XBLOCK), XBLOCK)
+    xindex = xoffset + _x_range[None, :]
+    xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 4194304*y1), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 4194304*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp5 = tl.full([1, 1], 0.0, tl.float32)
@@ -34,9 +46,7 @@

     tmp12 = tl.minimum(tmp11, tmp7)
     tmp13 = tl.full([1, 1], 0.5, tl.float32)
     tmp14 = tmp12 * tmp13
-    tl.store(out_ptr0 + (x2 + 65536*y3), tmp14, None)
-
-
+    tl.store(out_ptr0 + (x2 + 65536*y3), tmp14, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +56,7 @@

     ynumel = 8192
     xnumel = 65536
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_add_clamp_convolution_div_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 8192, 65536, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_add_clamp_convolution_div_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 8192, 65536)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
