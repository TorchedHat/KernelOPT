# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_native_layer_norm_permute_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15520 ns — **Best latency**: 10590 ns (**1.4655×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_uncoalesced_access_and_add_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_native_layer_norm_permute_view_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_11_triton_poi_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 3.45
- **Registers/thread**: 56
- **Duration**: 15520 ns
- **SM Throughput**: 39.13 % | **Memory**: 74.27 % | **DRAM**: 47.68 %
- **Achieved Occupancy**: 49.09 %
- **IPC Active**: 1.84 inst/cycle | **No Eligible**: 53.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 0.993× | 1.002× | ✓ |  |
| B | fix_uncoalesced_access_and_add_autotune | 1.758× | 1.466× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 15520 ns | 15490 ns | -30 ns (-0.2%) | 10590 ns | -4930 ns (-31.8%) |
| SM Throughput (%) | 39.13 % | 39.24 % | +0.11 pp | 54.04 % | +14.91 pp |
| Memory Throughput (%) | 74.27 % | 74.48 % | +0.21 pp | 55.37 % | -18.90 pp |
| DRAM Throughput (%) | 47.68 % | 47.44 % | -0.24 pp | 16.71 % | -30.97 pp |
| Achieved Occupancy (%) | 49.09 % | 49.08 % | -0.01 pp | 26.68 % | -22.41 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 96.00 register/thread | +40 |
| Waves/SM | 3.45 | 3.45 | +0.00 | 3.10 | -0.35 |
| No Eligible (%) | 53.88 % | 54.48 % | +0.60 pp | 30.72 % | -23.16 pp |
| IPC Active | 1.84 inst/cycle | 1.84 inst/cycle | +0.00 | 2.75 inst/cycle | +0.91 |

---

## Cumulative Progress

```
Baseline:  15520 ns  (1.000×)
Iter 1     10590 ns  (1.466×)  — fix_uncoalesced_access_and_add_autotune

Final best: 10590 ns  (1.4655×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_native_layer_norm_permute_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 256
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,13 +33,13 @@

     x3 = xindex
     y0 = yindex
     x2 = xindex // 128
-    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x3 + 256*y0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (y0 + 16384*x3), xmask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr3 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x1), mask=xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x3 + 256*y0), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (y0 + 16384*x3), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x2 + 2*y0), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr3 + (x2 + 2*y0), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (x1), mask=xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr5 + (x1), mask=xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp6 = tmp4 - tmp5
@@ -38,9 +52,7 @@

     tmp15 = tmp13 * tmp14
     tmp17 = tmp15 + tmp16
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x3 + 256*y0), tmp17, xmask)
-
-
+    tl.store(in_out_ptr0 + (x3 + 256*y0), tmp17, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -53,7 +65,7 @@

     ynumel = 16384
     xnumel = 256
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_add_addmm_native_layer_norm_permute_view_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 256, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_add_addmm_native_layer_norm_permute_view_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 256)
     return arg_6
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
