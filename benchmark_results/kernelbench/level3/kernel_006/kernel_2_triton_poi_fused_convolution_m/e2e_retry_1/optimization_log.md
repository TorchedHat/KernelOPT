# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2910000 ns — **Best latency**: 3050000 ns (**0.9541×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (235200,1,1), Waves/SM: 356.36
- **Registers/thread**: 94
- **Duration**: 2910000 ns
- **SM Throughput**: 24.81 % | **Memory**: 69.94 % | **DRAM**: 33.63 %
- **Achieved Occupancy**: 30.35 %
- **IPC Active**: 1.00 inst/cycle | **No Eligible**: 75.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 5.269× | 0.954× | ✓ | **✓ best** |
| B | memory_coalescing_swap_xblock_yblock | 1.006× | 1.018× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2910000 ns | 3050000 ns | +140000 ns (+4.8%) | 2860000 ns | -50000 ns (-1.7%) |
| SM Throughput (%) | 24.81 % | 26.82 % | +2.01 pp | 25.17 % | +0.36 pp |
| Memory Throughput (%) | 69.94 % | 79.37 % | +9.43 pp | 71.01 % | +1.07 pp |
| DRAM Throughput (%) | 33.63 % | 32.08 % | -1.55 pp | 34.11 % | +0.48 pp |
| Achieved Occupancy (%) | 30.35 % | 30.42 % | +0.07 pp | 30.36 % | +0.01 pp |
| Registers/Thread | 94.00 register/thread | 95.00 register/thread | +1 | 93.00 register/thread | -1 |
| Waves/SM | 356.36 | 356.36 | +0.00 | 356.36 | +0.00 |
| No Eligible (%) | 75.07 % | 73.18 % | -1.89 pp | 74.74 % | -0.33 pp |
| IPC Active | 1.00 inst/cycle | 1.07 inst/cycle | +0.07 | 1.01 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  2910000 ns  (1.000×)

Final best: 3050000 ns  (0.9541×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 32,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4800
@@ -14,6 +25,7 @@

     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xindex, [1, 16]), [1, 16])
     xmask = xindex < xnumel
     x3 = xindex // 224
     x2 = (xindex % 224)
@@ -21,7 +33,7 @@

     y5 = yindex
     y0 = (yindex % 480)
     y1 = yindex // 480
-    tmp52 = tl.load(in_ptr0 + (x4 + 50176*y5), xmask & ymask, eviction_policy='evict_last')
+    tmp52 = tl.load(in_ptr0 + (x4 + 50176*y5), mask=xmask & ymask, eviction_policy='evict_last')
     tmp0 = (-1) + x3
     tmp1 = tl.full([1, 1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
@@ -33,53 +45,51 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-225) + x4 + 50176*y5), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-225) + x4 + 50176*y5), mask=tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp12 = x2
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-224) + x4 + 50176*y5), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-224) + x4 + 50176*y5), mask=tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + x2
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-223) + x4 + 50176*y5), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-223) + x4 + 50176*y5), mask=tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = x3
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + 50176*y5), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + 50176*y5), mask=tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x4 + 50176*y5), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x4 + 50176*y5), mask=tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (1 + x4 + 50176*y5), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (1 + x4 + 50176*y5), mask=tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + x3
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (223 + x4 + 50176*y5), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (223 + x4 + 50176*y5), mask=tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (224 + x4 + 50176*y5), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (224 + x4 + 50176*y5), mask=tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (225 + x4 + 50176*y5), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (225 + x4 + 50176*y5), mask=tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (y0 + 480*x4 + 24084480*y1), tmp51, xmask & ymask)
-    tl.store(out_ptr1 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-    tl.store(out_ptr2 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-    tl.store(out_ptr3 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 480*x4 + 24084480*y1), tmp51, mask=xmask & ymask)
+    tl.store(out_ptr1 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
+    tl.store(out_ptr2 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
+    tl.store(out_ptr3 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -90,7 +100,7 @@

     ynumel = 4800
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_max_pool2d_with_indices_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4800, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_max_pool2d_with_indices_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4800, 50176)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
