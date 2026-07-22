# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 788060 ns — **Best latency**: 3490 ns (**225.8052×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_correct_r0_block`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 788060 ns
- **SM Throughput**: 12.99 % | **Memory**: 38.65 % | **DRAM**: 38.65 %
- **Achieved Occupancy**: 11.25 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 94.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_correct_r0_block | 121.661× | 225.805× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 788060 ns | 3490 ns | -784570 ns (-99.6%) |
| SM Throughput (%) | 12.99 % | 8.17 % | -4.82 pp |
| Memory Throughput (%) | 38.65 % | 11.49 % | -27.16 pp |
| DRAM Throughput (%) | 38.65 % | 0.62 % | -38.03 pp |
| Achieved Occupancy (%) | 11.25 % | 11.08 % | -0.17 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 |
| Waves/SM | 0.62 | 0.12 | -0.50 |
| No Eligible (%) | 94.96 % | 81.84 % | -13.12 pp |
| IPC Active | 0.20 inst/cycle | 0.68 inst/cycle | +0.48 |

---

## Cumulative Progress

```
Baseline:  788060 ns  (1.000×)
Iter 1     3490 ns  (225.805×)  — vectorized_loads_and_correct_r0_block

Final best: 3490 ns  (225.8052×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -24,12 +38,13 @@

     x3 = xindex
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
+        r0_index_vec = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
+        r0_mask = r0_index_vec < r0_numel
         roffset = r0_offset
-        rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        rindex = r0_index_vec
+        r0_2 = r0_index_vec
+        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -44,14 +59,15 @@

     tmp9 = tmp7[:, None]
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
+        r0_index_vec = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
+        r0_mask = r0_index_vec < r0_numel
         roffset = r0_offset
-        rindex = r0_index
-        r0_2 = r0_index
-        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp11 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp20 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp22 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        rindex = r0_index_vec
+        r0_2 = r0_index_vec
+        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp11 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp20 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp22 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp12 = tmp10 + tmp11
         tmp13 = tmp12 - tmp4
         tmp14 = tl.full([1, 1], 96.0, tl.float32)
@@ -62,9 +78,7 @@

         tmp19 = tmp13 * tmp18
         tmp21 = tmp19 * tmp20
         tmp23 = tmp21 + tmp22
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +89,7 @@

     xnumel = 31360
     r0_numel = 96
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
