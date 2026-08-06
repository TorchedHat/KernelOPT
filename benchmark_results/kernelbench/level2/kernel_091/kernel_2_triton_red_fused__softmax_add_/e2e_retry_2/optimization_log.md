# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 57450000 ns — **Best latency**: 303260 ns (**189.4414×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_fix_strided_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_091/kernel_2_triton_red_fused__softmax_add_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16641,1,1), Waves/SM: 42.02
- **Registers/thread**: 32
- **Duration**: 57450000 ns
- **SM Throughput**: 11.89 % | **Memory**: 45.35 % | **DRAM**: 45.35 %
- **Achieved Occupancy**: 18.65 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_fix_strided_store | 446.912× | 189.441× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 57450000 ns | 303260 ns | -57146740 ns (-99.5%) |
| SM Throughput (%) | 11.89 % | 75.14 % | +63.25 pp |
| Memory Throughput (%) | 45.35 % | 35.18 % | -10.17 pp |
| DRAM Throughput (%) | 45.35 % | 35.18 % | -10.17 pp |
| Achieved Occupancy (%) | 18.65 % | 24.38 % | +5.73 pp |
| Registers/Thread | 32.00 register/thread | 127.00 register/thread | +95 |
| Waves/SM | 42.02 | 31.52 | -10.50 |
| No Eligible (%) | 91.32 % | 23.95 % | -67.37 pp |
| IPC Active | 0.35 inst/cycle | 3.05 inst/cycle | +2.70 |

---

## Cumulative Progress

```
Baseline:  57450000 ns  (1.000×)
Iter 1     303260 ns  (189.441×)  — memory_coalescing_fix_strided_store

Final best: 303260 ns  (189.4414×)
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
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2130048
@@ -27,8 +39,8 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
 
@@ -49,9 +61,9 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp6 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp7 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp12 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp6 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp7 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp12 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp8 = tmp6 + tmp7
         tmp9 = tmp8 - tmp4
         tmp10 = tl.exp(tmp9)
@@ -60,9 +72,7 @@

         tmp14 = tl.full([1, 1], 2.0, tl.float32)
         tmp15 = tmp13 * tmp14
         tmp16 = tl.sigmoid(tmp15)
-        tl.store(out_ptr2 + (x0 + 16641*r0_2 + 2130048*x1), tmp16, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (x0 + 16641*r0_2 + 2130048*x1), tmp16, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -72,7 +82,7 @@

     xnumel = 2130048
     r0_numel = 128
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2[grid](arg_0, arg_1, arg_2, arg_3, 2130048, 128, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2[grid](arg_0, arg_1, arg_2, arg_3, 2130048, 128)
     return arg_3
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
