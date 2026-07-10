# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 57470000 ns — **Best latency**: 302820 ns (**189.7827×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`reduce_shared_memory_and_register_spill_via_smaller_R0_BLOCK`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_091/kernel_2_triton_red_fused__softmax_add_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16641,1,1), Waves/SM: 42.02
- **Registers/thread**: 32
- **Duration**: 57470000 ns
- **SM Throughput**: 11.86 % | **Memory**: 45.33 % | **DRAM**: 45.33 %
- **Achieved Occupancy**: 18.65 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_shared_memory_and_register_spill_via_smaller_R0_BLOCK | 445.793× | 189.783× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 57470000 ns | 302820 ns | -57167180 ns (-99.5%) |
| SM Throughput (%) | 11.86 % | 74.93 % | +63.07 pp |
| Memory Throughput (%) | 45.33 % | 35.24 % | -10.09 pp |
| DRAM Throughput (%) | 45.33 % | 35.24 % | -10.09 pp |
| Achieved Occupancy (%) | 18.65 % | 24.38 % | +5.73 pp |
| Registers/Thread | 32.00 register/thread | 127.00 register/thread | +95 |
| Waves/SM | 42.02 | 31.52 | -10.50 |
| No Eligible (%) | 91.31 % | 23.86 % | -67.45 pp |
| IPC Active | 0.35 inst/cycle | 3.05 inst/cycle | +2.70 |

---

## Cumulative Progress

```
Baseline:  57470000 ns  (1.000×)
Iter 1     302820 ns  (189.783×)  — reduce_shared_memory_and_register_spill_via_smaller_R0_BLOCK

Final best: 302820 ns  (189.7827×)
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
