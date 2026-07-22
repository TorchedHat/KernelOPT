# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10190000 ns — **Best latency**: 20320 ns (**501.4764×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_6_triton_red_fused__softmax_eq_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 10190000 ns
- **SM Throughput**: 13.98 % | **Memory**: 67.89 % | **DRAM**: 67.89 %
- **Achieved Occupancy**: 92.99 %
- **IPC Active**: 0.44 inst/cycle | **No Eligible**: 88.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_xblock_autotune | 591.741× | 501.476× | ✓ | **✓ best** |
| B | online_softmax_attention_single_pass | 13.034× | 4.489× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | online_softmax_at... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10190000 ns | 20320 ns | -10169680 ns (-99.8%) | 2270000 ns | -7920000 ns (-77.7%) |
| SM Throughput (%) | 13.98 % | 72.22 % | +58.24 pp | 11.26 % | -2.72 pp |
| Memory Throughput (%) | 67.89 % | 54.75 % | -13.14 pp | 28.59 % | -39.30 pp |
| DRAM Throughput (%) | 67.89 % | 9.46 % | -58.43 pp | 28.59 % | -39.30 pp |
| Achieved Occupancy (%) | 92.99 % | 89.62 % | -3.37 pp | 12.01 % | -80.98 pp |
| Registers/Thread | 32.00 register/thread | 19.00 register/thread | -13 | 255.00 register/thread | +223 |
| Waves/SM | 1.94 | 7.76 | +5.82 | 15.52 | +13.58 |
| No Eligible (%) | 88.95 % | 19.13 % | -69.82 pp | 88.53 % | -0.42 pp |
| IPC Active | 0.44 inst/cycle | 3.21 inst/cycle | +2.77 | 0.46 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  10190000 ns  (1.000×)
Iter 1     20320 ns  (501.476×)  — memory_coalescing_xblock_autotune

Final best: 20320 ns  (501.4764×)
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
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 256}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 524288
@@ -13,7 +25,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x0 = (xindex % 512)
@@ -26,8 +38,8 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
         tmp1 = tl.full([1, 1], 0.0, tl.float32)
         tmp2 = tmp0 == tmp1
         tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
@@ -53,8 +65,8 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
         tmp12 = tl.full([1, 1], 0.0, tl.float32)
         tmp13 = tmp11 == tmp12
         tmp15 = tl.full([1, 1], 0.10206207261596577, tl.float32)
@@ -64,9 +76,7 @@

         tmp19 = tmp18 - tmp9
         tmp20 = tl.exp(tmp19)
         tmp21 = (tmp20 / tmp10)
-        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, r0_mask)
-
-
+        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, mask=xmask & r0_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -74,7 +84,7 @@

     xnumel = 524288
     r0_numel = 512
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
