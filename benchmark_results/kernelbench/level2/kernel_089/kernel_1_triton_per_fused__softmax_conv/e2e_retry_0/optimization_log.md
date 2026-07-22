# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 273280 ns — **Best latency**: 35940 ns (**7.6038×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_swap_xblock_rblock_dimensions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_089/kernel_1_triton_per_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 7.76
- **Registers/thread**: 32
- **Duration**: 273280 ns
- **SM Throughput**: 36.58 % | **Memory**: 82.82 % | **DRAM**: 82.82 %
- **Achieved Occupancy**: 93.31 %
- **IPC Active**: 1.51 inst/cycle | **No Eligible**: 62.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_swap_xblock_rblock_dimensions | 6.368× | 7.604× | ✓ | **✓ best** |
| B | fix_uncoalesced_strided_loads_with_evict_first_and_wider_xblock | 0.677× | 0.946× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 273280 ns | 35940 ns | -237340 ns (-86.8%) | 288990 ns | +15710 ns (+5.7%) |
| SM Throughput (%) | 36.58 % | 60.22 % | +23.64 pp | 34.89 % | -1.69 pp |
| Memory Throughput (%) | 82.82 % | 77.94 % | -4.88 pp | 82.24 % | -0.58 pp |
| DRAM Throughput (%) | 82.82 % | 77.94 % | -4.88 pp | 82.24 % | -0.58 pp |
| Achieved Occupancy (%) | 93.31 % | 68.16 % | -25.15 pp | 93.72 % | +0.41 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 32.00 register/thread | +0 |
| Waves/SM | 7.76 | 5.17 | -2.59 | 7.76 | +0.00 |
| No Eligible (%) | 62.44 % | 31.15 % | -31.29 pp | 64.12 % | +1.68 pp |
| IPC Active | 1.51 inst/cycle | 2.70 inst/cycle | +1.19 | 1.45 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  273280 ns  (1.000×)
Iter 1     35940 ns  (7.604×)  — memory_coalescing_swap_xblock_rblock_dimensions

Final best: 35940 ns  (7.6038×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 2097152
@@ -14,7 +25,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
@@ -27,15 +38,15 @@

     x3 = xindex // 16384
     x5 = (xindex % 16384)
     x6 = xindex
-    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp28 = tl.load(in_ptr1 + (r0_4), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), mask=xmask, eviction_policy='evict_last')
+    tmp28 = tl.load(in_ptr1 + (r0_4), mask=xmask, eviction_policy='evict_last')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
@@ -58,9 +69,7 @@

     tmp31 = tmp30 * tmp29
     tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
     tmp34 = triton_helpers.max2(tmp32, 1)[:, None].to(tl.float32)
-    tl.store(in_out_ptr0 + (x6), tmp34, None)
-
-
+    tl.store(in_out_ptr0 + (x6), tmp34, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -69,7 +78,7 @@

     xnumel = 2097152
     r0_numel = 16
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1[grid](arg_0, arg_1, arg_2, 2097152, 16, XBLOCK=128)
+    triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1[grid](arg_0, arg_1, arg_2, 2097152, 16)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
