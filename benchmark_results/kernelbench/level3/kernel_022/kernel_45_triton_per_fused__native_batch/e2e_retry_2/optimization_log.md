# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4320 ns — **Best latency**: 3620 ns (**1.1934×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`reduce_xblock_increase_grid_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_45_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (100,1,1), Waves/SM: 0.09
- **Registers/thread**: 64
- **Duration**: 4320 ns
- **SM Throughput**: 5.40 % | **Memory**: 12.10 % | **DRAM**: 12.10 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.52 inst/cycle | **No Eligible**: 86.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_coverage_via_autotune_xblock | — | fail | ✓ |  |
| B | reduce_xblock_increase_grid_coverage | 1.140× | 1.193× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_xblock_inc... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4320 ns | 3620 ns | -700 ns (-16.2%) |
| SM Throughput (%) | 5.40 % | 2.44 % | -2.96 pp |
| Memory Throughput (%) | 12.10 % | 3.15 % | -8.95 pp |
| DRAM Throughput (%) | 12.10 % | 1.96 % | -10.14 pp |
| Achieved Occupancy (%) | 6.24 % | 3.10 % | -3.14 pp |
| Registers/Thread | 64.00 register/thread | 40.00 register/thread | -24 |
| Waves/SM | 0.09 | 0.03 | -0.06 |
| No Eligible (%) | 86.53 % | 85.97 % | -0.56 pp |
| IPC Active | 0.52 inst/cycle | 0.28 inst/cycle | -0.24 |

---

## Cumulative Progress

```
Baseline:  4320 ns  (1.000×)
Iter 1     3620 ns  (1.193×)  — reduce_xblock_increase_grid_coverage

Final best: 3620 ns  (1.1934×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 16}, num_warps=2),
+    triton.Config({'XBLOCK': 32}, num_warps=2),
+    triton.Config({'XBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 64}, num_warps=4),
+], key=['xnumel', 'r0_numel'])
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 12800
@@ -24,11 +31,11 @@

     x0 = (xindex % 1280)
     x1 = xindex // 1280
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 1280*r0_2 + 62720*x1), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0 + 1280*r0_2 + 62720*x1), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -48,9 +55,7 @@

     tmp22 = tl.full([1, 1], 49.0, tl.float32)
     tmp23 = (tmp21 / tmp22)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp23, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -62,7 +67,7 @@

     xnumel = 12800
     r0_numel = 49
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12800, 49, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12800, 49)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
