# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10180 ns — **Best latency**: 6270 ns (**1.6236×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`add_autotune_reduce_register_pressure_via_smaller_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_17_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.48
- **Registers/thread**: 128
- **Duration**: 10180 ns
- **SM Throughput**: 17.98 % | **Memory**: 21.74 % | **DRAM**: 16.88 %
- **Achieved Occupancy**: 11.53 %
- **IPC Active**: 0.93 inst/cycle | **No Eligible**: 76.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_reduce_register_pressure_via_smaller_xblock | — | 1.624× | ✓ | **✓ best** |
| B | autotune_xblock_to_improve_occupancy_and_wave_count | — | 2.288× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10180 ns | 6270 ns | -3910 ns (-38.4%) | 4450 ns | -5730 ns (-56.3%) |
| SM Throughput (%) | 17.98 % | 6.58 % | -11.40 pp | 8.59 % | -9.39 pp |
| Memory Throughput (%) | 21.74 % | 9.59 % | -12.15 pp | 10.43 % | -11.31 pp |
| DRAM Throughput (%) | 16.88 % | 6.94 % | -9.94 pp | 4.95 % | -11.93 pp |
| Achieved Occupancy (%) | 11.53 % | 3.02 % | -8.51 pp | 5.94 % | -5.59 pp |
| Registers/Thread | 128.00 register/thread | 136.00 register/thread | +8 | 56.00 register/thread | -72 |
| Waves/SM | 0.48 | 0.16 | -0.32 | 0.11 | -0.37 |
| No Eligible (%) | 76.42 % | 78.54 % | +2.12 pp | 83.22 % | +6.80 pp |
| IPC Active | 0.93 inst/cycle | 0.42 inst/cycle | -0.51 | 0.63 inst/cycle | -0.30 |

---

## Cumulative Progress

```
Baseline:  10180 ns  (1.000×)
Iter 1     6270 ns  (1.624×)  — add_autotune_reduce_register_pressure_via_smaller_xblock

Final best: 6270 ns  (1.6236×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32},  num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 32768
@@ -24,12 +33,12 @@

     x0 = (xindex % 512)
     x1 = xindex // 512
     x3 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1, 1], 1e-05, tl.float32)
@@ -52,9 +61,7 @@

     tmp28 = tmp17 - tmp22
     tmp29 = tl.exp(tmp28)
     tmp30 = (tmp29 / tmp27)
-    tl.store(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), tmp30, None)
-
-
+    tl.store(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), tmp30, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -66,7 +73,7 @@

     xnumel = 32768
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 64, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 64)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
