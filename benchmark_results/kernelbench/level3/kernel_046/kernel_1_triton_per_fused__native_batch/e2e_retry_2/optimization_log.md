# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17820 ns — **Best latency**: 10370 ns (**1.7184×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`reduce_xblock_to_improve_occupancy_and_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_1_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1600,1,1), Waves/SM: 1.52
- **Registers/thread**: 60
- **Duration**: 17820 ns
- **SM Throughput**: 40.31 % | **Memory**: 45.42 % | **DRAM**: 31.34 %
- **Achieved Occupancy**: 38.87 %
- **IPC Active**: 1.88 inst/cycle | **No Eligible**: 51.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_to_improve_occupancy_and_eliminate_partial_wave | 2.266× | 1.718× | ✓ | **✓ best** |
| B | autotune_xblock_and_warps_to_improve_occupancy | 2.243× | 2.784× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17820 ns | 10370 ns | -7450 ns (-41.8%) | 6400 ns | -11420 ns (-64.1%) |
| SM Throughput (%) | 40.31 % | 35.19 % | -5.12 pp | 28.16 % | -12.15 pp |
| Memory Throughput (%) | 45.42 % | 39.04 % | -6.38 pp | 30.46 % | -14.96 pp |
| DRAM Throughput (%) | 31.34 % | 25.80 % | -5.54 pp | 20.88 % | -10.46 pp |
| Achieved Occupancy (%) | 38.87 % | 31.90 % | -6.97 pp | 32.95 % | -5.92 pp |
| Registers/Thread | 60.00 register/thread | 64.00 register/thread | +4 | 44.00 register/thread | -16 |
| Waves/SM | 1.52 | 0.76 | -0.76 | 0.61 | -0.91 |
| No Eligible (%) | 51.64 % | 53.90 % | +2.26 pp | 54.79 % | +3.15 pp |
| IPC Active | 1.88 inst/cycle | 1.82 inst/cycle | -0.06 | 1.63 inst/cycle | -0.25 |

---

## Cumulative Progress

```
Baseline:  17820 ns  (1.000×)
Iter 1     10370 ns  (1.718×)  — reduce_xblock_to_improve_occupancy_and_eliminate_partial_wave

Final best: 10370 ns  (1.7184×)
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
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 204800
@@ -39,7 +50,6 @@

     result = exps / row_sum
     tl.store(in_out_ptr0 + (r0_1 + 32*x0), result, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -49,7 +59,7 @@

     xnumel = 204800
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
