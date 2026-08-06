# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17760 ns — **Best latency**: 4510 ns (**3.9379×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_reduce_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_1_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1600,1,1), Waves/SM: 1.52
- **Registers/thread**: 60
- **Duration**: 17760 ns
- **SM Throughput**: 40.46 % | **Memory**: 45.49 % | **DRAM**: 31.49 %
- **Achieved Occupancy**: 39.36 %
- **IPC Active**: 1.90 inst/cycle | **No Eligible**: 52.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_reduce_xblock | 2.641× | 3.938× | ✓ | **✓ best** |
| B | autotune_xblock_num_warps_to_reduce_register_pressure_and_tail_wave | 2.247× | 2.762× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17760 ns | 4510 ns | -13250 ns (-74.6%) | 6430 ns | -11330 ns (-63.8%) |
| SM Throughput (%) | 40.46 % | 19.63 % | -20.83 pp | 28.70 % | -11.76 pp |
| Memory Throughput (%) | 45.49 % | 21.47 % | -24.02 pp | 30.89 % | -14.60 pp |
| DRAM Throughput (%) | 31.49 % | 14.96 % | -16.53 pp | 20.83 % | -10.66 pp |
| Achieved Occupancy (%) | 39.36 % | 15.94 % | -23.42 pp | 32.47 % | -6.89 pp |
| Registers/Thread | 60.00 register/thread | 72.00 register/thread | +12 | 44.00 register/thread | -16 |
| Waves/SM | 1.52 | 0.43 | -1.09 | 0.61 | -0.91 |
| No Eligible (%) | 52.42 % | 64.17 % | +11.75 pp | 54.59 % | +2.17 pp |
| IPC Active | 1.90 inst/cycle | 1.48 inst/cycle | -0.42 | 1.78 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  17760 ns  (1.000×)
Iter 1     4510 ns  (3.938×)  — memory_coalescing_reduce_xblock

Final best: 4510 ns  (3.9379×)
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
+        triton.Config({'XBLOCK': 16}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 204800
@@ -39,7 +51,6 @@

     result = exps / row_sum
     tl.store(in_out_ptr0 + (r0_1 + 32*x0), result, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -49,7 +60,7 @@

     xnumel = 204800
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
