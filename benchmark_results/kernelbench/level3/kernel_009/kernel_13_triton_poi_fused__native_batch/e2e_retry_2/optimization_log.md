# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3620 ns — **Best latency**: 3170 ns (**1.1420×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (784,1,1), Waves/SM: 0.37
- **Registers/thread**: 18
- **Duration**: 3620 ns
- **SM Throughput**: 10.24 % | **Memory**: 3.58 % | **DRAM**: 2.34 %
- **Achieved Occupancy**: 29.95 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 1.040× | 1.000× | ✓ |  |
| B | autotune_xblock_for_occupancy | 1.059× | 1.142× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 3620 ns | +0 ns (+0.0%) | 3170 ns | -450 ns (-12.4%) |
| SM Throughput (%) | 10.24 % | 18.07 % | +7.83 pp | 7.42 % | -2.82 pp |
| Memory Throughput (%) | 3.58 % | 4.80 % | +1.22 pp | 4.09 % | +0.51 pp |
| DRAM Throughput (%) | 2.34 % | 2.34 % | +0.00 pp | 2.66 % | +0.32 pp |
| Achieved Occupancy (%) | 29.95 % | 25.38 % | -4.57 pp | 15.94 % | -14.01 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 0.37 | 0.37 | +0.00 | 0.19 | -0.18 |
| No Eligible (%) | 70.06 % | 59.36 % | -10.70 pp | 79.67 % | +9.61 pp |
| IPC Active | 1.12 inst/cycle | 1.59 inst/cycle | +0.47 | 0.82 inst/cycle | -0.30 |

---

## Cumulative Progress

```
Baseline:  3620 ns  (1.000×)
Iter 1     3170 ns  (1.142×)  — autotune_xblock_for_occupancy

Final best: 3170 ns  (1.1420×)
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
+    triton.Config({'XBLOCK': 128}, num_warps=2),
+    triton.Config({'XBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 256}, num_warps=4),
+    triton.Config({'XBLOCK': 256}, num_warps=8),
+    triton.Config({'XBLOCK': 512}, num_warps=4),
+    triton.Config({'XBLOCK': 512}, num_warps=8),
+    triton.Config({'XBLOCK': 1024}, num_warps=4),
+    triton.Config({'XBLOCK': 1024}, num_warps=8),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 100352
@@ -33,8 +44,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -43,7 +52,7 @@

     arg_4 = inputs[4]
     xnumel = 100352
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
