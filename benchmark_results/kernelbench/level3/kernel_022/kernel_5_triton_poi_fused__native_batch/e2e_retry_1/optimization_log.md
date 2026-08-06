# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 65980 ns — **Best latency**: 20510 ns (**3.2170×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_swap_tile_dimensions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 32
- **Duration**: 65980 ns
- **SM Throughput**: 25.96 % | **Memory**: 63.20 % | **DRAM**: 55.80 %
- **Achieved Occupancy**: 92.63 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_swap_tile_dimensions | 3.852× | 3.217× | ✓ | **✓ best** |
| B | software_pipelining_with_autotune | 2.015× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 65980 ns | 20510 ns | -45470 ns (-68.9%) | 66020 ns | +40 ns (+0.1%) |
| SM Throughput (%) | 25.96 % | 66.77 % | +40.81 pp | 26.27 % | +0.31 pp |
| Memory Throughput (%) | 63.20 % | 33.09 % | -30.11 pp | 63.14 % | -0.06 pp |
| DRAM Throughput (%) | 55.80 % | 0.78 % | -55.02 pp | 56.29 % | +0.49 pp |
| Achieved Occupancy (%) | 92.63 % | 75.85 % | -16.78 pp | 93.17 % | +0.54 pp |
| Registers/Thread | 32.00 register/thread | 21.00 register/thread | -11 | 32.00 register/thread | +0 |
| Waves/SM | 11.14 | 22.27 | +11.13 | 11.14 | +0.00 |
| No Eligible (%) | 72.69 % | 25.25 % | -47.44 pp | 72.69 % | +0.00 pp |
| IPC Active | 1.08 inst/cycle | 2.97 inst/cycle | +1.89 | 1.10 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  65980 ns  (1.000×)
Iter 1     20510 ns  (3.217×)  — memory_coalescing_swap_tile_dimensions

Final best: 20510 ns  (3.2170×)
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
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 256}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4,  num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1920
@@ -37,7 +51,6 @@

     tmp15 = tmp0 * combined_scale + combined_bias
     tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -48,7 +61,7 @@

     ynumel = 1920
     xnumel = 12544
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
