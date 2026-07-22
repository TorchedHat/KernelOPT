# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 66020 ns — **Best latency**: 41790 ns (**1.5798×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`software_pipelining_via_autotune_num_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 32
- **Duration**: 66020 ns
- **SM Throughput**: 26.36 % | **Memory**: 63.27 % | **DRAM**: 57.02 %
- **Achieved Occupancy**: 92.95 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | software_pipelining_via_autotune_num_stages | 1.497× | 1.580× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 66020 ns | 41790 ns | -24230 ns (-36.7%) |
| SM Throughput (%) | 26.36 % | 18.19 % | -8.17 pp |
| Memory Throughput (%) | 63.27 % | 73.93 % | +10.66 pp |
| DRAM Throughput (%) | 57.02 % | 39.40 % | -17.62 pp |
| Achieved Occupancy (%) | 92.95 % | 68.16 % | -24.79 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 |
| Waves/SM | 11.14 | 7.42 | -3.72 |
| No Eligible (%) | 72.53 % | 79.98 % | +7.45 pp |
| IPC Active | 1.08 inst/cycle | 0.81 inst/cycle | -0.27 |

---

## Cumulative Progress

```
Baseline:  66020 ns  (1.000×)
Iter 1     41790 ns  (1.580×)  — software_pipelining_via_autotune_num_stages

Final best: 41790 ns  (1.5798×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK':  4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK':  8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK':  8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK':  4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK':  4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1920
@@ -37,7 +55,6 @@

     tmp15 = tmp0 * combined_scale + combined_bias
     tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -48,7 +65,7 @@

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
