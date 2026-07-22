# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3740 ns — **Best latency**: 3360 ns (**1.1131×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_dimensions_for_small_xnumel_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_37_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1792,1,1), Waves/SM: 0.85
- **Registers/thread**: 28
- **Duration**: 3740 ns
- **SM Throughput**: 25.52 % | **Memory**: 23.46 % | **DRAM**: 2.85 %
- **Achieved Occupancy**: 58.42 %
- **IPC Active**: 2.18 inst/cycle | **No Eligible**: 48.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_dimensions_for_small_xnumel_transpose | 1.016× | 1.113× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3740 ns | 3360 ns | -380 ns (-10.2%) |
| SM Throughput (%) | 25.52 % | 16.56 % | -8.96 pp |
| Memory Throughput (%) | 23.46 % | 14.81 % | -8.65 pp |
| DRAM Throughput (%) | 2.85 % | 3.20 % | +0.35 pp |
| Achieved Occupancy (%) | 58.42 % | 31.61 % | -26.81 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 |
| Waves/SM | 0.85 | 0.42 | -0.43 |
| No Eligible (%) | 48.31 % | 54.13 % | +5.82 pp |
| IPC Active | 2.18 inst/cycle | 2.01 inst/cycle | -0.17 |

---

## Cumulative Progress

```
Baseline:  3740 ns  (1.000×)
Iter 1     3360 ns  (1.113×)  — fix_tile_dimensions_for_small_xnumel_transpose

Final best: 3360 ns  (1.1131×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 14336
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = (yindex < ynumel)
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +27,8 @@

     y3 = yindex
     y0 = (yindex % 448)
     y1 = yindex // 448
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +36,7 @@

     ynumel = 14336
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37[grid](arg_0, arg_1, 14336, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37[grid](arg_0, arg_1, 14336, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
