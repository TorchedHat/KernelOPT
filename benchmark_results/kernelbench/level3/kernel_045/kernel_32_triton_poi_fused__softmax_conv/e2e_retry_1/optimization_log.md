# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5060 ns — **Best latency**: 4700 ns (**1.0766×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_and_autotune_for_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_32`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_32_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 27
- **Duration**: 5060 ns
- **SM Throughput**: 34.02 % | **Memory**: 34.97 % | **DRAM**: 2.13 %
- **Achieved Occupancy**: 69.73 %
- **IPC Active**: 2.42 inst/cycle | **No Eligible**: 46.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_and_autotune_for_transpose_kernel | — | 1.077× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5060 ns | 4700 ns | -360 ns (-7.1%) |
| SM Throughput (%) | 34.02 % | 13.59 % | -20.43 pp |
| Memory Throughput (%) | 34.97 % | 5.53 % | -29.44 pp |
| DRAM Throughput (%) | 2.13 % | 2.30 % | +0.17 pp |
| Achieved Occupancy (%) | 69.73 % | 45.34 % | -24.39 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 |
| Waves/SM | 1.94 | 0.97 | -0.97 |
| No Eligible (%) | 46.18 % | 41.67 % | -4.51 pp |
| IPC Active | 2.42 inst/cycle | 1.83 inst/cycle | -0.59 |

---

## Cumulative Progress

```
Baseline:  5060 ns  (1.000×)
Iter 1     4700 ns  (1.077×)  — fix_tile_sizes_and_autotune_for_transpose_kernel

Final best: 4700 ns  (1.0766×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 4}, num_warps=2),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 4}, num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 4}, num_warps=4),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 4}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32768
     xnumel = 4
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +29,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     ynumel = 32768
     xnumel = 4
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_32[grid](arg_0, arg_1, 32768, 4, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_32[grid](arg_0, arg_1, 32768, 4)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
