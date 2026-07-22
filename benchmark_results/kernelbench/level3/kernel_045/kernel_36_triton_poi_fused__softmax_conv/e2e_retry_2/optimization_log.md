# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3010 ns — **Best latency**: 2880 ns (**1.0451×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_36_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 27
- **Duration**: 3010 ns
- **SM Throughput**: 13.50 % | **Memory**: 13.40 % | **DRAM**: 0.92 %
- **Achieved Occupancy**: 40.12 %
- **IPC Active**: 1.35 inst/cycle | **No Eligible**: 59.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_underutilized_transpose_kernel | — | 1.045× | ✓ | **✓ best** |
| B | add_triton_autotune | — | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3010 ns | 2880 ns | -130 ns (-4.3%) | 3070 ns | +60 ns (+2.0%) |
| SM Throughput (%) | 13.50 % | 6.39 % | -7.11 pp | 14.60 % | +1.10 pp |
| Memory Throughput (%) | 13.40 % | 2.45 % | -10.95 pp | 14.52 % | +1.12 pp |
| DRAM Throughput (%) | 0.92 % | 0.97 % | +0.05 pp | 0.91 % | -0.01 pp |
| Achieved Occupancy (%) | 40.12 % | 14.48 % | -25.64 pp | 40.15 % | +0.03 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 | 27.00 register/thread | +0 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.48 | +0.00 |
| No Eligible (%) | 59.43 % | 81.96 % | +22.53 pp | 59.89 % | +0.46 pp |
| IPC Active | 1.35 inst/cycle | 0.82 inst/cycle | -0.53 | 1.54 inst/cycle | +0.19 |

---

## Cumulative Progress

```
Baseline:  3010 ns  (1.000×)
Iter 1     2880 ns  (1.045×)  — autotune_tile_sizes_for_underutilized_transpose_kernel

Final best: 2880 ns  (1.0451×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 512}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 4
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +30,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +39,7 @@

     ynumel = 8192
     xnumel = 4
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
