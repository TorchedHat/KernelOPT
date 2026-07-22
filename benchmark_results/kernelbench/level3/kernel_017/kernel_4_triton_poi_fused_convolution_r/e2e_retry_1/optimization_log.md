# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2400 ns — **Best latency**: 2270 ns (**1.0573×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_to_maximize_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_4_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (48,1,1), Waves/SM: 0.02
- **Registers/thread**: 30
- **Duration**: 2400 ns
- **SM Throughput**: 1.04 % | **Memory**: 1.14 % | **DRAM**: 0.16 %
- **Achieved Occupancy**: 6.17 %
- **IPC Active**: 0.41 inst/cycle | **No Eligible**: 88.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_and_add_autotune_for_tiny_transpose_kernel | 0.327× | 0.938× | ✓ |  |
| B | fix_tile_sizes_to_maximize_sm_coverage | 1.023× | 1.057× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ | fix_tile_sizes_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2560 ns | +160 ns (+6.7%) | 2270 ns | -130 ns (-5.4%) |
| SM Throughput (%) | 1.04 % | 0.26 % | -0.78 pp | 0.07 % | -0.97 pp |
| Memory Throughput (%) | 1.14 % | 1.27 % | +0.13 pp | 1.08 % | -0.06 pp |
| DRAM Throughput (%) | 0.16 % | 0.19 % | +0.03 pp | 0.23 % | +0.07 pp |
| Achieved Occupancy (%) | 6.17 % | 1.56 % | -4.61 pp | 1.56 % | -4.61 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 16.00 register/thread | -14 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 88.52 % | 88.47 % | -0.05 pp | 93.78 % | +5.26 pp |
| IPC Active | 0.41 inst/cycle | 0.10 inst/cycle | -0.31 | 0.05 inst/cycle | -0.36 |

---

## Cumulative Progress

```
Baseline:  2400 ns  (1.000×)
Iter 1     2270 ns  (1.057×)  — fix_tile_sizes_to_maximize_sm_coverage

Final best: 2270 ns  (1.0573×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     # Simple 2D tiled kernel -- NO persistent loop
@@ -20,14 +37,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9 * yindex), mask=ymask & xmask, other=0.0)
     tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, mask=ymask & xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
