# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2240 ns (**1.1562×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_flatten_grid_for_underutilized_layout_kernel`** beats baseline latency.

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
- **Duration**: 2590 ns
- **SM Throughput**: 1.07 % | **Memory**: 1.17 % | **DRAM**: 0.42 %
- **Achieved Occupancy**: 6.17 %
- **IPC Active**: 0.40 inst/cycle | **No Eligible**: 90.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_and_flatten_grid_for_underutilized_layout_kernel | 1.020× | 1.156× | ✓ | **✓ best** |
| B | increase_tile_sizes_for_better_parallelism_and_occupancy | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2240 ns | -350 ns (-13.5%) | 2590 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 1.07 % | 0.10 % | -0.97 pp | 0.26 % | -0.81 pp |
| Memory Throughput (%) | 1.17 % | 0.50 % | -0.67 pp | 0.57 % | -0.60 pp |
| DRAM Throughput (%) | 0.42 % | 0.06 % | -0.36 pp | 0.15 % | -0.27 pp |
| Achieved Occupancy (%) | 6.17 % | 1.56 % | -4.61 pp | 1.56 % | -4.61 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 30.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 90.91 % | 95.14 % | +4.23 pp | 84.86 % | -6.05 pp |
| IPC Active | 0.40 inst/cycle | 0.03 inst/cycle | -0.37 | 0.13 inst/cycle | -0.27 |

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)
Iter 1     2240 ns  (1.156×)  — autotune_tile_sizes_and_flatten_grid_for_underutilized_layout_kernel

Final best: 2240 ns  (1.1562×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,9 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 4}, num_warps=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 4}, num_warps=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8}, num_warps=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    # Simple 2D tiled kernel -- NO persistent loop
     yoffset = tl.program_id(1) * YBLOCK
     yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
     ymask   = yindex < ynumel
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
