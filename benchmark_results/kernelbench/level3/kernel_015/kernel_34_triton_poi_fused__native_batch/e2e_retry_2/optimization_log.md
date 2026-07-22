# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3780 ns — **Best latency**: 3070 ns (**1.2313×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`persistent_tile_scheduling_with_autotuned_block_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_34_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1664,1,1), Waves/SM: 0.79
- **Registers/thread**: 28
- **Duration**: 3780 ns
- **SM Throughput**: 23.70 % | **Memory**: 22.08 % | **DRAM**: 2.64 %
- **Achieved Occupancy**: 55.44 %
- **IPC Active**: 2.07 inst/cycle | **No Eligible**: 50.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotuned_block_sizes | 1.121× | 1.231× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_grid_coverage_and_occupancy | 1.050× | 1.256× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 3070 ns | -710 ns (-18.8%) | 3010 ns | -770 ns (-20.4%) |
| SM Throughput (%) | 23.70 % | 6.06 % | -17.64 pp | 18.39 % | -5.31 pp |
| Memory Throughput (%) | 22.08 % | 5.87 % | -16.21 pp | 6.62 % | -15.46 pp |
| DRAM Throughput (%) | 2.64 % | 3.25 % | +0.61 pp | 3.32 % | +0.68 pp |
| Achieved Occupancy (%) | 55.44 % | 14.40 % | -41.04 pp | 60.16 % | +4.72 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.79 | 0.39 | -0.40 | 1.58 | +0.79 |
| No Eligible (%) | 50.60 % | 81.82 % | +31.22 pp | 45.09 % | -5.51 pp |
| IPC Active | 2.07 inst/cycle | 0.65 inst/cycle | -1.42 | 2.28 inst/cycle | +0.21 |

---

## Cumulative Progress

```
Baseline:  3780 ns  (1.000×)
Iter 1     3070 ns  (1.231×)  — persistent_tile_scheduling_with_autotuned_block_sizes

Final best: 3070 ns  (1.2313×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 13312
     xnumel = 9
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    # x-dimension: XBLOCK=16 covers xnumel=9 with only 7 masked threads (vs 119 before)
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
-    y3 = yindex
-    y0 = (yindex % 416)
-    y1 = yindex // 416
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 416*x2 + 3744*y1), tmp0, xmask)
-
-
+    # Persistent y-loop: each CTA strides over all y-tiles
+    pid_y = tl.program_id(1)
+    num_pid_y = tl.num_programs(1)
+    for yoffset in tl.range(pid_y * YBLOCK, ynumel, num_pid_y * YBLOCK):
+        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+        ymask = yindex < ynumel
+        y3 = yindex
+        y0 = (yindex % 416)
+        y1 = yindex // 416
+        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+        tl.store(out_ptr0 + (y0 + 416*x2 + 3744*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

     ynumel = 13312
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34[grid](arg_0, arg_1, 13312, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34[grid](arg_0, arg_1, 13312, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
