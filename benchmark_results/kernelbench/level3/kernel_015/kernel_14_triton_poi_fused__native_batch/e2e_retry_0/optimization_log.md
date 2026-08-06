# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 2720 ns (**1.1176×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (896,1,1), Waves/SM: 0.42
- **Registers/thread**: 28
- **Duration**: 3040 ns
- **SM Throughput**: 15.44 % | **Memory**: 14.00 % | **DRAM**: 1.77 %
- **Achieved Occupancy**: 33.69 %
- **IPC Active**: 1.51 inst/cycle | **No Eligible**: 56.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.048× | 0.013× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy | 1.070× | 1.118× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 230750 ns | +227710 ns (+7490.5%) | 2720 ns | -320 ns (-10.5%) |
| SM Throughput (%) | 15.44 % | 36.31 % | +20.87 pp | 10.57 % | -4.87 pp |
| Memory Throughput (%) | 14.00 % | 31.94 % | +17.94 pp | 8.97 % | -5.03 pp |
| DRAM Throughput (%) | 1.77 % | 0.02 % | -1.75 pp | 1.99 % | +0.22 pp |
| Achieved Occupancy (%) | 33.69 % | 21.21 % | -12.48 pp | 14.26 % | -19.43 pp |
| Registers/Thread | 28.00 register/thread | 31.00 register/thread | +3 | 30.00 register/thread | +2 |
| Waves/SM | 0.42 | 0.21 | -0.21 | 0.21 | -0.21 |
| No Eligible (%) | 56.62 % | 63.19 % | +6.57 pp | 62.97 % | +6.35 pp |
| IPC Active | 1.51 inst/cycle | 1.48 inst/cycle | -0.03 | 1.23 inst/cycle | -0.28 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.013×)

Converting a simple 2-D grid kernel into a persistent while-loop kernel caused a 177x increase in executed instructions (376K → 66.7M) and a 76x increase in kernel duration (3.04 µs → 230.75 µs). The persistent loop forced each of the 132 CTAs to serially iterate through all tiles that were previously handled in parallel by 896 CTAs, serializing work that was already parallel. The original kernel was short-running (~3 µs) and memory-bound at low occupancy due to small grid size — the correct fix is to increase parallelism (more CTAs or larger tiles), not to serialize tile traversal inside fewer CTAs.

> **Applicable when**: Never applicable as described. Persistent kernels are beneficial when the kernel is compute-bound and has high arithmetic intensity, or when there is producer-consumer reuse across tiles that benefits from keeping data in registers/shared memory. They are NOT beneficial for simple memory-copy/transpose kernels with low arithmetic intensity.

> **Avoid when**: Do not apply persistent while-loop tile scheduling to short-running (<10 µs), memory-bandwidth-bound kernels with low arithmetic intensity (e.g., elementwise ops, transposes, batch norm fusions). The serialization of tiles inside fewer CTAs eliminates the parallelism that made the original kernel fast, causing instruction count to explode proportionally to (original_grid_size / num_sms). The original kernel had 896 CTAs running in parallel; reducing to 132 CTAs with a loop serializes 6.8x more work per CTA.

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)
Iter 1     2720 ns  (1.118×)  — autotune_tile_sizes_for_occupancy

Final best: 2720 ns  (1.1176×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,26 @@

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
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 7168
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +32,8 @@

     y3 = yindex
     y0 = (yindex % 224)
     y1 = yindex // 224
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 224*x2 + 2016*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 224*x2 + 2016*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 7168
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
