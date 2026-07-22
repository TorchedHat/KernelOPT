# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 124800 ns — **Best latency**: 43360 ns (**2.8782×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 29.70
- **Registers/thread**: 62
- **Duration**: 124800 ns
- **SM Throughput**: 60.82 % | **Memory**: 75.65 % | **DRAM**: 61.19 %
- **Achieved Occupancy**: 47.59 %
- **IPC Active**: 2.50 inst/cycle | **No Eligible**: 36.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_cache_reuse_and_occupancy | 1.812× | 1.329× | ✓ |  |
| B | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 2.664× | 2.878× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.000× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 124800 ns | 93890 ns | -30910 ns (-24.8%) | 43360 ns | -81440 ns (-65.3%) | 125120 ns | +320 ns (+0.3%) |
| SM Throughput (%) | 60.82 % | 77.16 % | +16.34 pp | 74.21 % | +13.39 pp | 59.80 % | -1.02 pp |
| Memory Throughput (%) | 75.65 % | 43.91 % | -31.74 pp | 59.36 % | -16.29 pp | 75.39 % | -0.26 pp |
| DRAM Throughput (%) | 61.19 % | 39.12 % | -22.07 pp | 40.58 % | -20.61 pp | 61.01 % | -0.18 pp |
| Achieved Occupancy (%) | 47.59 % | 47.00 % | -0.59 pp | 51.81 % | +4.22 pp | 47.59 % | +0.00 pp |
| Registers/Thread | 62.00 register/thread | 58.00 register/thread | -4 | 50.00 register/thread | -12 | 62.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 13.20 | -16.50 | 29.70 | +0.00 |
| No Eligible (%) | 36.98 % | 21.06 % | -15.92 pp | 20.75 % | -16.23 pp | 38.22 % | +1.24 pp |
| IPC Active | 2.50 inst/cycle | 3.16 inst/cycle | +0.66 | 3.14 inst/cycle | +0.64 | 2.47 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy` (2.878×)

When a Triton kernel is register-limited (Block Limit Registers is the binding occupancy constraint, e.g. 8 blocks/SM vs. hardware max of 16-32), reducing XBLOCK (tile width) lowers the number of simultaneously live values per thread, cutting registers/thread and allowing more blocks to reside concurrently on each SM. Wrapping the kernel with @triton.autotune over a range of smaller XBLOCK/YBLOCK configs lets the runtime select the Pareto-optimal tile size, increasing eligible warps/scheduler (1.45→2.59) and reducing warp stall cycles (12.07→10.49 cycles/instruction), which together cut kernel duration by ~65%. This pattern is most impactful when TheoreticalOccupancy NCU rule fires with 'limited by registers', L2 throughput is near-saturated (>80%), and L1TEX scoreboard stalls account for >30% of warp cycles.

> **Applicable when**: TheoreticalOccupancy NCU rule fires citing registers as the binding block limit; Block Limit Registers is the minimum among all block limits (e.g., 8 vs. 16-32 for other limits); L2 throughput >80% indicating memory-latency bottleneck; L1TEX scoreboard stalls >30% of warp cycles; eligible warps/scheduler <2.0; kernel has 2D tile structure (XBLOCK x YBLOCK) with large XBLOCK (>=128)

> **Avoid when**: Kernel is already compute-bound (Compute SM Throughput > Memory Throughput and no register-limit occupancy warning); Block Limit Registers is not the binding constraint (e.g., shared memory or warp count is limiting); reducing tile size would cause excessive boundary-check overhead relative to useful work; problem size is too small to amortize autotune overhead

---

## Cumulative Progress

```
Baseline:  124800 ns  (1.000×)
Iter 1     43360 ns  (2.878×)  — autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy

Final best: 43360 ns  (2.8782×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 640
@@ -19,16 +33,16 @@

     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr5 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr6 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr7 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr8 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -51,9 +65,7 @@

     tmp29 = tmp15 + tmp28
     tmp30 = tl.full([1, 1], 0, tl.int32)
     tmp31 = tl.maximum(tmp30, tmp29)
-    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +82,7 @@

     ynumel = 640
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176)
     return arg_10
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
