# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3550 ns — **Best latency**: 3580 ns (**0.9916×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 22
- **Duration**: 3550 ns
- **SM Throughput**: 16.89 % | **Memory**: 9.26 % | **DRAM**: 9.26 %
- **Achieved Occupancy**: 54.13 %
- **IPC Active**: 1.62 inst/cycle | **No Eligible**: 56.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 1.001× | 0.992× | ✓ | **✓ best** |
| B | autotune_xblock_tile_size_for_better_occupancy_and_wave_coverage | 0.976× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3550 ns | 3580 ns | +30 ns (+0.8%) | 3620 ns | +70 ns (+2.0%) |
| SM Throughput (%) | 16.89 % | 19.27 % | +2.38 pp | 24.50 % | +7.61 pp |
| Memory Throughput (%) | 9.26 % | 9.19 % | -0.07 pp | 9.42 % | +0.16 pp |
| DRAM Throughput (%) | 9.26 % | 9.19 % | -0.07 pp | 9.18 % | -0.08 pp |
| Achieved Occupancy (%) | 54.13 % | 53.46 % | -0.67 pp | 53.81 % | -0.32 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 28.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 56.67 % | 51.42 % | -5.25 pp | 39.14 % | -17.53 pp |
| IPC Active | 1.62 inst/cycle | 1.86 inst/cycle | +0.24 | 2.43 inst/cycle | +0.81 |

---

## Cumulative Progress

```
Baseline:  3550 ns  (1.000×)

Final best: 3580 ns  (0.9916×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 200704
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x2), xmask)
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -33,9 +44,7 @@

     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +55,7 @@

     arg_5 = inputs[5]
     xnumel = 200704
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
