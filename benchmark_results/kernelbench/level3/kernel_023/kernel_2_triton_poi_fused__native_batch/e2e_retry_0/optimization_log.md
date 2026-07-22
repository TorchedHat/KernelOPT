# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 126270 ns — **Best latency**: 126500 ns (**0.9982×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (200704,1,1), Waves/SM: 95.03
- **Registers/thread**: 19
- **Duration**: 126270 ns
- **SM Throughput**: 62.80 % | **Memory**: 28.91 % | **DRAM**: 28.82 %
- **Achieved Occupancy**: 60.94 %
- **IPC Active**: 2.57 inst/cycle | **No Eligible**: 35.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.629× | 0.617× | ✓ |  |
| B | autotune_xblock_num_warps_for_pointwise_kernel | 0.999× | 0.999× | ✓ |  |
| C | add_triton_autotune | 1.000× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 126270 ns | 204670 ns | +78400 ns (+62.1%) | 126430 ns | +160 ns (+0.1%) | 126500 ns | +230 ns (+0.2%) |
| SM Throughput (%) | 62.80 % | 87.15 % | +24.35 pp | 77.03 % | +14.23 pp | 76.45 % | +13.65 pp |
| Memory Throughput (%) | 28.91 % | 27.33 % | -1.58 pp | 29.21 % | +0.30 pp | 29.00 % | +0.09 pp |
| DRAM Throughput (%) | 28.82 % | 17.77 % | -11.05 pp | 28.78 % | -0.04 pp | 28.76 % | -0.06 pp |
| Achieved Occupancy (%) | 60.94 % | 88.11 % | +27.17 pp | 65.98 % | +5.04 pp | 66.67 % | +5.73 pp |
| Registers/Thread | 19.00 register/thread | 32.00 register/thread | +13 | 20.00 register/thread | +1 | 20.00 register/thread | +1 |
| Waves/SM | 95.03 | 95.03 | +0.00 | 95.03 | +0.00 | 95.03 | +0.00 |
| No Eligible (%) | 35.72 % | 11.90 % | -23.82 pp | 21.39 % | -14.33 pp | 20.45 % | -15.27 pp |
| IPC Active | 2.57 inst/cycle | 3.52 inst/cycle | +0.95 | 3.13 inst/cycle | +0.56 | 3.14 inst/cycle | +0.57 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.617×)

Adding @triton.autotune with larger XBLOCK values (512–2048) to a pointwise fused kernel increased occupancy from 60.9% to 88.1% and reduced CPIStall pressure, but caused a 62% regression in wall time. The total executed instruction count increased 2.2x (62.6M → 139.1M) and registers per thread jumped from 19 to 32, indicating the autotuner selected a configuration that introduced significant extra computation or unrolling overhead. The net effect was that the compute throughput increase (62.8% → 87.1%) was outweighed by the much longer elapsed cycle count (189K → 303K cycles), suggesting the selected XBLOCK configuration added more work than it saved in memory latency.

> **Applicable when**: Kernel has very low achieved occupancy (<50%) AND high CPIStall share (>40%) AND low register usage (<30 registers/thread) AND the bottleneck is confirmed to be memory latency hiding rather than compute throughput.

> **Avoid when**: The kernel already has moderate-to-high occupancy (>60%) with a large grid (>100K blocks) and high waves-per-SM (>90). In such cases, increasing XBLOCK via autotune tends to increase register pressure and instruction count without proportional memory latency reduction, causing regression. Also avoid when the xmask change from unconditional (tl.full True) to conditional (xindex < xnumel) introduces predication overhead across all loads and stores.

---

## Cumulative Progress

```
Baseline:  126270 ns  (1.000×)

Final best: 126500 ns  (0.9982×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 25690112
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 1e-05, tl.float32)
@@ -33,9 +40,7 @@

     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +51,7 @@

     arg_5 = inputs[5]
     xnumel = 25690112
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 25690112, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 25690112)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
