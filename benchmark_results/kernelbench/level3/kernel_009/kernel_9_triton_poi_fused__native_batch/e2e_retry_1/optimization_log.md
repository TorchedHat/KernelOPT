# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 26750 ns (**0.2800×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7490 ns
- **SM Throughput**: 39.48 % | **Memory**: 17.28 % | **DRAM**: 11.00 %
- **Achieved Occupancy**: 58.70 %
- **IPC Active**: 2.21 inst/cycle | **No Eligible**: 42.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_hide_l1tex_latency | 0.831× | 0.770× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.026× | 0.280× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 9730 ns | +2240 ns (+29.9%) | 26750 ns | +19260 ns (+257.1%) |
| SM Throughput (%) | 39.48 % | 66.99 % | +27.51 pp | 81.81 % | +42.33 pp |
| Memory Throughput (%) | 17.28 % | 18.19 % | +0.91 pp | 17.20 % | -0.08 pp |
| DRAM Throughput (%) | 11.00 % | 8.45 % | -2.55 pp | 3.06 % | -7.94 pp |
| Achieved Occupancy (%) | 58.70 % | 81.47 % | +22.77 pp | 54.92 % | -3.78 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 48.00 register/thread | +31 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 11.88 | +8.17 |
| No Eligible (%) | 42.70 % | 18.65 % | -24.05 pp | 11.15 % | -31.55 pp |
| IPC Active | 2.21 inst/cycle | 3.33 inst/cycle | +1.12 | 3.53 inst/cycle | +1.32 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_reduce_partial_wave_and_hide_l1tex_latency` (0.770×)

Adding autotune with larger XBLOCK values (512–2048) increased occupancy (58.7%→81.5%) and eligible warps (1.36→6.24), but the kernel duration increased by 30% (7.49→9.73 µs). The executed instruction count more than doubled (2.29M→5.03M) and registers per thread nearly doubled (17→32), indicating the autotuner selected a large XBLOCK config that introduced significant instruction overhead and register pressure without a commensurate reduction in memory latency stalls (warp cycles per issued instruction barely changed: 15.79→15.55). The grid was hardcoded to cdiv(xnumel, 128) in the launch wrapper but the kernel was called without XBLOCK, so the autotuner's selected tile size conflicted with the fixed grid, causing each block to process more elements than the grid covered—resulting in redundant work or incorrect coverage that inflated instruction count.

> **Applicable when**: Never apply this exact pattern. The strategy of increasing XBLOCK via autotune is sound in principle, but requires the launch grid to be a lambda using meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK while keeping a hardcoded fixed grid (cdiv(xnumel, FIXED_BLOCK)). The grid must be updated to lambda meta: (cdiv(xnumel, meta['XBLOCK']),) simultaneously. Failing to do so causes the autotuner to select a large XBLOCK while the grid remains sized for the small block, resulting in each block processing far more elements than intended, inflating instruction count (2.3M→5.0M) and register usage (17→32 regs/thread) without reducing latency stalls.

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)

Final best: 26750 ns  (0.2800×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
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
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -31,9 +44,7 @@

     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +54,7 @@

     arg_4 = inputs[4]
     xnumel = 1003520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1003520, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1003520)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
