# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 581020 ns — **Best latency**: 376580 ns (**1.5429×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_alignment_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_001/kernel_2_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256000,1,1), Waves/SM: 121.21
- **Registers/thread**: 30
- **Duration**: 581020 ns
- **SM Throughput**: 27.35 % | **Memory**: 78.69 % | **DRAM**: 72.23 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 72.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 2.011× | 1.543× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 581020 ns | 376580 ns | -204440 ns (-35.2%) |
| SM Throughput (%) | 27.35 % | 38.25 % | +10.90 pp |
| Memory Throughput (%) | 78.69 % | 95.85 % | +17.16 pp |
| DRAM Throughput (%) | 72.23 % | 55.33 % | -16.90 pp |
| Achieved Occupancy (%) | 94.01 % | 93.71 % | -0.30 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 |
| Waves/SM | 121.21 | 121.21 | +0.00 |
| No Eligible (%) | 72.60 % | 61.84 % | -10.76 pp |
| IPC Active | 1.10 inst/cycle | 1.53 inst/cycle | +0.43 |

---

## Iteration 2

**Best so far**: 1.5429× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | coalesced_load_hints_with_higher_pipeline_stages | 1.746× | 1.247× | ✓ |  |
| D | eviction_policy_and_load_alignment_hints_for_strided_input | 1.746× | 1.236× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_load_hi... | Δ | eviction_policy_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 581020 ns | 466050 ns | -114970 ns (-19.8%) | 469920 ns | -111100 ns (-19.1%) |
| SM Throughput (%) | 27.35 % | 30.60 % | +3.25 pp | 30.55 % | +3.20 pp |
| Memory Throughput (%) | 78.69 % | 78.37 % | -0.32 pp | 77.71 % | -0.98 pp |
| DRAM Throughput (%) | 72.23 % | 54.51 % | -17.72 pp | 54.13 % | -18.10 pp |
| Achieved Occupancy (%) | 94.01 % | 94.48 % | +0.47 pp | 94.42 % | +0.41 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 121.21 | 121.21 | +0.00 | 121.21 | +0.00 |
| No Eligible (%) | 72.60 % | 69.41 % | -3.19 pp | 69.71 % | -2.89 pp |
| IPC Active | 1.10 inst/cycle | 1.22 inst/cycle | +0.12 | 1.23 inst/cycle | +0.13 |

---

## Iteration 3

**Best so far**: 1.5429× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_misapplied_alignment_hints_to_enable_vectorized_output_stores | 1.746× | 1.258× | ✓ |  |
| C | fma_fusion_for_non_fused_fp32_instructions | 1.746× | 1.249× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_misapplied_al... | Δ | fma_fusion_for_no... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 581020 ns | 461920 ns | -119100 ns (-20.5%) | 465150 ns | -115870 ns (-19.9%) |
| SM Throughput (%) | 27.35 % | 30.52 % | +3.17 pp | 30.36 % | +3.01 pp |
| Memory Throughput (%) | 78.69 % | 78.96 % | +0.27 pp | 78.42 % | -0.27 pp |
| DRAM Throughput (%) | 72.23 % | 54.84 % | -17.39 pp | 54.62 % | -17.61 pp |
| Achieved Occupancy (%) | 94.01 % | 94.42 % | +0.41 pp | 94.41 % | +0.40 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 121.21 | 121.21 | +0.00 | 121.21 | +0.00 |
| No Eligible (%) | 72.60 % | 69.10 % | -3.50 pp | 69.51 % | -3.09 pp |
| IPC Active | 1.10 inst/cycle | 1.23 inst/cycle | +0.13 | 1.23 inst/cycle | +0.13 |


### Key Learnings

#### ✅ Plan C: `fma_fusion_for_non_fused_fp32_instructions` (1.249×)

Replacing plain FP32 additions (a + b) with libdevice.fma(1.0, a, b) and a float literal in tl.maximum to eliminate an int-to-float cast did not reduce the executed instruction count (111,104,000 before and after) nor meaningfully change any profiling metric. Duration improved by only ~0.2% (466.05us → 465.15us), which is within measurement noise. The FPInstructions NCU rule predicted a ~4-5% speedup, but the Triton compiler apparently already fuses or schedules these operations equivalently, and the kernel remains bottlenecked by uncoalesced L2 memory accesses (L2 throughput ~93%), not instruction throughput. The autotune config changes (adding larger XBLOCK=1024/2048 configs) likely account for the observed 1.25x end-to-end speedup rather than the FMA substitution itself.

> **Applicable when**: Never: this substitution did not reduce instruction count or improve any profiling metric in a memory-bound Triton kernel. The FPInstructions NCU rule may fire but the actual PTX emitted by Triton already handles these additions efficiently.

> **Avoid when**: When the kernel is memory-bound (L2 throughput >90%) and the FPInstructions rule predicts only a small speedup (<5%). The Triton compiler does not reliably emit FFMA from libdevice.fma(1.0, a, b) — the executed instruction count was unchanged, confirming no instruction reduction occurred. Do not use this as a primary optimization strategy for memory-bound kernels.

---

## Cumulative Progress

```
Baseline:  581020 ns  (1.000×)
Iter 1     376580 ns  (1.543×)  — vectorized_loads_with_alignment_hints

Final best: 376580 ns  (1.5429×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_convolution_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
@@ -19,16 +37,20 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 2032128*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
+    # Apply alignment hint on the contiguous x2 dimension (stride=1 in out_ptr0)
+    # x2 has shape [1, XBLOCK], so values must be a list matching dimensions
+    x2_vec = tl.max_contiguous(tl.multiple_of(x2, [1, XBLOCK]), [1, XBLOCK])
+    # Apply alignment hint on y0 (stride-1 axis of in_ptr0, channel dimension)
+    # y0 has shape [YBLOCK, 1]; use list [YBLOCK, 1] to match 2D shape
+    y0_vec = tl.max_contiguous(tl.multiple_of(y0, [YBLOCK, 1]), [YBLOCK, 1])
+    tmp0 = tl.load(in_ptr0 + (y0_vec + 128*x2 + 2032128*y1), mask=xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1, 1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
     tmp6 = tmp4 + tmp5
-    tl.store(out_ptr0 + (x2 + 15876*y3), tmp6, xmask)
-
-
+    tl.store(out_ptr0 + (x2_vec + 15876*y3), tmp6, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -38,7 +60,7 @@

     ynumel = 16384
     xnumel = 15876
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_add_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_add_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
