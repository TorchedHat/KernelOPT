# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_cat_expand_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3680 ns — **Best latency**: 5180 ns (**0.7104×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_cat_expand_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_2_triton_poi_fused_add_cat_expan/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3680 ns
- **SM Throughput**: 13.09 % | **Memory**: 7.87 % | **DRAM**: 6.80 %
- **Achieved Occupancy**: 51.77 %
- **IPC Active**: 1.28 inst/cycle | **No Eligible**: 67.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_persistent_tile_efficiency | 1.075× | 0.762× | ✓ |  |
| B | increase_xblock_with_autotune_to_improve_occupancy_and_hide_memory_latency | 1.089× | 0.710× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3680 ns | 4830 ns | +1150 ns (+31.2%) | 5180 ns | +1500 ns (+40.8%) |
| SM Throughput (%) | 13.09 % | 40.61 % | +27.52 pp | 43.45 % | +30.36 pp |
| Memory Throughput (%) | 7.87 % | 12.66 % | +4.79 pp | 12.16 % | +4.29 pp |
| DRAM Throughput (%) | 6.80 % | 5.13 % | -1.67 pp | 4.80 % | -2.00 pp |
| Achieved Occupancy (%) | 51.77 % | 70.17 % | +18.40 pp | 75.56 % | +23.79 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 24.00 register/thread | +8 |
| Waves/SM | 0.75 | 1.49 | +0.74 | 2.98 | +2.23 |
| No Eligible (%) | 67.35 % | 36.88 % | -30.47 pp | 34.27 % | -33.08 pp |
| IPC Active | 1.28 inst/cycle | 2.46 inst/cycle | +1.18 | 2.59 inst/cycle | +1.31 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_improve_occupancy_and_hide_memory_latency` (0.710×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a small pointwise kernel caused a 29% regression despite improving occupancy (51.8%→75.6%) and scheduler eligibility (67.4%→34.3% no-eligible). The kernel's total instruction count ballooned 4.6x (365K→1.69M executed instructions) because the autotune selected XBLOCK=512 with 4x more threads (201K→806K), doing far more redundant work per element due to the partial-wave tail effect (33.3% tail overhead flagged by LaunchConfiguration rule). The register pressure also increased (16→24 registers/thread), limiting the number of concurrent blocks per SM. For very short-duration kernels (<5 µs) with small total work, the overhead of larger blocks and the tail-wave penalty can outweigh the latency-hiding benefit of higher occupancy.

> **Applicable when**: Kernel has low waves_per_sm (<1.0), high no-eligible-warp % (>60%), and L1TEX stall is the dominant stall reason; total work is large enough that tail-wave overhead is negligible (<5% of runtime).

> **Avoid when**: Kernel duration is very short (<5 µs) and total element count is small (xnumel ~200K); increasing block size causes register pressure increase (>20 regs/thread) that limits concurrent blocks; the grid is not updated to use meta['XBLOCK'] in the lambda (implementation flaw that keeps grid fixed while XBLOCK changes); LaunchConfiguration tail-wave penalty is >20% of runtime at the chosen block size.

---

## Cumulative Progress

```
Baseline:  3680 ns  (1.000×)

Final best: 5180 ns  (0.7104×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_cat_expand_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 201728
@@ -16,22 +26,20 @@

     x2 = xindex // 100864
     x3 = (xindex % 100864)
     x4 = xindex
-    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
+    tmp11 = tl.load(in_ptr2 + (x3), mask=xmask, eviction_policy='evict_last')
     tmp0 = x1
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 1, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (x0), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 197, tl.int64)
     tmp8 = tmp0 < tmp7
-    tmp9 = tl.load(in_ptr1 + (x0 + 512*((-1) + x1) + 100352*x2), tmp6 & xmask, other=0.0)
+    tmp9 = tl.load(in_ptr1 + (x0 + 512*((-1) + x1) + 100352*x2), mask=tmp6 & xmask, other=0.0)
     tmp10 = tl.where(tmp4, tmp5, tmp9)
     tmp12 = tmp10 + tmp11
-    tl.store(out_ptr0 + (x4), tmp12, xmask)
-
-
+    tl.store(out_ptr0 + (x4), tmp12, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -40,7 +48,7 @@

     arg_3 = inputs[3]
     xnumel = 201728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_cat_expand_view_2[grid](arg_0, arg_1, arg_2, arg_3, 201728, XBLOCK=128)
+    triton_poi_fused_add_cat_expand_view_2[grid](arg_0, arg_1, arg_2, arg_3, 201728)
     return arg_3
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
