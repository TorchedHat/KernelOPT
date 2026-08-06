# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 3140 ns (**0.9873×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_2_triton_poi_fused_bitwise_not_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1152,1,1), Waves/SM: 0.55
- **Registers/thread**: 16
- **Duration**: 3100 ns
- **SM Throughput**: 8.58 % | **Memory**: 5.87 % | **DRAM**: 1.32 %
- **Achieved Occupancy**: 42.15 %
- **IPC Active**: 0.98 inst/cycle | **No Eligible**: 72.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_grid_and_improve_sm_utilization | 1.017× | 0.987× | ✓ | **✓ best** |
| B | autotune_xblock_for_occupancy_and_memory_coalescing | 0.993× | 1.054× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 3140 ns | +40 ns (+1.3%) | 2940 ns | -160 ns (-5.2%) |
| SM Throughput (%) | 8.58 % | 26.00 % | +17.42 pp | 15.13 % | +6.55 pp |
| Memory Throughput (%) | 5.87 % | 8.82 % | +2.95 pp | 5.49 % | -0.38 pp |
| DRAM Throughput (%) | 1.32 % | 1.32 % | +0.00 pp | 1.40 % | +0.08 pp |
| Achieved Occupancy (%) | 42.15 % | 42.48 % | +0.33 pp | 40.49 % | -1.66 pp |
| Registers/Thread | 16.00 register/thread | 29.00 register/thread | +13 | 22.00 register/thread | +6 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 72.68 % | 33.80 % | -38.88 pp | 47.73 % | -24.95 pp |
| IPC Active | 0.98 inst/cycle | 2.24 inst/cycle | +1.26 | 1.87 inst/cycle | +0.89 |

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)

Final best: 3140 ns  (0.9873×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 147456
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 3)
     x1 = ((xindex // 3) % 3)
     x3 = xindex // 3
     x2 = xindex // 9
     x4 = xindex
-    tmp6 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last')
+    tmp6 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), xmask, eviction_policy='evict_last')
     tmp0 = x0 + ((-1)*x1)
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 <= tmp1
@@ -28,16 +40,14 @@

     tmp9 = tl.full([1], float("-inf"), tl.float32)
     tmp10 = tl.where(tmp5, tmp9, tmp8)
     tmp11 = tl.exp(tmp10)
-    tl.store(out_ptr0 + (x4), tmp11, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp11, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 147456
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456, XBLOCK=128)
+    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
