# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 6560 ns (**0.8979×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_043/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.76 % | **Memory**: 42.12 % | **DRAM**: 20.88 %
- **Achieved Occupancy**: 55.13 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.030× | 1.000× | ✓ |  |
| B | fix_uncoalesced_stores_via_shared_memory_transpose_and_autotune | 1.070× | 0.898× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 5890 ns | +0 ns (+0.0%) | 6560 ns | +670 ns (+11.4%) |
| SM Throughput (%) | 15.76 % | 16.94 % | +1.18 pp | 26.19 % | +10.43 pp |
| Memory Throughput (%) | 42.12 % | 41.93 % | -0.19 pp | 41.73 % | -0.39 pp |
| DRAM Throughput (%) | 20.88 % | 20.95 % | +0.07 pp | 10.04 % | -10.84 pp |
| Achieved Occupancy (%) | 55.13 % | 55.43 % | +0.30 pp | 48.10 % | -7.03 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.06 % | 70.77 % | -0.29 pp | 52.13 % | -18.93 pp |
| IPC Active | 1.12 inst/cycle | 1.12 inst/cycle | +0.00 | 1.80 inst/cycle | +0.68 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)

Final best: 6560 ns  (0.8979×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -20,9 +32,11 @@

     y0 = (yindex % 3)
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    # FIX: split into 3 per-channel stores so each store covers a contiguous
+    # x-strip; the compiler can schedule/vectorize each strip independently.
+    for c in tl.static_range(3):
+        row_mask = (y0 == c) & ymask
+        tl.store(out_ptr0 + (c + 3*x2 + 150528*y1), tmp0, xmask & row_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
