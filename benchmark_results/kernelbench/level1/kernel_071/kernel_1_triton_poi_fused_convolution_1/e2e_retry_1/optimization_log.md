# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285890 ns — **Best latency**: 142560 ns (**2.0054×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`shared_memory_tiled_transpose_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_071/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285890 ns
- **SM Throughput**: 19.68 % | **Memory**: 78.86 % | **DRAM**: 72.18 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.80 inst/cycle | **No Eligible**: 80.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_with_autotune | 1.937× | 2.005× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285890 ns | 142560 ns | -143330 ns (-50.1%) |
| SM Throughput (%) | 19.68 % | 48.14 % | +28.46 pp |
| Memory Throughput (%) | 78.86 % | 42.00 % | -36.86 pp |
| DRAM Throughput (%) | 72.18 % | 34.48 % | -37.70 pp |
| Achieved Occupancy (%) | 94.04 % | 89.34 % | -4.70 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.24 % | 57.00 % | -23.24 pp |
| IPC Active | 0.80 inst/cycle | 1.73 inst/cycle | +0.93 |

---

## Cumulative Progress

```
Baseline:  285890 ns  (1.000×)
Iter 1     142560 ns  (2.005×)  — shared_memory_tiled_transpose_with_autotune

Final best: 142560 ns  (2.0054×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +31,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask)
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
