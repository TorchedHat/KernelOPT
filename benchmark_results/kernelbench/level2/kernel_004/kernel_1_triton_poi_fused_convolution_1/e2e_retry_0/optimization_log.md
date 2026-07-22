# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 284580 ns — **Best latency**: 235900 ns (**1.2064×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`shared_memory_tiled_transpose_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_004/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 284580 ns
- **SM Throughput**: 19.67 % | **Memory**: 79.21 % | **DRAM**: 72.51 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_for_coalesced_access | 1.312× | 1.206× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 284580 ns | 235900 ns | -48680 ns (-17.1%) |
| SM Throughput (%) | 19.67 % | 43.70 % | +24.03 pp |
| Memory Throughput (%) | 79.21 % | 52.52 % | -26.69 pp |
| DRAM Throughput (%) | 72.51 % | 21.38 % | -51.13 pp |
| Achieved Occupancy (%) | 94.04 % | 94.10 % | +0.06 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.04 % | 62.79 % | -17.25 pp |
| IPC Active | 0.79 inst/cycle | 1.50 inst/cycle | +0.71 |

---

## Cumulative Progress

```
Baseline:  284580 ns  (1.000×)
Iter 1     235900 ns  (1.206×)  — shared_memory_tiled_transpose_for_coalesced_access

Final best: 235900 ns  (1.2064×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 31500
+    # ynumel=4096 = batch*C = 32*128  (C is stride-1 in input)
+    # xnumel=31500 = H*W = 126*250   (W is stride-1 in output)
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # yindex: [None, YBLOCK] -> fast dim for LOADS (stride-1 in input along channel)
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
+    # xindex: [XBLOCK, 1] -> fast dim for STORES (stride-1 in output along W)
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
+    ymask = yindex < ynumel
     xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    y0 = (yindex % 128)       # channel index, stride-1 in input
+    y1 = yindex // 128        # batch index, stride-4032000 in input
+    x2 = xindex               # H*W flat index, stride-128 in input, stride-1 in output
+    # LOAD: consecutive threads (along YBLOCK) access y0 with stride-1 -> COALESCED
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, other=0.0)
+    # STORE: consecutive threads (along XBLOCK) access x2 with stride-1 -> COALESCED
+    tl.store(out_ptr0 + (x2 + 31500*yindex), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

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
