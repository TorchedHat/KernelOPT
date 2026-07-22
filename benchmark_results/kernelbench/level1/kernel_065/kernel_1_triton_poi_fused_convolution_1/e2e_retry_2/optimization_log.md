# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286300 ns — **Best latency**: 144000 ns (**1.9882×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`shared_memory_tiled_transpose_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_065/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286300 ns
- **SM Throughput**: 19.61 % | **Memory**: 78.74 % | **DRAM**: 72.07 %
- **Achieved Occupancy**: 94.02 %
- **IPC Active**: 0.80 inst/cycle | **No Eligible**: 79.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_for_coalesced_access | 1.937× | 1.988× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286300 ns | 144000 ns | -142300 ns (-49.7%) |
| SM Throughput (%) | 19.61 % | 48.67 % | +29.06 pp |
| Memory Throughput (%) | 78.74 % | 42.39 % | -36.35 pp |
| DRAM Throughput (%) | 72.07 % | 34.25 % | -37.82 pp |
| Achieved Occupancy (%) | 94.02 % | 92.07 % | -1.95 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.77 % | 53.16 % | -26.61 pp |
| IPC Active | 0.80 inst/cycle | 1.83 inst/cycle | +1.03 |

---

## Cumulative Progress

```
Baseline:  286300 ns  (1.000×)
Iter 1     144000 ns  (1.988×)  — shared_memory_tiled_transpose_for_coalesced_access

Final best: 144000 ns  (1.9882×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8,  num_stages=3),
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
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_last')
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
