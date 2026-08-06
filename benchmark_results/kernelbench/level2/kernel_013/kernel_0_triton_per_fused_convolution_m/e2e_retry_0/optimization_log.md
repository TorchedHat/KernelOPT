# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_mean_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 510020 ns — **Best latency**: 508800 ns (**1.0024×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_via_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_mean_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_013/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 82.75
- **Registers/thread**: 40
- **Duration**: 510020 ns
- **SM Throughput**: 23.62 % | **Memory**: 88.40 % | **DRAM**: 88.40 %
- **Achieved Occupancy**: 72.75 %
- **IPC Active**: 0.94 inst/cycle | **No Eligible**: 76.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_xblock_autotune | 1.001× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 510020 ns | 508800 ns | -1220 ns (-0.2%) |
| SM Throughput (%) | 23.62 % | 25.95 % | +2.33 pp |
| Memory Throughput (%) | 88.40 % | 88.61 % | +0.21 pp |
| DRAM Throughput (%) | 88.40 % | 88.61 % | +0.21 pp |
| Achieved Occupancy (%) | 72.75 % | 72.95 % | +0.20 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 82.75 | +0.00 |
| No Eligible (%) | 76.57 % | 73.57 % | -3.00 pp |
| IPC Active | 0.94 inst/cycle | 1.05 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  510020 ns  (1.000×)
Iter 1     508800 ns  (1.002×)  — memory_coalescing_via_xblock_autotune

Final best: 508800 ns  (1.0024×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=3),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_convolution_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 16777216
@@ -14,7 +25,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
@@ -25,14 +36,12 @@

     x4 = xindex // 16384
     x1 = ((xindex // 16384) % 64)
     x5 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 16384*r0_3 + 524288*x4), None)
-    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0 + 16384*r0_3 + 524288*x4), xmask)
+    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
     tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x5), tmp5, None)
-
-
+    tl.store(out_ptr0 + (x5), tmp5, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +50,7 @@

     xnumel = 16777216
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32, XBLOCK=128)
+    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
