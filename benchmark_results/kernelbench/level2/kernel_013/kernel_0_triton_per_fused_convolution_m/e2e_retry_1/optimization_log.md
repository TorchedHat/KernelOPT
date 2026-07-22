# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_mean_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 508900 ns — **Best latency**: 509540 ns (**0.9987×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_mean_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_013/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 82.75
- **Registers/thread**: 40
- **Duration**: 508900 ns
- **SM Throughput**: 23.55 % | **Memory**: 88.60 % | **DRAM**: 88.60 %
- **Achieved Occupancy**: 72.87 %
- **IPC Active**: 0.94 inst/cycle | **No Eligible**: 76.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.001× | 0.999× | ✓ | **✓ best** |
| B | memory_coalescing_autotune_xblock | 0.702× | 0.698× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 508900 ns | 509540 ns | +640 ns (+0.1%) | 728700 ns | +219800 ns (+43.2%) |
| SM Throughput (%) | 23.55 % | 25.83 % | +2.28 pp | 35.34 % | +11.79 pp |
| Memory Throughput (%) | 88.60 % | 88.48 % | -0.12 pp | 68.28 % | -20.32 pp |
| DRAM Throughput (%) | 88.60 % | 88.48 % | -0.12 pp | 61.86 % | -26.74 pp |
| Achieved Occupancy (%) | 72.87 % | 72.99 % | +0.12 pp | 36.18 % | -36.69 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 80.00 register/thread | +40 |
| Waves/SM | 82.75 | 82.75 | +0.00 | 165.49 | +82.74 |
| No Eligible (%) | 76.23 % | 73.60 % | -2.63 pp | 64.85 % | -11.38 pp |
| IPC Active | 0.94 inst/cycle | 1.06 inst/cycle | +0.12 | 1.41 inst/cycle | +0.47 |

---

## Cumulative Progress

```
Baseline:  508900 ns  (1.000×)

Final best: 509540 ns  (0.9987×)
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
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused_convolution_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 16777216
@@ -14,25 +25,24 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
     roffset = r0_offset
     rindex = r0_index
     r0_3 = r0_index
-    x0 = (xindex % 16384)
+    x0_raw = (xindex % 16384)
+    x0 = tl.max_contiguous(tl.multiple_of(x0_raw, [16, 1]), [16, 1])
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
@@ -41,7 +51,7 @@

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
