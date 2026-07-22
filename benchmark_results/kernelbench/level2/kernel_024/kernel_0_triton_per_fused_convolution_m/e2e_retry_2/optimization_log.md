# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_min_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 64540 ns — **Best latency**: 19810 ns (**3.2580×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_via_autotune_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_min_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_024/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (21600,1,1), Waves/SM: 10.23
- **Registers/thread**: 32
- **Duration**: 64540 ns
- **SM Throughput**: 33.42 % | **Memory**: 80.33 % | **DRAM**: 80.33 %
- **Achieved Occupancy**: 95.32 %
- **IPC Active**: 1.40 inst/cycle | **No Eligible**: 64.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_autotune_xblock | 2.655× | 3.258× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 64540 ns | 19810 ns | -44730 ns (-69.3%) |
| SM Throughput (%) | 33.42 % | 55.92 % | +22.50 pp |
| Memory Throughput (%) | 80.33 % | 68.99 % | -11.34 pp |
| DRAM Throughput (%) | 80.33 % | 66.31 % | -14.02 pp |
| Achieved Occupancy (%) | 95.32 % | 85.82 % | -9.50 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 |
| Waves/SM | 10.23 | 5.11 | -5.12 |
| No Eligible (%) | 64.90 % | 40.54 % | -24.36 pp |
| IPC Active | 1.40 inst/cycle | 2.39 inst/cycle | +0.99 |

---

## Cumulative Progress

```
Baseline:  64540 ns  (1.000×)
Iter 1     19810 ns  (3.258×)  — memory_coalescing_via_autotune_xblock

Final best: 19810 ns  (3.2580×)
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
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=4),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 2764800
@@ -14,7 +26,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = r0_index < r0_numel
@@ -25,15 +37,13 @@

     x4 = xindex // 900
     x1 = ((xindex // 900) % 24)
     x5 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 900*r0_3 + 19800*x4), r0_mask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0 + 900*r0_3 + 19800*x4), xmask & r0_mask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
     tmp5 = tl.where(r0_mask, tmp3, float("inf"))
     tmp6 = triton_helpers.min2(tmp5, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x5), tmp6, None)
-
-
+    tl.store(out_ptr0 + (x5), tmp6, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +52,7 @@

     xnumel = 2764800
     r0_numel = 22
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_min_0[grid](arg_0, arg_1, arg_2, 2764800, 22, XBLOCK=128)
+    triton_per_fused_convolution_min_0[grid](arg_0, arg_1, arg_2, 2764800, 22)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
