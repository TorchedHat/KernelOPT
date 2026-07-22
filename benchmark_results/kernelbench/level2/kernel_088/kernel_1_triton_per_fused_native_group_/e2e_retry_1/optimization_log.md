# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11140 ns — **Best latency**: 7170 ns (**1.5537×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_via_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_group_norm_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_088/kernel_1_triton_per_fused_native_group_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 11140 ns
- **SM Throughput**: 27.92 % | **Memory**: 64.00 % | **DRAM**: 64.00 %
- **Achieved Occupancy**: 82.68 %
- **IPC Active**: 1.36 inst/cycle | **No Eligible**: 64.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_xblock_autotune | 1.448× | 1.554× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 11140 ns | 7170 ns | -3970 ns (-35.6%) |
| SM Throughput (%) | 27.92 % | 28.42 % | +0.50 pp |
| Memory Throughput (%) | 64.00 % | 47.87 % | -16.13 pp |
| DRAM Throughput (%) | 64.00 % | 47.87 % | -16.13 pp |
| Achieved Occupancy (%) | 82.68 % | 76.99 % | -5.69 pp |
| Registers/Thread | 32.00 register/thread | 20.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 64.51 % | 59.15 % | -5.36 pp |
| IPC Active | 1.36 inst/cycle | 1.60 inst/cycle | +0.24 |

---

## Cumulative Progress

```
Baseline:  11140 ns  (1.000×)
Iter 1     7170 ns  (1.554×)  — memory_coalescing_via_xblock_autotune

Final best: 7170 ns  (1.5537×)
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
+        triton.Config({'XBLOCK': 64},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 524288
@@ -22,7 +33,7 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0, eviction_policy='evict_first')
+    tmp0 = tl.load(in_ptr0 + (xindex * R0_BLOCK + r0_index), xmask, other=0.0, eviction_policy='evict_first')
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
     tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
@@ -34,7 +45,6 @@

     tl.store(out_ptr0 + (x0), tmp8, xmask, eviction_policy='evict_last')
     tl.store(out_ptr1 + (x0), tmp13, xmask, eviction_policy='evict_last')
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -42,7 +52,7 @@

     xnumel = 524288
     r0_numel = 16
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_native_group_norm_1[grid](arg_0, arg_1, arg_2, 524288, 16, XBLOCK=128)
+    triton_per_fused_native_group_norm_1[grid](arg_0, arg_1, arg_2, 524288, 16)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
