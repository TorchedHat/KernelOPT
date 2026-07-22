# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 49540 ns — **Best latency**: 7680 ns (**6.4505×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_via_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_013/kernel_1_triton_per_fused__softmax_add_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 3.88
- **Registers/thread**: 128
- **Duration**: 49540 ns
- **SM Throughput**: 42.68 % | **Memory**: 44.35 % | **DRAM**: 44.35 %
- **Achieved Occupancy**: 22.89 %
- **IPC Active**: 1.90 inst/cycle | **No Eligible**: 51.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_via_xblock_autotune | 5.570× | 6.450× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 49540 ns | 7680 ns | -41860 ns (-84.5%) |
| SM Throughput (%) | 42.68 % | 16.89 % | -25.79 pp |
| Memory Throughput (%) | 44.35 % | 38.91 % | -5.44 pp |
| DRAM Throughput (%) | 44.35 % | 1.41 % | -42.94 pp |
| Achieved Occupancy (%) | 22.89 % | 70.15 % | +47.26 pp |
| Registers/Thread | 128.00 register/thread | 18.00 register/thread | -110 |
| Waves/SM | 3.88 | 0.97 | -2.91 |
| No Eligible (%) | 51.62 % | 70.41 % | +18.79 pp |
| IPC Active | 1.90 inst/cycle | 1.10 inst/cycle | -0.80 |

---

## Cumulative Progress

```
Baseline:  49540 ns  (1.000×)
Iter 1     7680 ns  (6.450×)  — memory_coalescing_via_xblock_autotune

Final best: 7680 ns  (6.4505×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 262144
@@ -14,18 +28,18 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = tl.full([1, R0_BLOCK], True, tl.int1)
     roffset = r0_offset
     rindex = r0_index
     r0_2 = r0_index
     x0 = (xindex % 16384)
     x1 = xindex // 16384
     x3 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (x0 + 16384*r0_2 + 1048576*x1), None)
-    tmp3 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x0 + 16384*r0_2 + 1048576*x1), xmask)
+    tmp3 = tl.load(in_ptr0 + (r0_2), xmask, eviction_policy='evict_last')
     tmp1 = tl.full([1, 1], 32.0, tl.float32)
     tmp2 = (tmp0 / tmp1)
     tmp4 = tmp2 + tmp3
@@ -42,9 +56,7 @@

     tmp18 = libdevice.tanh(tmp17)
     tmp19 = tl.full([1, 1], 2.0, tl.float32)
     tmp20 = tmp18 * tmp19
-    tl.store(in_out_ptr0 + (x0 + 16384*r0_2 + 1048576*x1), tmp20, None)
-
-
+    tl.store(in_out_ptr0 + (x0 + 16384*r0_2 + 1048576*x1), tmp20, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -52,7 +64,7 @@

     xnumel = 262144
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1[grid](arg_0, arg_1, 262144, 64, XBLOCK=128)
+    triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1[grid](arg_0, arg_1, 262144, 64)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
