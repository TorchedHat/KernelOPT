# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_min_native_group_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1240000 ns — **Best latency**: 26880 ns (**46.1310×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_and_r0_block_right_sizing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_min_native_group_norm_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_075/kernel_2_triton_red_fused_min_native_gr/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.29
- **Registers/thread**: 32
- **Duration**: 1240000 ns
- **SM Throughput**: 10.76 % | **Memory**: 39.83 % | **DRAM**: 39.83 %
- **Achieved Occupancy**: 16.76 %
- **IPC Active**: 0.23 inst/cycle | **No Eligible**: 94.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_and_r0_block_right_sizing | 69.062× | 46.131× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1240000 ns | 26880 ns | -1213120 ns (-97.8%) |
| SM Throughput (%) | 10.76 % | 20.01 % | +9.25 pp |
| Memory Throughput (%) | 39.83 % | 67.58 % | +27.75 pp |
| DRAM Throughput (%) | 39.83 % | 14.55 % | -25.28 pp |
| Achieved Occupancy (%) | 16.76 % | 12.12 % | -4.64 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 |
| Waves/SM | 1.29 | 1.94 | +0.65 |
| No Eligible (%) | 94.12 % | 77.38 % | -16.74 pp |
| IPC Active | 0.23 inst/cycle | 0.91 inst/cycle | +0.68 |

---

## Cumulative Progress

```
Baseline:  1240000 ns  (1.000×)
Iter 1     26880 ns  (46.131×)  — memory_coalescing_and_r0_block_right_sizing

Final best: 26880 ns  (46.1310×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_min_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 65536
@@ -13,7 +26,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x3 = xindex
@@ -25,11 +38,11 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_2 // 16)), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_2 // 16)), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 - tmp1
         tmp4 = tl.full([1, 1], 16.0, tl.float32)
         tmp5 = (tmp3 / tmp4)
@@ -43,9 +56,7 @@

         tmp16 = tl.minimum(_tmp15, tmp14)
         _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
     tmp15 = triton_helpers.min2(_tmp15, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp15, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp15, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -57,7 +68,7 @@

     xnumel = 65536
     r0_numel = 128
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
