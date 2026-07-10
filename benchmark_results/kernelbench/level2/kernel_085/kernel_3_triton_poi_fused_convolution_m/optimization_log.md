# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mul_native_group_norm_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 626750 ns — **Best latency**: 626750 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mul_native_group_norm_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_085/kernel_3_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1016064,1,1), Waves/SM: 481.09
- **Registers/thread**: 20
- **Duration**: 626750 ns
- **SM Throughput**: 52.60 % | **Memory**: 32.93 % | **DRAM**: 32.93 %
- **Achieved Occupancy**: 49.70 %
- **IPC Active**: 2.07 inst/cycle | **No Eligible**: 48.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_latency_hiding_and_occupancy | 1.000× | 1.000× | ✓ |  |
| B | increase_xblock_with_autotune_to_reduce_latency_stalls | 0.715× | 0.570× | ✓ |  |
| C | autotune_xblock_num_warps_num_stages | 0.997× | 0.860× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 1.000× | 0.140× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 626750 ns | 626880 ns | +130 ns (+0.0%) | 1100000 ns | +473250 ns (+75.5%) | 728480 ns | +101730 ns (+16.2%) | 4470000 ns | +3843250 ns (+613.2%) |
| SM Throughput (%) | 52.60 % | 65.66 % | +13.06 pp | 81.21 % | +28.61 pp | 79.49 % | +26.89 pp | 79.47 % | +26.87 pp |
| Memory Throughput (%) | 32.93 % | 32.91 % | -0.02 pp | 37.48 % | +4.55 pp | 41.34 % | +8.41 pp | 28.53 % | -4.40 pp |
| DRAM Throughput (%) | 32.93 % | 32.91 % | -0.02 pp | 18.75 % | -14.18 pp | 28.32 % | -4.61 pp | 4.62 % | -28.31 pp |
| Achieved Occupancy (%) | 49.70 % | 54.41 % | +4.71 pp | 76.17 % | +26.47 pp | 81.72 % | +32.02 pp | 90.44 % | +40.74 pp |
| Registers/Thread | 20.00 register/thread | 22.00 register/thread | +2 | 32.00 register/thread | +12 | 30.00 register/thread | +10 | 32.00 register/thread | +12 |
| Waves/SM | 481.09 | 481.09 | +0.00 | 481.09 | +0.00 | 481.09 | +0.00 | 1924.36 | +1443.27 |
| No Eligible (%) | 48.19 % | 34.25 % | -13.94 pp | 18.64 % | -29.55 pp | 20.23 % | -27.96 pp | 20.48 % | -27.71 pp |
| IPC Active | 2.07 inst/cycle | 2.63 inst/cycle | +0.56 | 3.25 inst/cycle | +1.18 | 3.19 inst/cycle | +1.12 | 3.18 inst/cycle | +1.11 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_reduce_latency_stalls` (0.570×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) on a pointwise kernel caused registers per thread to jump from 20 to 32, which reduced Block Limit Registers from 21 to 16 blocks/SM and increased total executed instructions by ~2.74x (256M → 701M). Although eligible warps per scheduler improved from 0.87 to 6.52 and achieved occupancy rose from 49.7% to 76.2%, the massive instruction count increase (likely from unrolled masking logic and register pressure) dominated, causing a 1.75x slowdown. The implementation also incorrectly changed xmask from tl.full([XBLOCK], True) to xindex < xnumel, adding unnecessary predication overhead on a kernel where xnumel is always a multiple of the original XBLOCK.

> **Applicable when**: Never apply this specific pattern (autotune XBLOCK increase + adding unnecessary masking) when the original kernel already uses tl.full([XBLOCK], True) as xmask, indicating xnumel is always divisible by XBLOCK and no masking is needed.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) the kernel already has low registers per thread (≤20) and increasing XBLOCK causes register pressure to rise (>28 registers/thread), (2) the kernel body uses tl.full([XBLOCK], True) as xmask meaning no out-of-bounds risk exists and adding xindex < xnumel masking introduces unnecessary predication, (3) the instruction count increase from unrolling larger tiles outweighs latency hiding benefits.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.999× | 0.997× | ✓ |  |
| B | persistent_tile_scheduling | 0.999× | 0.996× | ✓ |  |
| C | persistent_tile_scheduling | 0.999× | 0.999× | ✓ |  |
| D | fix_broken_autotune_grid_and_promote_fma_fusion | 1.000× | 0.438× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 626750 ns | 628640 ns | +1890 ns (+0.3%) | 629060 ns | +2310 ns (+0.4%) | 627650 ns | +900 ns (+0.1%) | 1430000 ns | +803250 ns (+128.2%) |
| SM Throughput (%) | 52.60 % | 77.73 % | +25.13 pp | 77.58 % | +24.98 pp | 69.71 % | +17.11 pp | 79.75 % | +27.15 pp |
| Memory Throughput (%) | 32.93 % | 32.82 % | -0.11 pp | 32.80 % | -0.13 pp | 32.88 % | -0.05 pp | 35.48 % | +2.55 pp |
| DRAM Throughput (%) | 32.93 % | 32.82 % | -0.11 pp | 32.80 % | -0.13 pp | 32.88 % | -0.05 pp | 14.45 % | -18.48 pp |
| Achieved Occupancy (%) | 49.70 % | 70.23 % | +20.53 pp | 70.44 % | +20.74 pp | 58.86 % | +9.16 pp | 71.53 % | +21.83 pp |
| Registers/Thread | 20.00 register/thread | 32.00 register/thread | +12 | 32.00 register/thread | +12 | 32.00 register/thread | +12 | 30.00 register/thread | +10 |
| Waves/SM | 481.09 | 481.09 | +0.00 | 481.09 | +0.00 | 481.09 | +0.00 | 962.18 | +481.09 |
| No Eligible (%) | 48.19 % | 22.06 % | -26.13 pp | 21.94 % | -26.25 pp | 30.18 % | -18.01 pp | 20.08 % | -28.11 pp |
| IPC Active | 2.07 inst/cycle | 3.12 inst/cycle | +1.05 | 3.12 inst/cycle | +1.05 | 2.79 inst/cycle | +0.72 | 3.19 inst/cycle | +1.12 |

---

## Cumulative Progress

```
Baseline:  626750 ns  (1.000×)

Final best: 626750 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_mul_native_group_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 130056192
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x3 = xindex
     x0 = (xindex % 64)
     x2 = xindex // 1016064
-    tmp0 = tl.load(in_out_ptr0 + (x3), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (16*x2 + (x0 // 4)), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (16*x2 + (x0 // 4)), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (16*x2 + (x0 // 4)), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (16*x2 + (x0 // 4)), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 63504.0, tl.float32)
@@ -32,9 +46,7 @@

     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp17 = tmp15 * tmp16
-    tl.store(in_out_ptr0 + (x3), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +58,7 @@

     arg_6 = inputs[6]
     xnumel = 130056192
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_mul_native_group_norm_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 130056192, XBLOCK=128)
+    triton_poi_fused_convolution_mul_native_group_norm_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 130056192)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
