# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27550 ns — **Best latency**: 11040 ns (**2.4955×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`software_pipelining_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_1_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6144,1,1), Waves/SM: 3.88
- **Registers/thread**: 40
- **Duration**: 27550 ns
- **SM Throughput**: 38.28 % | **Memory**: 83.36 % | **DRAM**: 65.60 %
- **Achieved Occupancy**: 65.68 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | software_pipelining_with_autotune | 3.059× | 2.495× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 27550 ns | 11040 ns | -16510 ns (-59.9%) |
| SM Throughput (%) | 38.28 % | 29.32 % | -8.96 pp |
| Memory Throughput (%) | 83.36 % | 65.66 % | -17.70 pp |
| DRAM Throughput (%) | 65.60 % | 51.90 % | -13.70 pp |
| Achieved Occupancy (%) | 65.68 % | 81.30 % | +15.62 pp |
| Registers/Thread | 40.00 register/thread | 22.00 register/thread | -18 |
| Waves/SM | 3.88 | 2.91 | -0.97 |
| No Eligible (%) | 56.44 % | 63.23 % | +6.79 pp |
| IPC Active | 1.75 inst/cycle | 1.48 inst/cycle | -0.27 |

---

## Iteration 2

**Best so far**: 2.4955× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints_on_y_dimension | 3.008× | 2.518× | ✓ |  |
| C | eviction_policy_streaming_optimization | 2.890× | 2.502× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | eviction_policy_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27550 ns | 10940 ns | -16610 ns (-60.3%) | 11010 ns | -16540 ns (-60.0%) |
| SM Throughput (%) | 38.28 % | 29.24 % | -9.04 pp | 29.75 % | -8.53 pp |
| Memory Throughput (%) | 83.36 % | 65.48 % | -17.88 pp | 66.65 % | -16.71 pp |
| DRAM Throughput (%) | 65.60 % | 52.30 % | -13.30 pp | 51.58 % | -14.02 pp |
| Achieved Occupancy (%) | 65.68 % | 81.20 % | +15.52 pp | 81.14 % | +15.46 pp |
| Registers/Thread | 40.00 register/thread | 22.00 register/thread | -18 | 22.00 register/thread | -18 |
| Waves/SM | 3.88 | 2.91 | -0.97 | 2.91 | -0.97 |
| No Eligible (%) | 56.44 % | 63.62 % | +7.18 pp | 62.80 % | +6.36 pp |
| IPC Active | 1.75 inst/cycle | 1.47 inst/cycle | -0.28 | 1.47 inst/cycle | -0.28 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguous_hints_on_y_dimension` (2.518×)

Adding tl.max_contiguous and tl.multiple_of compiler hints to the Y and X index computations produced a 2.52x speedup in wall-clock benchmarking, but the NCU profiling metrics are essentially unchanged: duration dropped only from 11.04 µs to 10.94 µs (~1%), excessive sectors remain at 59%, load bytes utilized per sector stays at 11.9/32, and store bytes utilized per sector stays at 21.3/32. The speedup measured by the benchmark harness is therefore attributable to the previously applied autotune optimization (iteration 1, item exp_triton_autotune_num_stages_register_reduction_001) rather than the contiguous hints, which produced no measurable improvement in any NCU metric. When the dominant access pattern has non-unit strides (stride-2 loads, stride-6 stores), tl.max_contiguous/tl.multiple_of hints do not enable vectorization because the compiler cannot widen transactions across non-contiguous addresses regardless of index contiguity hints.

> **Applicable when**: Never: this strategy produced no measurable improvement in any NCU metric when the underlying access pattern has non-unit strides. The benchmark speedup (2.52x) is a carryover from the prior autotune optimization, not from these hints.

> **Avoid when**: When the kernel has non-unit-stride global loads or stores (e.g., stride-2 loads, stride-6 stores): tl.max_contiguous and tl.multiple_of hints cannot enable vectorization because the addresses are not physically contiguous in memory. The hints only affect compiler reasoning about index contiguity, not the actual memory layout. Do not apply as a standalone optimization when excessive sectors and bytes-utilized-per-sector metrics remain unchanged after application.

#### ✅ Plan C: `eviction_policy_streaming_optimization` (2.502×)

Switching tl.load eviction_policy from 'evict_last' to 'evict_first' on streaming workloads where each input element is consumed exactly once (e.g., pooling, elementwise ops) is theoretically motivated by reducing L2 cache pollution from dead data. However, the profiling data shows that the NCU metrics barely changed between before and after (Duration: 11.04→11.01 µs, L2 Hit Rate: 41.35→41.94%, No-Eligible: 63.23→62.8%), indicating the eviction policy change alone had negligible impact on the kernel's actual bottlenecks. The reported 2.5x speedup appears to reflect the cumulative improvement from the prior iteration's autotune change (which already brought duration from 27.55→11.04 µs), not from this 4-line eviction policy diff.

> **Applicable when**: Triton kernel performs a streaming read pattern where each input element is consumed exactly once (no spatial or temporal reuse across tiles), L2 Cache Throughput is high (>70%) but L2 Hit Rate is low (<50%), and the dominant bottleneck is L1TEX scoreboard stalls (CPIStall rule triggered, >70% of stall cycles from L1TEX). The eviction_policy='evict_last' hint is currently used on loads that have no reuse benefit.

> **Avoid when**: Kernel has genuine data reuse (e.g., convolution with overlapping receptive fields, attention with repeated key/value access) where L2 retention provides real cache hits. Also avoid when the kernel's dominant bottleneck is uncoalesced access patterns or compute — eviction policy changes cannot fix stride-induced sector waste and will have negligible effect if L2 hit rate is already low due to structural access pattern issues rather than cache pollution.

---

## Iteration 3

**Best so far**: 2.4955× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  27550 ns  (1.000×)
Iter 1     11040 ns  (2.495×)  — software_pipelining_with_autotune

Final best: 11040 ns  (2.4955×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
@@ -21,16 +37,14 @@

     x5 = xindex
     y0 = (yindex % 6)
     y1 = yindex // 6
-    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
-    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, xmask)
-
-
+    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, mask=ymask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -38,7 +52,7 @@

     ynumel = 24576
     xnumel = 196
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_max_pool2d_with_indices_relu_1[grid](arg_0, arg_1, 24576, 196, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_max_pool2d_with_indices_relu_1[grid](arg_0, arg_1, 24576, 196)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
