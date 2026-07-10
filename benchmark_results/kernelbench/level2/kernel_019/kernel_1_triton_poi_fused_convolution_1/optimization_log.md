# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286020 ns — **Best latency**: 285820 ns (**1.0007×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`memory_coalescing_swap_innermost_warp_dimension`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286020 ns
- **SM Throughput**: 19.70 % | **Memory**: 78.85 % | **DRAM**: 72.14 %
- **Achieved Occupancy**: 94.08 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.000× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286020 ns | 285730 ns | -290 ns (-0.1%) |
| SM Throughput (%) | 19.70 % | 19.78 % | +0.08 pp |
| Memory Throughput (%) | 78.85 % | 78.94 % | +0.09 pp |
| DRAM Throughput (%) | 72.14 % | 72.21 % | +0.07 pp |
| Achieved Occupancy (%) | 94.08 % | 94.08 % | +0.00 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.08 % | 80.35 % | +0.27 pp |
| IPC Active | 0.78 inst/cycle | 0.78 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.001× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cache_eviction_policy_streaming_load | 0.753× | 0.882× | ✓ |  |
| C | memory_coalescing_swap_innermost_warp_dimension | 1.001× | 1.001× | ✓ | **✓ best** |
| D | pipeline_l2_latency_with_evict_first_and_num_stages | 0.691× | 0.699× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cache_eviction_po... | Δ | memory_coalescing... | Δ | pipeline_l2_laten... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286020 ns | 324130 ns | +38110 ns (+13.3%) | 285820 ns | -200 ns (-0.1%) | 409380 ns | +123360 ns (+43.1%) |
| SM Throughput (%) | 19.70 % | 20.16 % | +0.46 pp | 19.34 % | -0.36 pp | 13.47 % | -6.23 pp |
| Memory Throughput (%) | 78.85 % | 69.59 % | -9.26 pp | 78.85 % | +0.00 pp | 60.23 % | -18.62 pp |
| DRAM Throughput (%) | 72.14 % | 63.61 % | -8.53 pp | 72.20 % | +0.06 pp | 57.82 % | -14.32 pp |
| Achieved Occupancy (%) | 94.08 % | 93.67 % | -0.41 pp | 94.06 % | -0.02 pp | 94.96 % | +0.88 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.08 % | 79.85 % | -0.23 pp | 80.03 % | -0.05 pp | 86.47 % | +6.39 pp |
| IPC Active | 0.78 inst/cycle | 0.80 inst/cycle | +0.02 | 0.80 inst/cycle | +0.02 | 0.54 inst/cycle | -0.24 |

---

## Iteration 3

**Best so far**: 1.001× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_via_num_stages_autotune | 0.999× | 1.001× | ✓ |  |
| B | coalesced_load_via_y_innermost_warp_dimension | 0.984× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | coalesced_load_vi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286020 ns | 285790 ns | -230 ns (-0.1%) | 288740 ns | +2720 ns (+1.0%) |
| SM Throughput (%) | 19.70 % | 19.47 % | -0.23 pp | 17.64 % | -2.06 pp |
| Memory Throughput (%) | 78.85 % | 78.89 % | +0.04 pp | 78.22 % | -0.63 pp |
| DRAM Throughput (%) | 72.14 % | 72.19 % | +0.05 pp | 71.48 % | -0.66 pp |
| Achieved Occupancy (%) | 94.08 % | 94.00 % | -0.08 pp | 93.94 % | -0.14 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.08 % | 80.42 % | +0.34 pp | 82.22 % | +2.14 pp |
| IPC Active | 0.78 inst/cycle | 0.80 inst/cycle | +0.02 | 0.71 inst/cycle | -0.07 |

---

## Iteration 4

**Best so far**: 1.001× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 0.999× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286020 ns | 284900 ns | -1120 ns (-0.4%) |
| SM Throughput (%) | 19.70 % | 19.39 % | -0.31 pp |
| Memory Throughput (%) | 78.85 % | 79.15 % | +0.30 pp |
| DRAM Throughput (%) | 72.14 % | 72.43 % | +0.29 pp |
| Achieved Occupancy (%) | 94.08 % | 94.07 % | -0.01 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.08 % | 80.03 % | -0.05 pp |
| IPC Active | 0.78 inst/cycle | 0.79 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  286020 ns  (1.000×)
Iter 1     285730 ns  (1.001×)  — vectorized_loads_with_contiguous_hints

Final best: 285820 ns  (1.0007×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=2, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
@@ -15,14 +31,12 @@

     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    x2 = xindex
+    x2 = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
