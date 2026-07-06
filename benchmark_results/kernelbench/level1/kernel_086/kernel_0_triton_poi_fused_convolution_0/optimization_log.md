# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 5310 ns (**1.1394×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tl_trans_coalesced_store_for_nchw_nhwc_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_086/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 16.71 % | **Memory**: 40.57 % | **DRAM**: 20.32 %
- **Achieved Occupancy**: 54.98 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.203× | 0.995× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_to_improve_occupancy_and_hide_l1tex_latency | 1.118× | 0.985× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.101× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 6080 ns | +30 ns (+0.5%) | 6140 ns | +90 ns (+1.5%) | 6240 ns | +190 ns (+3.1%) |
| SM Throughput (%) | 16.71 % | 16.43 % | -0.28 pp | 17.18 % | +0.47 pp | 17.06 % | +0.35 pp |
| Memory Throughput (%) | 40.57 % | 41.07 % | +0.50 pp | 40.54 % | -0.03 pp | 40.20 % | -0.37 pp |
| DRAM Throughput (%) | 20.32 % | 20.24 % | -0.08 pp | 20.08 % | -0.24 pp | 19.69 % | -0.63 pp |
| Achieved Occupancy (%) | 54.98 % | 55.21 % | +0.23 pp | 55.70 % | +0.72 pp | 55.18 % | +0.20 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.44 % | 70.63 % | +0.19 pp | 71.65 % | +1.21 pp | 71.05 % | +0.61 pp |
| IPC Active | 1.16 inst/cycle | 1.14 inst/cycle | -0.02 | 1.14 inst/cycle | -0.02 | 1.15 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_transpose_for_coalesced_stores | 1.169× | 0.985× | ✓ |  |
| B | shared_memory_tile_transpose_for_coalesced_stores | 1.211× | 1.139× | ✓ | **✓ best** |
| C | streaming_load_evict_first_plus_vectorized_load_hints | 1.000× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | shared_memory_til... | Δ | streaming_load_ev... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 6140 ns | +90 ns (+1.5%) | 5310 ns | -740 ns (-12.2%) | 6110 ns | +60 ns (+1.0%) |
| SM Throughput (%) | 16.71 % | 16.17 % | -0.54 pp | 18.73 % | +2.02 pp | 16.84 % | +0.13 pp |
| Memory Throughput (%) | 40.57 % | 40.88 % | +0.31 pp | 40.22 % | -0.35 pp | 40.76 % | +0.19 pp |
| DRAM Throughput (%) | 20.32 % | 20.10 % | -0.22 pp | 23.14 % | +2.82 pp | 20.18 % | -0.14 pp |
| Achieved Occupancy (%) | 54.98 % | 55.63 % | +0.65 pp | 52.04 % | -2.94 pp | 55.30 % | +0.32 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.44 % | 71.32 % | +0.88 pp | 65.70 % | -4.74 pp | 71.93 % | +1.49 pp |
| IPC Active | 1.16 inst/cycle | 1.15 inst/cycle | -0.01 | 1.34 inst/cycle | +0.18 | 1.16 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.1394× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tl_trans_coalesced_store_for_nchw_nhwc_transpose | 0.956× | 1.139× | ✓ |  |
| C | tl_trans_register_tile_transpose_for_coalesced_stores | 1.168× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tl_trans_coalesce... | Δ | tl_trans_register... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 5310 ns | -740 ns (-12.2%) | 6140 ns | +90 ns (+1.5%) |
| SM Throughput (%) | 16.71 % | 19.21 % | +2.50 pp | 18.96 % | +2.25 pp |
| Memory Throughput (%) | 40.57 % | 39.84 % | -0.73 pp | 35.02 % | -5.55 pp |
| DRAM Throughput (%) | 20.32 % | 23.20 % | +2.88 pp | 20.07 % | -0.25 pp |
| Achieved Occupancy (%) | 54.98 % | 53.21 % | -1.77 pp | 52.63 % | -2.35 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.44 % | 65.15 % | -5.29 pp | 66.31 % | -4.13 pp |
| IPC Active | 1.16 inst/cycle | 1.35 inst/cycle | +0.19 | 1.32 inst/cycle | +0.16 |

---

## Iteration 4

**Best so far**: 1.1394× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 0.952× | 1.118× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6050 ns | 5410 ns | -640 ns (-10.6%) |
| SM Throughput (%) | 16.71 % | 18.61 % | +1.90 pp |
| Memory Throughput (%) | 40.57 % | 38.89 % | -1.68 pp |
| DRAM Throughput (%) | 20.32 % | 22.77 % | +2.45 pp |
| Achieved Occupancy (%) | 54.98 % | 52.59 % | -2.39 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.44 % | 66.05 % | -4.39 pp |
| IPC Active | 1.16 inst/cycle | 1.37 inst/cycle | +0.21 |

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 2     5310 ns  (1.139×)  — shared_memory_tile_transpose_for_coalesced_stores

Final best: 5310 ns  (1.1394×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +25,20 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), [XBLOCK]), [XBLOCK])[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 3)
-    y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
+    tmp0_T = tl.trans(tmp0)
+    xindex2 = xoffset + tl.arange(0, XBLOCK)[:, None]
+    yindex2 = yoffset + tl.arange(0, YBLOCK)[None, :]
+    xmask2 = xindex2 < xnumel
+    ymask2 = yindex2 < ynumel
+    x2b = xindex2
+    y0b = (yindex2 % 3)
+    y1b = yindex2 // 3
+    tl.store(out_ptr0 + (y0b + 3*x2b + 150528*y1b), tmp0_T, xmask2 & ymask2)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
