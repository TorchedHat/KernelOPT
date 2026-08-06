# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3230 ns — **Best latency**: 3170 ns (**1.0189×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_to_correct_3d_and_vectorize_with_xblock16`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_4_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 38
- **Duration**: 3230 ns
- **SM Throughput**: 0.07 % | **Memory**: 0.34 % | **DRAM**: 0.05 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.49 inst/cycle | **No Eligible**: 88.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_grid_launch_configuration | 1.012× | 0.979× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3230 ns | 3300 ns | +70 ns (+2.2%) |
| SM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Memory Throughput (%) | 0.34 % | 0.33 % | -0.01 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 38.00 register/thread | 38.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.30 % | 86.64 % | -1.66 pp |
| IPC Active | 0.49 inst/cycle | 0.51 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | rewrite_kernel_function_with_correct_2d_grid_and_fixed_tile_sizes | 1.044× | 0.953× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | rewrite_kernel_fu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3230 ns | 3390 ns | +160 ns (+5.0%) |
| SM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Memory Throughput (%) | 0.34 % | 0.39 % | +0.05 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 38.00 register/thread | 38.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.30 % | 87.44 % | -0.86 pp |
| IPC Active | 0.49 inst/cycle | 0.52 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_to_correct_3d_and_vectorize_with_xblock16 | 1.052× | 1.019× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_to_corre... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3230 ns | 3170 ns | -60 ns (-1.9%) |
| SM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Memory Throughput (%) | 0.34 % | 0.34 % | +0.00 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 38.00 register/thread | 38.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.30 % | 87.08 % | -1.22 pp |
| IPC Active | 0.49 inst/cycle | 0.51 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.0189× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_correct_3d_preserving_exact_kernel_body | 0.997× | 1.029× | ✓ |  |
| B | rewrite_kernel_with_explicit_stride_based_index_arithmetic_and_3d_grid | 0.999× | 1.162× | ✓ |  |
| D | fix_grid_with_correct_3d_decomposition_matching_kernel_pid_logic | 1.005× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_corre... | Δ | rewrite_kernel_wi... | Δ | fix_grid_with_cor... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3230 ns | 3140 ns | -90 ns (-2.8%) | 2780 ns | -450 ns (-13.9%) | 3200 ns | -30 ns (-0.9%) |
| SM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.08 % | +0.01 pp | 0.07 % | +0.00 pp |
| Memory Throughput (%) | 0.34 % | 0.34 % | +0.00 pp | 0.39 % | +0.05 pp | 0.46 % | +0.12 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp | 0.06 % | +0.01 pp | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 6.23 % | -0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 38.00 register/thread | 38.00 register/thread | +0 | 36.00 register/thread | -2 | 38.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.30 % | 86.26 % | -2.04 pp | 83.34 % | -4.96 pp | 86.92 % | -1.38 pp |
| IPC Active | 0.49 inst/cycle | 0.52 inst/cycle | +0.03 | 0.63 inst/cycle | +0.14 | 0.47 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0189× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_wrapper_to_2d_grid_matching_kernel_pid_usage | 0.961× | 1.145× | ✓ |  |
| C | increase_tile_sizes_with_correct_2d_grid_and_stride_based_indexing | 0.990× | 1.133× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_wrappe... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3230 ns | 2820 ns | -410 ns (-12.7%) | 2850 ns | -380 ns (-11.8%) |
| SM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.07 % | +0.00 pp |
| Memory Throughput (%) | 0.34 % | 0.39 % | +0.05 pp | 0.38 % | +0.04 pp |
| DRAM Throughput (%) | 0.05 % | 0.06 % | +0.01 pp | 0.06 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 6.23 % | -0.01 pp |
| Registers/Thread | 38.00 register/thread | 36.00 register/thread | -2 | 32.00 register/thread | -6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.30 % | 83.90 % | -4.40 pp | 87.40 % | -0.90 pp |
| IPC Active | 0.49 inst/cycle | 0.55 inst/cycle | +0.06 | 0.48 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  3230 ns  (1.000×)
Iter 3     3170 ns  (1.019×)  — fix_grid_to_correct_3d_and_vectorize_with_xblock16

Final best: 3170 ns  (1.0189×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,31 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 102400
-    xnumel = 16
-    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
+    # Output: (N=4096, C=16, OH=5, OW=5), strides=(400, 25, 5, 1)
+    # Input:  (N=4096, C=16, IH=10, IW=10), strides=(1600, 1, 160, 16)
+    # Max-pool 2x2 stride=2
+    yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    x3 = xindex
-    y0 = (yindex % 5)
-    y4 = yindex // 5
-    y2 = yindex // 25
-    y5 = (yindex % 25)
-    tmp0 = tl.load(in_ptr0 + (x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (16 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (160 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (176 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
-    tmp2 = tl.maximum(tmp0, tmp1)
-    tmp4 = tl.maximum(tmp2, tmp3)
-    tmp6 = tl.maximum(tmp4, tmp5)
-    tl.store(out_ptr0 + (y5 + 25*x3 + 400*y2), tmp6, xmask & ymask)
-
-
+    # Decode yindex -> (n, oh, ow): yindex = n*25 + oh*5 + ow
+    n  = yindex // 25
+    rem = yindex % 25
+    oh = rem // 5
+    ow = rem % 5
+    c = xindex  # channel index
+    # Input base offset: n*1600 + c*1 + oh*2*160 + ow*2*16
+    base = n * 1600 + c + oh * 320 + ow * 32
+    mask = xmask & ymask
+    # Load 2x2 pool window: offsets 0, +16, +160, +176
+    v00 = tl.load(in_ptr0 + base,       mask=mask, other=-3.4e38)
+    v01 = tl.load(in_ptr0 + base + 16,  mask=mask, other=-3.4e38)
+    v10 = tl.load(in_ptr0 + base + 160, mask=mask, other=-3.4e38)
+    v11 = tl.load(in_ptr0 + base + 176, mask=mask, other=-3.4e38)
+    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
+    # Output offset: n*400 + c*25 + oh*5 + ow
+    out_offset = n * 400 + c * 25 + oh * 5 + ow
+    tl.store(out_ptr0 + out_offset, result, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
