# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 242820 ns — **Best latency**: 243710 ns (**0.9963×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_12_triton_poi_fused_cat_clone_per/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 242820 ns
- **SM Throughput**: 39.21 % | **Memory**: 26.21 % | **DRAM**: 26.21 %
- **Achieved Occupancy**: 32.78 %
- **IPC Active**: 1.45 inst/cycle | **No Eligible**: 63.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_to_reduce_cta_count_and_hide_memory_latency | 0.924× | 0.941× | ✓ |  |
| B | increase_xblock_tile_size_for_latency_hiding | — | 0.996× | ✓ | **✓ best** |
| C | increase_xblock_with_autotune_to_hide_l1tex_latency | — | 0.357× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | — | 0.928× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 242820 ns | 258140 ns | +15320 ns (+6.3%) | 243710 ns | +890 ns (+0.4%) | 679460 ns | +436640 ns (+179.8%) | 261600 ns | +18780 ns (+7.7%) |
| SM Throughput (%) | 39.21 % | 50.16 % | +10.95 pp | 39.13 % | -0.08 pp | 75.17 % | +35.96 pp | 45.87 % | +6.66 pp |
| Memory Throughput (%) | 26.21 % | 24.50 % | -1.71 pp | 26.20 % | -0.01 pp | 11.53 % | -14.68 pp | 24.20 % | -2.01 pp |
| DRAM Throughput (%) | 26.21 % | 24.50 % | -1.71 pp | 26.20 % | -0.01 pp | 9.31 % | -16.90 pp | 24.20 % | -2.01 pp |
| Achieved Occupancy (%) | 32.78 % | 32.03 % | -0.75 pp | 40.98 % | +8.20 pp | 78.34 % | +45.56 pp | 32.55 % | -0.23 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 744.73 | +558.55 | 186.18 | +0.00 |
| No Eligible (%) | 63.61 % | 58.79 % | -4.82 pp | 64.40 % | +0.79 pp | 39.74 % | -23.87 pp | 60.61 % | -3.00 pp |
| IPC Active | 1.45 inst/cycle | 1.65 inst/cycle | +0.20 | 1.42 inst/cycle | -0.03 | 2.41 inst/cycle | +0.96 | 1.56 inst/cycle | +0.11 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_to_reduce_cta_count_and_improve_sm_utilization | — | 1.000× | ✓ |  |
| B | persistent_tile_scheduling | — | 0.006× | ✓ |  |
| C | persistent_tile_scheduling_via_fixed_autotune_grid_lambda | — | 0.925× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 242820 ns | 242820 ns | +0 ns (+0.0%) | 40870000 ns | +40627180 ns (+16731.4%) | 262620 ns | +19800 ns (+8.2%) |
| SM Throughput (%) | 39.21 % | 44.15 % | +4.94 pp | 60.86 % | +21.65 pp | 46.69 % | +7.48 pp |
| Memory Throughput (%) | 26.21 % | 26.20 % | -0.01 pp | 60.02 % | +33.81 pp | 24.12 % | -2.09 pp |
| DRAM Throughput (%) | 26.21 % | 26.20 % | -0.01 pp | 60.02 % | +33.81 pp | 24.12 % | -2.09 pp |
| Achieved Occupancy (%) | 32.78 % | 36.82 % | +4.04 pp | 98.33 % | +65.55 pp | 32.43 % | -0.35 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.61 % | 55.38 % | -8.23 pp | 45.36 % | -18.25 pp | 60.53 % | -3.08 pp |
| IPC Active | 1.45 inst/cycle | 1.79 inst/cycle | +0.34 | 2.19 inst/cycle | +0.74 | 1.57 inst/cycle | +0.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_broken_persistent_kernel_and_restore_correct_launch_grid | — | 0.687× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_persis... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 242820 ns | 353340 ns | +110520 ns (+45.5%) |
| SM Throughput (%) | 39.21 % | 72.74 % | +33.53 pp |
| Memory Throughput (%) | 26.21 % | 17.91 % | -8.30 pp |
| DRAM Throughput (%) | 26.21 % | 17.91 % | -8.30 pp |
| Achieved Occupancy (%) | 32.78 % | 74.88 % | +42.10 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 372.36 | +186.18 |
| No Eligible (%) | 63.61 % | 40.19 % | -23.42 pp |
| IPC Active | 1.45 inst/cycle | 2.40 inst/cycle | +0.95 |

---

## Cumulative Progress

```
Baseline:  242820 ns  (1.000×)

Final best: 243710 ns  (0.9963×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,39 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 50331648
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x1 = ((xindex // 1024) % 3)
-    x0 = (xindex % 1024)
-    x2 = ((xindex // 3072) % 8)
-    x3 = xindex // 24576
-    x4 = xindex
-    tmp0 = x1
-    tmp1 = tl.full([1], 0, tl.int64)
-    tmp2 = tmp0 >= tmp1
-    tmp3 = tl.full([1], 1, tl.int64)
-    tmp4 = tmp0 < tmp3
-    tmp5 = tl.full([1], 0.0, tl.float32)
-    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
-    tmp7 = tl.where(tmp4, tmp5, tmp6)
-    tmp8 = tmp0 >= tmp3
-    tmp9 = tl.full([1], 3, tl.int64)
-    tmp10 = tmp0 < tmp9
-    tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), tmp8, other=0.0)
-    tmp12 = tl.where(tmp4, tmp7, tmp11)
-    tl.store(out_ptr0 + (x4), tmp12, None)
-
-
+def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
+    # Persistent kernel: each CTA loops over multiple tiles
+    # NUM_PROGRAMS = 528 (132 SMs * 4 tiles/SM), XBLOCK = 1024
+    # stride = 528 * 1024 = 540672; tiles per CTA = 50331648 / 540672 = 93 exactly
+    pid = tl.program_id(0)
+    xoffset = pid * XBLOCK
+    num_programs = tl.num_programs(0)
+    stride = num_programs * XBLOCK  # 528 * 1024 = 540672
+    while xoffset < xnumel:  # loops 93 times per CTA
+        xindex = xoffset + tl.arange(0, XBLOCK)
+        xmask = tl.full([XBLOCK], True, tl.int1)  # safe: 50331648 % 540672 == 0
+        x1 = ((xindex // 1024) % 3)
+        x0 = (xindex % 1024)
+        x2 = ((xindex // 3072) % 8)
+        x3 = xindex // 24576
+        x4 = xindex
+        tmp0 = x1
+        tmp1 = tl.full([1], 0, tl.int64)
+        tmp2 = tmp0 >= tmp1
+        tmp3 = tl.full([1], 1, tl.int64)
+        tmp4 = tmp0 < tmp3
+        tmp5 = tl.full([1], 0.0, tl.float32)
+        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
+        tmp7 = tl.where(tmp4, tmp5, tmp6)
+        tmp8 = tmp0 >= tmp3
+        tmp9 = tl.full([1], 3, tl.int64)
+        tmp10 = tmp0 < tmp9
+        tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), mask=tmp8, other=0.0)
+        tmp12 = tl.where(tmp4, tmp7, tmp11)
+        tl.store(out_ptr0 + (x4), tmp12, mask=xmask)
+        xoffset += stride
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
