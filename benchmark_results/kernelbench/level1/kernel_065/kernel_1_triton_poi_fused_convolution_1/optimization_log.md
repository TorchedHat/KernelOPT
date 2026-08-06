# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285760 ns — **Best latency**: 158780 ns (**1.7997×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_065/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285760 ns
- **SM Throughput**: 19.63 % | **Memory**: 78.85 % | **DRAM**: 72.22 %
- **Achieved Occupancy**: 94.02 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_fix_eviction_policy | 2.290× | 1.958× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285760 ns | 145920 ns | -139840 ns (-48.9%) |
| SM Throughput (%) | 19.63 % | 46.51 % | +26.88 pp |
| Memory Throughput (%) | 78.85 % | 40.45 % | -38.40 pp |
| DRAM Throughput (%) | 72.22 % | 33.07 % | -39.15 pp |
| Achieved Occupancy (%) | 94.02 % | 92.06 % | -1.96 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.60 % | 53.82 % | -25.78 pp |
| IPC Active | 0.81 inst/cycle | 1.80 inst/cycle | +0.99 |

---

## Iteration 2

**Best so far**: 1.9583× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_via_max_contiguous_multiple_of_hints | 2.304× | 1.963× | ✓ |  |
| B | shared_memory_tiled_transpose_for_coalesced_stores | 2.555× | 1.952× | ✓ | **✓ best** |
| C | evict_first_streaming_store_to_reduce_l2_write_pollution | 2.236× | 1.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | shared_memory_til... | Δ | evict_first_strea... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285760 ns | 145600 ns | -140160 ns (-49.0%) | 146400 ns | -139360 ns (-48.8%) | 145150 ns | -140610 ns (-49.2%) |
| SM Throughput (%) | 19.63 % | 46.30 % | +26.67 pp | 46.98 % | +27.35 pp | 47.68 % | +28.05 pp |
| Memory Throughput (%) | 78.85 % | 40.19 % | -38.66 pp | 40.78 % | -38.07 pp | 41.68 % | -37.17 pp |
| DRAM Throughput (%) | 72.22 % | 33.09 % | -39.13 pp | 32.98 % | -39.24 pp | 33.54 % | -38.68 pp |
| Achieved Occupancy (%) | 94.02 % | 92.02 % | -2.00 pp | 91.97 % | -2.05 pp | 91.98 % | -2.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.60 % | 54.32 % | -25.28 pp | 54.57 % | -25.03 pp | 51.96 % | -27.64 pp |
| IPC Active | 0.81 inst/cycle | 1.83 inst/cycle | +1.02 | 1.81 inst/cycle | +1.00 | 1.89 inst/cycle | +1.08 |

---

## Iteration 3

**Best so far**: 1.9583× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_num_stages_and_num_warps_for_latency_hiding | 2.230× | 1.960× | ✓ |  |
| C | increase_num_stages_for_latency_hiding_via_software_pipelining | 2.556× | 1.949× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285760 ns | 145760 ns | -140000 ns (-49.0%) | 146620 ns | -139140 ns (-48.7%) |
| SM Throughput (%) | 19.63 % | 47.25 % | +27.62 pp | 47.09 % | +27.46 pp |
| Memory Throughput (%) | 78.85 % | 41.20 % | -37.65 pp | 40.89 % | -37.96 pp |
| DRAM Throughput (%) | 72.22 % | 33.44 % | -38.78 pp | 32.88 % | -39.34 pp |
| Achieved Occupancy (%) | 94.02 % | 91.99 % | -2.03 pp | 91.98 % | -2.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.60 % | 52.12 % | -27.48 pp | 55.05 % | -24.55 pp |
| IPC Active | 0.81 inst/cycle | 1.95 inst/cycle | +1.14 | 1.83 inst/cycle | +1.02 |

---

## Iteration 4

**Best so far**: 1.9583× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_on_stride1_source_dimension_y0 | 2.663× | 1.815× | ✓ | **✓ best** |
| D | fuse_elementwise_epilogue_into_layout_transpose_kernel | 2.558× | 1.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fuse_elementwise_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285760 ns | 157440 ns | -128320 ns (-44.9%) | 148100 ns | -137660 ns (-48.2%) |
| SM Throughput (%) | 19.63 % | 51.15 % | +31.52 pp | 47.16 % | +27.53 pp |
| Memory Throughput (%) | 78.85 % | 50.79 % | -28.06 pp | 40.92 % | -37.93 pp |
| DRAM Throughput (%) | 72.22 % | 30.63 % | -41.59 pp | 32.64 % | -39.58 pp |
| Achieved Occupancy (%) | 94.02 % | 92.96 % | -1.06 pp | 91.94 % | -2.08 pp |
| Registers/Thread | 25.00 register/thread | 27.00 register/thread | +2 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.60 % | 47.48 % | -32.12 pp | 54.97 % | -24.63 pp |
| IPC Active | 0.81 inst/cycle | 2.12 inst/cycle | +1.31 | 1.81 inst/cycle | +1.00 |

---

## Iteration 5

**Best so far**: 1.9583× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | smem_staged_coalesced_transpose_corrected_index_mapping | 2.058× | 1.923× | ✓ |  |
| D | correct_tl_trans_staged_coalesced_transpose | 1.896× | 1.980× | ✓ |  |
| E | add_triton_autotune | 2.666× | 1.800× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | smem_staged_coale... | Δ | correct_tl_trans_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285760 ns | 148610 ns | -137150 ns (-48.0%) | 144290 ns | -141470 ns (-49.5%) | 158780 ns | -126980 ns (-44.4%) |
| SM Throughput (%) | 19.63 % | 47.64 % | +28.01 pp | 47.06 % | +27.43 pp | 51.34 % | +31.71 pp |
| Memory Throughput (%) | 78.85 % | 41.54 % | -37.31 pp | 40.90 % | -37.95 pp | 50.97 % | -27.88 pp |
| DRAM Throughput (%) | 72.22 % | 32.84 % | -39.38 pp | 33.40 % | -38.82 pp | 30.48 % | -41.74 pp |
| Achieved Occupancy (%) | 94.02 % | 91.89 % | -2.13 pp | 91.97 % | -2.05 pp | 93.01 % | -1.01 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 27.00 register/thread | +2 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.60 % | 51.75 % | -27.85 pp | 53.95 % | -25.65 pp | 47.62 % | -31.98 pp |
| IPC Active | 0.81 inst/cycle | 1.90 inst/cycle | +1.09 | 1.86 inst/cycle | +1.05 | 2.12 inst/cycle | +1.31 |


### Key Learnings

#### ✅ Plan A: `smem_staged_coalesced_transpose_corrected_index_mapping` (1.923×)

For layout-transpose kernels where the load's stride-1 source dimension is along the slow (Y) tile axis but the warp-fast (X) dimension has a large stride (causing maximally uncoalesced loads), using tl.trans() to perform a register-level staged transpose can achieve correctness while reducing scheduler no-eligible cycles (54.97% → 51.75%) and warp cycles per issued instruction (32.73 → 30.59). The key correctness requirements are: (1) all XBLOCK and YBLOCK autotune values must be powers of 2 (required by tl.trans()), (2) the store address must be fully recomputed in transposed [XBLOCK, YBLOCK] shape using tl.trans(x2) and tl.trans(y3), and (3) the store mask must be recomputed directly from arange rather than transposing the original masks to avoid shape errors. Despite achieving correctness where prior attempts failed (max_diff ~8.5), the NCU metrics show only modest improvement because the load itself remains structurally uncoalesced — tl.trans() transposes the register tile after the uncoalesced load has already occurred, so DRAM throughput barely changes (32.64% → 32.84%).

> **Applicable when**: Layout-transpose kernel where loads are structurally uncoalesced (warp-fast X dimension has large stride in source tensor), stores are coalesced (stride-1 along X in destination), and prior single-pass approaches have failed. Use when correctness of index mapping is the primary concern and tl.trans() is available (Triton). All tile sizes must be powers of 2.

> **Avoid when**: When the goal is to coalesce the loads themselves — tl.trans() transposes the register tile AFTER the uncoalesced load, so DRAM throughput does not improve significantly (only ~0.2pp improvement observed). Do not apply when XBLOCK or YBLOCK cannot be constrained to powers of 2. Do not apply when the speedup target requires >2x improvement over baseline — this approach achieves correctness but does not eliminate the root-cause uncoalesced load pattern. Also do not apply when the kernel has non-power-of-2 tile dimensions in autotune configs, as tl.trans() will fail.

---

## Cumulative Progress

```
Baseline:  285760 ns  (1.000×)
Iter 1     145920 ns  (1.958×)  — autotune_tile_sizes_and_fix_eviction_policy

Final best: 158780 ns  (1.7997×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,65 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # num_stages=2 (existing baseline)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        # num_stages=3 — deeper pipeline, hides L2 latency
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        # num_stages=4 — maximum pipeline depth for high-latency DRAM
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+        # YBLOCK=128 configs — y0 = yindex % 128 spans [0..127] exactly per CTA, enabling LDG.128
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=3),
+        # Large XBLOCK configs — explore wider tiles for potential throughput gains
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    _yr = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), 1), YBLOCK)
+    yindex = yoffset + _yr[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    mask = ymask & xmask
     x2 = xindex
     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=mask, other=0.0, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +71,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
