# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 175580 ns — **Best latency**: 175140 ns (**1.0025×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_050/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 175580 ns
- **SM Throughput**: 28.00 % | **Memory**: 81.16 % | **DRAM**: 66.89 %
- **Achieved Occupancy**: 94.23 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 70.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_l2_reuse | 1.682× | 1.013× | ✓ |  |
| D | autotune_tile_sizes_and_warps | 3.541× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 175580 ns | 173250 ns | -2330 ns (-1.3%) | 175140 ns | -440 ns (-0.3%) |
| SM Throughput (%) | 28.00 % | 26.87 % | -1.13 pp | 26.89 % | -1.11 pp |
| Memory Throughput (%) | 81.16 % | 81.60 % | +0.44 pp | 80.62 % | -0.54 pp |
| DRAM Throughput (%) | 66.89 % | 67.77 % | +0.88 pp | 67.09 % | +0.20 pp |
| Achieved Occupancy (%) | 94.23 % | 94.78 % | +0.55 pp | 94.75 % | +0.52 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 70.88 % | 73.30 % | +2.42 pp | 73.39 % | +2.51 pp |
| IPC Active | 1.17 inst/cycle | 1.09 inst/cycle | -0.08 | 1.07 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0025× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_correct_1d_hints | 1.652× | 0.790× | ✓ |  |
| B | shared_memory_transpose_for_coalesced_load | 1.815× | 0.796× | ✓ |  |
| C | vectorized_stores_and_eviction_policy_fix | 3.530× | 0.807× | ✓ |  |
| D | eviction_policy_to_reduce_l2_saturation | 3.528× | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_tra... | Δ | vectorized_stores... | Δ | eviction_policy_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 175580 ns | 222140 ns | +46560 ns (+26.5%) | 220510 ns | +44930 ns (+25.6%) | 217540 ns | +41960 ns (+23.9%) | 182940 ns | +7360 ns (+4.2%) |
| SM Throughput (%) | 28.00 % | 20.24 % | -7.76 pp | 21.24 % | -6.76 pp | 20.33 % | -7.67 pp | 26.44 % | -1.56 pp |
| Memory Throughput (%) | 81.16 % | 66.17 % | -14.99 pp | 66.73 % | -14.43 pp | 67.61 % | -13.55 pp | 77.27 % | -3.89 pp |
| DRAM Throughput (%) | 66.89 % | 58.13 % | -8.76 pp | 58.37 % | -8.52 pp | 58.80 % | -8.09 pp | 63.91 % | -2.98 pp |
| Achieved Occupancy (%) | 94.23 % | 95.34 % | +1.11 pp | 95.30 % | +1.07 pp | 95.38 % | +1.15 pp | 94.97 % | +0.74 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 31.00 register/thread | -1 | 31.00 register/thread | -1 | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 70.88 % | 79.42 % | +8.54 pp | 79.98 % | +9.10 pp | 78.72 % | +7.84 pp | 72.50 % | +1.62 pp |
| IPC Active | 1.17 inst/cycle | 0.83 inst/cycle | -0.34 | 0.85 inst/cycle | -0.32 | 0.83 inst/cycle | -0.34 | 1.11 inst/cycle | -0.06 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_correct_1d_hints` (0.790×)

Applying tl.max_contiguous/tl.multiple_of to 1D arange vectors before broadcasting, combined with changing eviction_policy from evict_last to evict_first on a large non-reused tensor, caused a 21% slowdown (173 µs → 222 µs). Despite the hints being syntactically correct (applied to 1D tensors), warp cycles per issued instruction worsened from 56.5 to 76.1 cycles and eligible warps per scheduler dropped from 0.62 to 0.50, indicating increased memory latency stalls. The evict_first policy change likely reduced L2 hit rate (69.04% → 64.4%) by evicting data that was still being reused across tiles, and the added autotune configs with YBLOCK=64 may have selected a suboptimal tile shape that increased cross-tile channel splits rather than eliminating them.

> **Applicable when**: Never — this specific combination of 1D hints + evict_first on this kernel caused regression. Do not apply.

> **Avoid when**: When the kernel has a stride-96 uncoalesced load pattern (y0 + 96*x2 + 290400*y1) where the root cause is the non-unit stride in the X-dimension. Adding tl.max_contiguous/tl.multiple_of hints to 1D aranges does NOT fix the underlying stride-96 uncoalescing — it only provides alignment hints that the compiler cannot exploit when the access stride is inherently non-contiguous. Changing eviction_policy from evict_last to evict_first on a tensor with moderate L2 reuse (69% hit rate) reduces cache effectiveness and increases memory latency stalls. The combination worsened warp stall cycles by 35% (56.5 → 76.1 cycles/instruction).

---

## Iteration 3

**Best so far**: 1.0025× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staging_for_coalesced_store | 3.530× | 0.777× | ✓ |  |
| D | software_prefetch_depth_and_l2_compression_store | 1.627× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | software_prefetch... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 175580 ns | 225980 ns | +50400 ns (+28.7%) | 181630 ns | +6050 ns (+3.4%) |
| SM Throughput (%) | 28.00 % | 20.91 % | -7.09 pp | 26.09 % | -1.91 pp |
| Memory Throughput (%) | 81.16 % | 65.11 % | -16.05 pp | 77.72 % | -3.44 pp |
| DRAM Throughput (%) | 66.89 % | 57.51 % | -9.38 pp | 64.37 % | -2.52 pp |
| Achieved Occupancy (%) | 94.23 % | 95.38 % | +1.15 pp | 95.01 % | +0.78 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 70.88 % | 80.34 % | +9.46 pp | 73.94 % | +3.06 pp |
| IPC Active | 1.17 inst/cycle | 0.81 inst/cycle | -0.36 | 1.05 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  175580 ns  (1.000×)
Iter 1     175140 ns  (1.002×)  — autotune_tile_sizes_and_warps

Final best: 175140 ns  (1.0025×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,12 +35,10 @@

     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +47,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
