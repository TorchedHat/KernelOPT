# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 284540 ns — **Best latency**: 168580 ns (**1.6879×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_stores_with_contiguity_hints_and_eviction_policy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_062/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 284540 ns
- **SM Throughput**: 20.04 % | **Memory**: 79.23 % | **DRAM**: 72.52 %
- **Achieved Occupancy**: 94.02 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.779× | 1.562× | ✓ |  |
| B | autotune_tile_sizes_for_memory_coalescing | 1.930× | 0.999× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.000× | 0.998× | ✓ |  |
| D | autotune_tile_sizes_for_l2_bound_transpose | 2.014× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284540 ns | 182180 ns | -102360 ns (-36.0%) | 284930 ns | +390 ns (+0.1%) | 285150 ns | +610 ns (+0.2%) | 284740 ns | +200 ns (+0.1%) |
| SM Throughput (%) | 20.04 % | 29.81 % | +9.77 pp | 19.87 % | -0.17 pp | 20.04 % | +0.00 pp | 19.70 % | -0.34 pp |
| Memory Throughput (%) | 79.23 % | 90.54 % | +11.31 pp | 79.11 % | -0.12 pp | 79.02 % | -0.21 pp | 79.18 % | -0.05 pp |
| DRAM Throughput (%) | 72.52 % | 55.84 % | -16.68 pp | 72.41 % | -0.11 pp | 72.36 % | -0.16 pp | 72.48 % | -0.04 pp |
| Achieved Occupancy (%) | 94.02 % | 92.72 % | -1.30 pp | 94.02 % | +0.00 pp | 94.00 % | -0.02 pp | 94.04 % | +0.02 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 69.95 % | -10.25 pp | 79.52 % | -0.68 pp | 80.12 % | -0.08 pp | 79.74 % | -0.46 pp |
| IPC Active | 0.79 inst/cycle | 1.21 inst/cycle | +0.42 | 0.82 inst/cycle | +0.03 | 0.82 inst/cycle | +0.03 | 0.81 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_and_load_eviction_for_strided_transpose | 1.478× | 1.122× | ✓ |  |
| B | vectorized_stores_with_contiguity_hints_on_store_index | 2.010× | 0.999× | ✓ |  |
| C | vectorized_stores_with_contiguity_hints_and_eviction_policy | 2.227× | 1.688× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | vectorized_stores... | Δ | vectorized_stores... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284540 ns | 253540 ns | -31000 ns (-10.9%) | 284770 ns | +230 ns (+0.1%) | 168580 ns | -115960 ns (-40.8%) |
| SM Throughput (%) | 20.04 % | 20.77 % | +0.73 pp | 19.62 % | -0.42 pp | 45.96 % | +25.92 pp |
| Memory Throughput (%) | 79.23 % | 74.35 % | -4.88 pp | 79.22 % | -0.01 pp | 91.95 % | +12.72 pp |
| DRAM Throughput (%) | 72.52 % | 52.85 % | -19.67 pp | 72.47 % | -0.05 pp | 13.66 % | -58.86 pp |
| Achieved Occupancy (%) | 94.02 % | 94.24 % | +0.22 pp | 93.98 % | -0.04 pp | 93.48 % | -0.54 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 79.00 % | -1.20 pp | 79.73 % | -0.47 pp | 52.96 % | -27.24 pp |
| IPC Active | 0.79 inst/cycle | 0.85 inst/cycle | +0.06 | 0.81 inst/cycle | +0.02 | 1.88 inst/cycle | +1.09 |

---

## Iteration 3

**Best so far**: 1.6879× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiling_for_coalesced_transpose_loads | 2.200× | 1.686× | ✓ |  |
| B | store_evict_first_to_free_l2_for_read_traffic | 1.604× | 1.419× | ✓ |  |
| C | evict_first_store_to_free_l2_for_strided_reads | 2.225× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | store_evict_first... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284540 ns | 168800 ns | -115740 ns (-40.7%) | 200450 ns | -84090 ns (-29.6%) | 285280 ns | +740 ns (+0.3%) |
| SM Throughput (%) | 20.04 % | 45.93 % | +25.89 pp | 28.56 % | +8.52 pp | 20.53 % | +0.49 pp |
| Memory Throughput (%) | 79.23 % | 91.87 % | +12.64 pp | 82.79 % | +3.56 pp | 79.37 % | +0.14 pp |
| DRAM Throughput (%) | 72.52 % | 13.63 % | -58.89 pp | 50.52 % | -22.00 pp | 72.13 % | -0.39 pp |
| Achieved Occupancy (%) | 94.02 % | 93.51 % | -0.51 pp | 93.04 % | -0.98 pp | 93.37 % | -0.65 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 23.00 register/thread | -2 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 52.96 % | -27.24 pp | 70.97 % | -9.23 pp | 79.41 % | -0.79 pp |
| IPC Active | 0.79 inst/cycle | 1.88 inst/cycle | +1.09 | 1.16 inst/cycle | +0.37 | 0.84 inst/cycle | +0.05 |

---

## Iteration 4

**Best so far**: 1.6879× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_yblock1_wide_xblock_configs_for_strided_transpose | 1.965× | 1.419× | ✓ |  |
| C | evict_first_on_strided_load_to_free_l2_for_contiguous_write_traffic | 2.055× | 0.875× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_yblock1_... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 284540 ns | 200510 ns | -84030 ns (-29.5%) | 324990 ns | +40450 ns (+14.2%) |
| SM Throughput (%) | 20.04 % | 28.84 % | +8.80 pp | 20.09 % | +0.05 pp |
| Memory Throughput (%) | 79.23 % | 82.80 % | +3.57 pp | 69.45 % | -9.78 pp |
| DRAM Throughput (%) | 72.52 % | 50.48 % | -22.04 pp | 63.43 % | -9.09 pp |
| Achieved Occupancy (%) | 94.02 % | 93.04 % | -0.98 pp | 93.52 % | -0.50 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 70.91 % | -9.29 pp | 79.26 % | -0.94 pp |
| IPC Active | 0.79 inst/cycle | 1.17 inst/cycle | +0.38 | 0.83 inst/cycle | +0.04 |


### Key Learnings

#### ✅ Plan B: `autotune_yblock1_wide_xblock_configs_for_strided_transpose` (1.419×)

Adding YBLOCK=1 wide-XBLOCK autotune configs (XBLOCK=512/1024/2048) to a Triton layout-permutation kernel that already had YBLOCK=1 configs in its autotune list caused a significant regression (1.42x slower): L2 throughput jumped from 68% to 91%, DRAM throughput from 14% to 50%, warp cycles per issued instruction from 31.7 to 50.5, and No Eligible warp % from 53% to 71%. The diff reveals the 'before' kernel already contained YBLOCK=1 configs (including XBLOCK=1024, YBLOCK=1, num_warps=4, num_stages=2) and also had evict_first on the load; the 'after' kernel replaced the existing config set with a different set that removed some previously winning configs and simultaneously changed the load eviction policy from 'evict_last' to 'evict_first'. The regression is attributable to the load eviction policy change (evict_last→evict_first on strided loads) combined with the replacement of the previously-tuned winning config set, which had already achieved a good L2 hit rate (87.7%) and low DRAM traffic (13.6%). Switching strided loads to evict_first when L2 hit rate is already high (>85%) and DRAM throughput is already low (<15%) destroys L2 reuse and forces more DRAM traffic.

> **Applicable when**: Never: this is a negative result. The strategy of adding YBLOCK=1 wide-XBLOCK configs when they already exist, combined with changing load eviction policy from evict_last to evict_first when L2 hit rate is already >85% and DRAM throughput is already <15%, causes regression.

> **Avoid when**: Do not change load eviction policy from evict_last to evict_first when the kernel already achieves high L2 hit rate (>85%) and low DRAM throughput (<20%). Do not replace an existing autotune config set that already contains YBLOCK=1 configs and is performing well — the autotuner has already found the optimal config. Do not add YBLOCK=1 configs when they are already present in the autotune list.

---

## Cumulative Progress

```
Baseline:  284540 ns  (1.000×)
Iter 2     168580 ns  (1.688×)  — vectorized_stores_with_contiguity_hints_and_eviction_policy

Final best: 168580 ns  (1.6879×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,46 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=16, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    x2 = xindex
+    x2 = tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])
     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +52,7 @@

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
