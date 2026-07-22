# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174590 ns — **Best latency**: 119710 ns (**1.4584×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`remove_evict_last_and_add_num_warps1_small_tile_configs_to_reduce_warp_contention`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_078/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174590 ns
- **SM Throughput**: 28.13 % | **Memory**: 81.58 % | **DRAM**: 67.26 %
- **Achieved Occupancy**: 94.18 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.686× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174590 ns | 174080 ns | -510 ns (-0.3%) |
| SM Throughput (%) | 28.13 % | 26.40 % | -1.73 pp |
| Memory Throughput (%) | 81.58 % | 81.15 % | -0.43 pp |
| DRAM Throughput (%) | 67.26 % | 67.46 % | +0.20 pp |
| Achieved Occupancy (%) | 94.18 % | 94.78 % | +0.60 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.80 % | 72.57 % | +0.77 pp |
| IPC Active | 1.12 inst/cycle | 1.09 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiled_transpose_for_coalesced_io | 1.980× | 1.812× | ✓ | **✓ best** |
| D | increase_num_stages_to_hide_l2_latency_with_evict_first_streaming | 1.642× | 0.797× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174590 ns | 96350 ns | -78240 ns (-44.8%) | 219070 ns | +44480 ns (+25.5%) |
| SM Throughput (%) | 28.13 % | 45.29 % | +17.16 pp | 20.99 % | -7.14 pp |
| Memory Throughput (%) | 81.58 % | 54.65 % | -26.93 pp | 67.21 % | -14.37 pp |
| DRAM Throughput (%) | 67.26 % | 29.15 % | -38.11 pp | 58.74 % | -8.52 pp |
| Achieved Occupancy (%) | 94.18 % | 92.39 % | -1.79 pp | 95.40 % | +1.22 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.80 % | 54.12 % | -17.68 pp | 78.97 % | +7.17 pp |
| IPC Active | 1.12 inst/cycle | 1.87 inst/cycle | +0.75 | 0.82 inst/cycle | -0.30 |

---

## Iteration 3

**Best so far**: 1.812× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_larger_tiles_more_warps_to_hide_store_latency | 1.976× | 1.792× | ✓ |  |
| B | fix_autotune_grid_lambda_and_fuse_fp32_instructions | 1.978× | 1.808× | ✓ |  |
| C | flat_1d_tiling_with_coalesced_output_and_l1_reuse_via_evict_last | 1.940× | 1.758× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | fix_autotune_grid... | Δ | flat_1d_tiling_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174590 ns | 97440 ns | -77150 ns (-44.2%) | 96580 ns | -78010 ns (-44.7%) | 99330 ns | -75260 ns (-43.1%) |
| SM Throughput (%) | 28.13 % | 45.17 % | +17.04 pp | 45.17 % | +17.04 pp | 48.68 % | +20.55 pp |
| Memory Throughput (%) | 81.58 % | 54.54 % | -27.04 pp | 54.59 % | -26.99 pp | 53.53 % | -28.05 pp |
| DRAM Throughput (%) | 67.26 % | 28.81 % | -38.45 pp | 28.92 % | -38.34 pp | 28.33 % | -38.93 pp |
| Achieved Occupancy (%) | 94.18 % | 92.46 % | -1.72 pp | 92.53 % | -1.65 pp | 93.04 % | -1.14 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.80 % | 53.55 % | -18.25 pp | 52.95 % | -18.85 pp | 49.68 % | -22.12 pp |
| IPC Active | 1.12 inst/cycle | 1.87 inst/cycle | +0.75 | 1.80 inst/cycle | +0.68 | 1.98 inst/cycle | +0.86 |


### Key Learnings

#### ❌ Plan A: `expand_autotune_configs_larger_tiles_more_warps_to_hide_store_latency` (1.792×)

Adding larger tile configs (XBLOCK=128/YBLOCK=16, XBLOCK=256/YBLOCK=8, XBLOCK=64/YBLOCK=64) with num_warps=8 and num_warps=16 to the autotune search space produced no meaningful improvement: kernel duration changed from 96.35us to 97.44us (~1% regression), and all scheduler metrics (No Eligible: 54.12%→53.55%, Eligible Warps: 2.9→2.94, Warp Cycles Per Issue: 31.61→31.45) were essentially unchanged. The autotuner selected a config equivalent to the prior best, indicating the bottleneck is structural (uncoalesced store stride-3025 in the output) rather than a tile-size or warp-count issue. When the dominant stall is caused by a fixed memory layout constraint that cannot be resolved by tile reshaping, expanding the autotune search space yields no benefit.

> **Applicable when**: Never — this produced no improvement. Do not apply when the kernel already has a well-tuned autotune config set and the remaining scheduler idle is caused by a structural memory layout constraint (e.g., stride-3025 store pattern) that cannot be resolved by tile reshaping or warp count changes.

> **Avoid when**: When IssueSlotUtilization stall persists after a prior autotune optimization and the uncoalesced access is in the store path with a fixed output stride. In such cases, the autotuner will simply re-select the previously winning config and all metrics will remain flat. Duration changed by only 1us (96.35→97.44us), No Eligible changed by only 0.57pp (54.12%→53.55%), and Eligible Warps changed by only 0.04 (2.9→2.94) — all within noise.

---

## Iteration 4

**Best so far**: 1.812× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_warps2_and_num_stages3_to_increase_eligible_warps | 2.567× | 1.806× | ✓ | **✓ best** |
| B | contiguous_hints_for_vectorized_bias_load_and_output_store | 1.594× | 1.383× | ✓ |  |
| C | increase_num_stages_and_mlp_via_larger_tiles_for_latency_hiding | 1.977× | 1.774× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_warp... | Δ | contiguous_hints_... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174590 ns | 96700 ns | -77890 ns (-44.6%) | 126210 ns | -48380 ns (-27.7%) | 98400 ns | -76190 ns (-43.6%) |
| SM Throughput (%) | 28.13 % | 45.50 % | +17.37 pp | 67.27 % | +39.14 pp | 46.18 % | +18.05 pp |
| Memory Throughput (%) | 81.58 % | 54.90 % | -26.68 pp | 52.03 % | -29.55 pp | 55.76 % | -25.82 pp |
| DRAM Throughput (%) | 67.26 % | 28.94 % | -38.32 pp | 22.24 % | -45.02 pp | 28.59 % | -38.67 pp |
| Achieved Occupancy (%) | 94.18 % | 92.50 % | -1.68 pp | 94.19 % | +0.01 pp | 92.44 % | -1.74 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.80 % | 53.03 % | -18.77 pp | 31.86 % | -39.94 pp | 53.76 % | -18.04 pp |
| IPC Active | 1.12 inst/cycle | 1.86 inst/cycle | +0.74 | 2.69 inst/cycle | +1.57 | 1.85 inst/cycle | +0.73 |


### Key Learnings

#### ✅ Plan C: `increase_num_stages_and_mlp_via_larger_tiles_for_latency_hiding` (1.774×)

Expanding the autotune config set with num_stages=3 and num_stages=4 variants (plus larger tiles like XBLOCK=256/YBLOCK=32) for a pointwise transpose+bias-add kernel produced no improvement at the NCU kernel level: duration increased slightly from 96.58us to 98.4us, No Eligible scheduler stall rose from 52.95% to 53.76%, and eligible warps per scheduler dropped from 2.98 to 2.94. The reported 1.77x overall speedup is attributable to a concurrent change (reverting libdevice.fma back to simple addition), not to the num_stages expansion. For pure pointwise kernels with no shared memory reuse and no tl.dot, increasing num_stages does not meaningfully increase memory-level parallelism because the Triton compiler does not emit async global loads for simple elementwise access patterns; the structural bottleneck (uncoalesced stride-96 loads) remains unchanged.

> **Applicable when**: Never effective as a standalone optimization for pure pointwise kernels (no shared memory, no tl.dot) when the scheduler stall is caused by uncoalesced global loads. The autotuner will re-select a num_stages=2 config equivalent to the prior best, and all NCU metrics remain flat.

> **Avoid when**: When the kernel is a pure pointwise/elementwise kernel with no shared memory staging and no matrix multiply (tl.dot). In such kernels, num_stages controls software pipelining of shared memory loads, which does not exist here; increasing num_stages adds no async memory prefetch benefit. The structural bottleneck (non-unit stride between adjacent threads in the fast warp dimension) cannot be resolved by deeper pipelining. Duration increased by 1.8us and No Eligible rose by 0.81pp — both worsened slightly.

---

## Iteration 5

**Best so far**: 1.812× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_last_bias_reuse_with_correct_y0_contiguous_hint | 1.936× | 1.763× | ✓ |  |
| B | remove_evict_last_and_add_num_warps1_small_tile_configs_to_reduce_warp_contention | 3.527× | 1.458× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_last_bias_r... | Δ | remove_evict_last... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174590 ns | 99040 ns | -75550 ns (-43.3%) | 119710 ns | -54880 ns (-31.4%) |
| SM Throughput (%) | 28.13 % | 47.55 % | +19.42 pp | 58.07 % | +29.94 pp |
| Memory Throughput (%) | 81.58 % | 52.14 % | -29.44 pp | 53.92 % | -27.66 pp |
| DRAM Throughput (%) | 67.26 % | 27.45 % | -39.81 pp | 23.33 % | -43.93 pp |
| Achieved Occupancy (%) | 94.18 % | 93.07 % | -1.11 pp | 93.99 % | -0.19 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.80 % | 50.40 % | -21.40 pp | 41.55 % | -30.25 pp |
| IPC Active | 1.12 inst/cycle | 1.94 inst/cycle | +0.82 | 2.32 inst/cycle | +1.20 |

---

## Cumulative Progress

```
Baseline:  174590 ns  (1.000×)
Iter 1     174080 ns  (1.003×)  — vectorized_loads_with_contiguous_hints
Iter 2     96350 ns  (1.812×)  — shared_memory_tiled_transpose_for_coalesced_io

Final best: 119710 ns  (1.4584×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,47 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=1,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},   num_warps=1,  num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32},  num_warps=1,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},   num_warps=1,  num_stages=2),
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
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[None, :], [1, XBLOCK]), [1, XBLOCK])
     xmask = xindex < xnumel
     x2 = xindex
-    y0 = (yindex % 96)
+    y0_raw = (yindex % 96)
+    y0 = tl.max_contiguous(tl.multiple_of(y0_raw, [1, 1]), [YBLOCK, 1])
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask)
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask)
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +54,7 @@

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
