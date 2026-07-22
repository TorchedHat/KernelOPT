# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22660 ns — **Best latency**: 22620 ns (**1.0018×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_persistent_loop_to_flat_launch_eliminate_instruction_explosion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_2_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22660 ns
- **SM Throughput**: 36.14 % | **Memory**: 18.77 % | **DRAM**: 15.99 %
- **Achieved Occupancy**: 39.26 %
- **IPC Active**: 1.61 inst/cycle | **No Eligible**: 59.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 0.640× | 0.642× | ✓ |  |
| B | increase_xblock_tile_size_for_latency_hiding | 0.841× | 0.857× | ✓ |  |
| C | autotune_xblock_increase_memory_level_parallelism | 1.001× | 0.439× | ✓ | **✓ best** |
| D | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.979× | 0.232× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | increase_xblock_t... | Δ | autotune_xblock_i... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22660 ns | 35300 ns | +12640 ns (+55.8%) | 26430 ns | +3770 ns (+16.6%) | 51580 ns | +28920 ns (+127.6%) | 97760 ns | +75100 ns (+331.4%) |
| SM Throughput (%) | 36.14 % | 87.51 % | +51.37 pp | 84.42 % | +48.28 pp | 85.59 % | +49.45 pp | 88.61 % | +52.47 pp |
| Memory Throughput (%) | 18.77 % | 43.95 % | +25.18 pp | 37.00 % | +18.23 pp | 30.10 % | +11.33 pp | 28.14 % | +9.37 pp |
| DRAM Throughput (%) | 15.99 % | 10.21 % | -5.78 pp | 13.75 % | -2.24 pp | 6.95 % | -9.04 pp | 3.67 % | -12.32 pp |
| Achieved Occupancy (%) | 39.26 % | 88.99 % | +49.73 pp | 82.62 % | +43.36 pp | 86.04 % | +46.78 pp | 86.58 % | +47.32 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 26.00 register/thread | +10 | 25.00 register/thread | +9 | 26.00 register/thread | +10 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 31.03 | +15.51 | 62.06 | +46.54 |
| No Eligible (%) | 59.32 % | 7.14 % | -52.18 pp | 8.52 % | -50.80 pp | 10.13 % | -49.19 pp | 9.47 % | -49.85 pp |
| IPC Active | 1.61 inst/cycle | 3.70 inst/cycle | +2.09 | 3.65 inst/cycle | +2.04 | 3.59 inst/cycle | +1.98 | 3.63 inst/cycle | +2.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | simplify_index_arithmetic_to_eliminate_alu_instruction_explosion | 1.006× | 0.995× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.004× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | simplify_index_ar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 22660 ns | 22780 ns | +120 ns (+0.5%) | 22660 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 36.14 % | 39.85 % | +3.71 pp | 38.34 % | +2.20 pp |
| Memory Throughput (%) | 18.77 % | 18.73 % | -0.04 pp | 18.78 % | +0.01 pp |
| DRAM Throughput (%) | 15.99 % | 15.76 % | -0.23 pp | 15.90 % | -0.09 pp |
| Achieved Occupancy (%) | 39.26 % | 40.96 % | +1.70 pp | 40.21 % | +0.95 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 59.32 % | 55.23 % | -4.09 pp | 56.89 % | -2.43 pp |
| IPC Active | 1.61 inst/cycle | 1.78 inst/cycle | +0.17 | 1.71 inst/cycle | +0.10 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.005× | 1.002× | ✓ |  |
| C | persistent_tile_scheduling | 0.116× | 0.102× | ✓ |  |
| D | fuse_fp32_add_mul_to_fma_and_increase_num_warps | 1.006× | 1.002× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fuse_fp32_add_mul... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22660 ns | 22620 ns | -40 ns (-0.2%) | 221020 ns | +198360 ns (+875.4%) | 22620 ns | -40 ns (-0.2%) | 22660 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 36.14 % | 48.13 % | +11.99 pp | 93.23 % | +57.09 pp | 40.52 % | +4.38 pp | 38.33 % | +2.19 pp |
| Memory Throughput (%) | 18.77 % | 18.85 % | +0.08 pp | 58.40 % | +39.63 pp | 18.84 % | +0.07 pp | 18.77 % | +0.00 pp |
| DRAM Throughput (%) | 15.99 % | 15.90 % | -0.09 pp | 2.80 % | -13.19 pp | 15.88 % | -0.11 pp | 16.00 % | +0.01 pp |
| Achieved Occupancy (%) | 39.26 % | 44.61 % | +5.35 pp | 96.43 % | +57.17 pp | 41.08 % | +1.82 pp | 40.08 % | +0.82 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 59.32 % | 46.29 % | -13.03 pp | 5.45 % | -53.87 pp | 54.33 % | -4.99 pp | 56.94 % | -2.38 pp |
| IPC Active | 1.61 inst/cycle | 2.14 inst/cycle | +0.53 | 3.78 inst/cycle | +2.17 | 1.80 inst/cycle | +0.19 | 1.70 inst/cycle | +0.09 |

---

## Iteration 4

**Best so far**: 1.0018× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_add_mul_to_fma_instruction | 1.002× | 0.999× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding | 0.117× | 0.104× | ✓ |  |
| C | increase_num_warps_for_l1tex_stall_latency_hiding | 0.781× | 0.818× | ✓ |  |
| D | fuse_fp32_add_mul_to_fma_instruction_standalone | 0.118× | 0.105× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_add_mul... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | fuse_fp32_add_mul... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22660 ns | 22690 ns | +30 ns (+0.1%) | 218080 ns | +195420 ns (+862.4%) | 27710 ns | +5050 ns (+22.3%) | 214880 ns | +192220 ns (+848.3%) |
| SM Throughput (%) | 36.14 % | 39.09 % | +2.95 pp | 93.49 % | +57.35 pp | 65.11 % | +28.97 pp | 94.99 % | +58.85 pp |
| Memory Throughput (%) | 18.77 % | 18.78 % | +0.01 pp | 59.23 % | +40.46 pp | 18.78 % | +0.01 pp | 60.04 % | +41.27 pp |
| DRAM Throughput (%) | 15.99 % | 15.85 % | -0.14 pp | 2.89 % | -13.10 pp | 13.05 % | -2.94 pp | 2.91 % | -13.08 pp |
| Achieved Occupancy (%) | 39.26 % | 40.09 % | +0.83 pp | 96.39 % | +57.13 pp | 71.12 % | +31.86 pp | 97.11 % | +57.85 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 31.03 | +15.51 | 15.52 | +0.00 |
| No Eligible (%) | 59.32 % | 56.30 % | -3.02 pp | 5.65 % | -53.67 pp | 29.26 % | -30.06 pp | 3.75 % | -55.57 pp |
| IPC Active | 1.61 inst/cycle | 1.75 inst/cycle | +0.14 | 3.77 inst/cycle | +2.16 | 2.83 inst/cycle | +1.22 | 3.85 inst/cycle | +2.24 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_for_l1tex_stall_latency_hiding` (0.818×)

Increasing num_warps from 4 to 8 (block size 128→256) improved achieved occupancy (44.6%→71.1%) and reduced no-eligible cycles (46.3%→29.3%), but caused a 22% slowdown. The autotune selected block_size=256, which doubled the total thread count (4.2M→8.4M) and executed instructions (8.5M→14.0M), while warp cycles per issued instruction actually increased (13.0→15.7 cycles). The L1TEX stall remained dominant (60.6% of stall budget) and the extra warps did not hide it effectively — instead, the doubled instruction count from processing twice as many elements per block dominated the runtime. When a kernel's bottleneck is inherent non-coalesced memory access (not simply insufficient warp count), adding more warps via larger block size amplifies total work without proportionally reducing per-element latency.

> **Applicable when**: Kernel is purely latency-bound with coalesced memory access and the L1TEX stall is caused by insufficient warp count (very few active warps per scheduler, e.g. <4), not by non-coalesced access patterns. Effective when increasing num_warps does NOT increase total instruction count (i.e., XBLOCK stays fixed and block size increase is purely from num_warps, not from processing more elements).

> **Avoid when**: Kernel has non-coalesced global memory access patterns (complex modular index arithmetic causing strided loads). In such cases, increasing num_warps via larger block size doubles total executed instructions (because autotune selects larger block_size to match num_warps×warp_size), and the L1TEX stall persists because the root cause is access pattern inefficiency, not warp count. Also avoid when waves_per_sm is already high (>15) — the kernel already saturates the GPU with work and adding more warps per block doubles the wave count without reducing per-wave latency.

#### ❌ Plan D: `fuse_fp32_add_mul_to_fma_instruction_standalone` (0.105×)

The diff reveals that alongside the FMA rewrite, the persistent while-loop's tile increment was changed from `tile_id += tl.num_programs(0)` (dynamic, equals actual grid size) to `tile_id += GRID_SIZE` where GRID_SIZE was hardcoded to 528 — far smaller than the actual grid of 32768. This caused each CTA to process ~62x more tiles than intended (32768/528 ≈ 62), exploding executed instructions from 8.5M to 159.8M (18.8x increase) and duration from 22.6µs to 214.9µs. The L2 hit rate jumped from 51.8% to 96.8% because the same data was re-read many times from cache, and occupancy rose to 97% because CTAs ran far longer. The FMA rewrite itself is not the cause — the catastrophic regression is entirely due to the incorrect hardcoded GRID_SIZE replacing `tl.num_programs(0)`.

> **Applicable when**: Never: hardcoding GRID_SIZE to a value smaller than the actual launch grid in a persistent kernel loop is always incorrect and catastrophically slow.

> **Avoid when**: Always avoid replacing `tl.num_programs(0)` with a hardcoded constant in persistent kernel tile loops. The actual grid size at launch must match the stride used in the persistent loop; any mismatch causes each CTA to process far more tiles than intended, multiplying instruction count and duration proportionally to the ratio (actual_grid / hardcoded_stride).

---

## Iteration 5

**Best so far**: 1.0018× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | revert_persistent_loop_to_flat_one_tile_per_cta_launch | 0.994× | 1.000× | ✓ |  |
| D | revert_persistent_loop_to_flat_launch_eliminate_instruction_explosion | 1.006× | 1.002× | ✓ |  |
| E | add_triton_autotune | 1.013× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | revert_persistent... | Δ | revert_persistent... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22660 ns | 22660 ns | +0 ns (+0.0%) | 22620 ns | -40 ns (-0.2%) | 22620 ns | -40 ns (-0.2%) |
| SM Throughput (%) | 36.14 % | 40.69 % | +4.55 pp | 40.61 % | +4.47 pp | 38.01 % | +1.87 pp |
| Memory Throughput (%) | 18.77 % | 18.77 % | +0.00 pp | 18.88 % | +0.11 pp | 18.79 % | +0.02 pp |
| DRAM Throughput (%) | 15.99 % | 15.85 % | -0.14 pp | 16.00 % | +0.01 pp | 16.01 % | +0.02 pp |
| Achieved Occupancy (%) | 39.26 % | 41.12 % | +1.86 pp | 41.23 % | +1.97 pp | 40.47 % | +1.21 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 59.32 % | 54.31 % | -5.01 pp | 54.64 % | -4.68 pp | 56.90 % | -2.42 pp |
| IPC Active | 1.61 inst/cycle | 1.81 inst/cycle | +0.20 | 1.82 inst/cycle | +0.21 | 1.69 inst/cycle | +0.08 |

---

## Cumulative Progress

```
Baseline:  22660 ns  (1.000×)
Iter 3     22620 ns  (1.002×)  — fuse_fp32_add_mul_to_fma_and_increase_num_warps

Final best: 22620 ns  (1.0018×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4194304
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
     x1 = ((xindex // 32) % 16384)
     x2 = xindex // 524288
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (384*((x0 + 32*x2) // 128) + 768*x1 + 768*((x0 + 32*x2) // 256) + (((x0 + 32*x2) % 128))), None)
-    tmp1 = tl.load(in_ptr1 + (((x0 + 32*x2) % 128)), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (384*((x0 + 32*x2) // 128) + 768*x1 + 768*((x0 + 32*x2) // 256) + (((x0 + 32*x2) % 128))), xmask)
+    tmp1 = tl.load(in_ptr1 + (((x0 + 32*x2) % 128)), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
     tmp4 = tmp2 * tmp3
-    tl.store(out_ptr0 + (x3), tmp4, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp4, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     arg_2 = inputs[2]
     xnumel = 4194304
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304, XBLOCK=128)
+    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
