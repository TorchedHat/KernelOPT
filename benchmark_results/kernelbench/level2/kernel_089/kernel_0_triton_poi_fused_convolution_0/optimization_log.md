# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 3170 ns (**1.8864×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_089/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.57 % | **Memory**: 41.08 % | **DRAM**: 20.68 %
- **Achieved Occupancy**: 55.81 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 71.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.274× | 0.979× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.328× | 1.886× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_swap | 1.396× | 1.005× | ✓ | **✓ best** |
| D | add_autotune_increase_tile_sizes | 1.208× | 1.473× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 6110 ns | +130 ns (+2.2%) | 3170 ns | -2810 ns (-47.0%) | 5950 ns | -30 ns (-0.5%) | 4060 ns | -1920 ns (-32.1%) |
| SM Throughput (%) | 16.57 % | 16.61 % | +0.04 pp | 5.23 % | -11.34 pp | 16.13 % | -0.44 pp | 12.16 % | -4.41 pp |
| Memory Throughput (%) | 41.08 % | 40.77 % | -0.31 pp | 11.43 % | -29.65 pp | 42.19 % | +1.11 pp | 34.25 % | -6.83 pp |
| DRAM Throughput (%) | 20.68 % | 20.26 % | -0.42 pp | 2.63 % | -18.05 pp | 20.73 % | +0.05 pp | 16.34 % | -4.34 pp |
| Achieved Occupancy (%) | 55.81 % | 55.31 % | -0.50 pp | 25.77 % | -30.04 pp | 56.04 % | +0.23 pp | 28.10 % | -27.71 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.59 % | 71.08 % | -0.51 pp | 80.35 % | +8.76 pp | 70.36 % | -1.23 pp | 74.76 % | +3.17 pp |
| IPC Active | 1.08 inst/cycle | 1.15 inst/cycle | +0.07 | 0.66 inst/cycle | -0.42 | 1.13 inst/cycle | +0.05 | 1.03 inst/cycle | -0.05 |

---

## Iteration 2

**Best so far**: 1.005× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.419× | 1.834× | ✓ | **✓ best** |
| B | vectorized_loads_with_max_contiguous_multiple_of_hints | 1.248× | 1.427× | ✓ |  |
| C | vectorized_loads_with_yblock1_and_contiguous_hints | 1.386× | 1.869× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3260 ns | -2720 ns (-45.5%) | 4190 ns | -1790 ns (-29.9%) | 3200 ns | -2780 ns (-46.5%) |
| SM Throughput (%) | 16.57 % | 5.05 % | -11.52 pp | 12.22 % | -4.35 pp | 5.07 % | -11.50 pp |
| Memory Throughput (%) | 41.08 % | 10.99 % | -30.09 pp | 33.24 % | -7.84 pp | 11.37 % | -29.71 pp |
| DRAM Throughput (%) | 20.68 % | 2.55 % | -18.13 pp | 15.65 % | -5.03 pp | 2.61 % | -18.07 pp |
| Achieved Occupancy (%) | 55.81 % | 26.89 % | -28.92 pp | 28.29 % | -27.52 pp | 27.57 % | -28.24 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 71.59 % | 80.58 % | +8.99 pp | 73.49 % | +1.90 pp | 80.22 % | +8.63 pp |
| IPC Active | 1.08 inst/cycle | 0.60 inst/cycle | -0.48 | 0.98 inst/cycle | -0.10 | 0.69 inst/cycle | -0.39 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_max_contiguous_multiple_of_hints` (1.834×)

Adding tl.max_contiguous and tl.multiple_of hints to the x-index of a layout permutation kernel (NCHW→NHWC) to enable 128-bit vectorized loads produced no improvement and a slight regression (3.17 µs → 3.26 µs, 1.83x reported speedup is misleading — the NCU duration increased). The profiling metrics are nearly identical before and after: warp cycles per issued instruction increased from 21.0 to 24.1, memory throughput dropped from 127.6 to 123.9 GB/s, and all scheduler/occupancy metrics remained flat. The dominant bottleneck is stride-3 uncoalesced stores (50% excess sectors, 10.7/32 bytes utilized per sector on stores), which these load-side hints cannot address; the load side was already being served efficiently from L2 cache (78% hit rate), so vectorizing loads provided no measurable benefit.

> **Applicable when**: Never beneficial in this configuration: load-side vectorization hints are ineffective when the dominant bottleneck is uncoalesced stores (stride > 1) and the load side already has high L2 hit rate (>70%), meaning load latency is not the binding constraint.

> **Avoid when**: When the kernel has strided stores as the primary source of excess sectors (UncoalescedGlobalAccess on stores, MemoryCacheAccessPattern flagging stores) and L2 hit rate on loads is already high (>70%). In this regime, tl.max_contiguous/tl.multiple_of on the load index provides no measurable benefit and may slightly increase warp stall cycles.

#### ✅ Plan B: `vectorized_loads_with_max_contiguous_multiple_of_hints` (1.427×)

Adding tl.max_contiguous and tl.multiple_of hints to a stride-1 index range in Triton can enable 128-bit vectorized loads, but the register cost of carrying the contiguity metadata (16→26 registers/thread) and the introduction of dynamic shared memory (0→2KB/block) can increase elapsed cycles and duration even when per-transaction efficiency improves. In this case, memory throughput jumped from 127.6 to 767.15 GB/s and L1 hit rate rose from 0% to 27.6%, confirming vectorization occurred; however, the higher register pressure reduced the effective tile size and increased total elapsed cycles from 4693 to 6255, resulting in a net regression relative to the prior autotuned baseline. The strategy is only beneficial when register pressure is already low and the load path is the dominant bottleneck, not when the kernel is already register-constrained.

> **Applicable when**: Triton kernel has a stride-1 load path that is the dominant bottleneck (L1TEX hit rate near 0%, CPIStall L1TEX stalls >50% of warp cycles), registers/thread is already low (<20), and no dynamic shared memory is allocated. XBLOCK must be a power-of-2 ≥128 to satisfy 128-bit alignment.

> **Avoid when**: Kernel is already register-constrained (registers/thread ≥20); adding hints increases register count and introduces dynamic shared memory overhead that outweighs vectorization gains. Also avoid when the load address has non-unit stride (hints would be semantically incorrect) or when the kernel has already been autotuned to a configuration where elapsed cycles are near-minimal.

---

## Iteration 3

**Best so far**: 1.8344× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_buffer_for_coalesced_stores | 1.433× | 1.886× | ✓ | **✓ best** |
| B | fix_hardcoded_grid_bypass_autotune_and_eviction_policy | 1.288× | 1.886× | ✓ |  |
| C | expand_autotune_configs_with_yblock1_for_strided_store_coalescing | 1.420× | 1.869× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | fix_hardcoded_gri... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3170 ns | -2810 ns (-47.0%) | 3170 ns | -2810 ns (-47.0%) | 3200 ns | -2780 ns (-46.5%) |
| SM Throughput (%) | 16.57 % | 5.10 % | -11.47 pp | 5.15 % | -11.42 pp | 5.21 % | -11.36 pp |
| Memory Throughput (%) | 41.08 % | 11.26 % | -29.82 pp | 11.25 % | -29.83 pp | 11.33 % | -29.75 pp |
| DRAM Throughput (%) | 20.68 % | 2.63 % | -18.05 pp | 2.63 % | -18.05 pp | 2.61 % | -18.07 pp |
| Achieved Occupancy (%) | 55.81 % | 25.52 % | -30.29 pp | 25.87 % | -29.94 pp | 27.30 % | -28.51 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 71.59 % | 79.82 % | +8.23 pp | 80.26 % | +8.67 pp | 82.81 % | +11.22 pp |
| IPC Active | 1.08 inst/cycle | 0.63 inst/cycle | -0.45 | 0.63 inst/cycle | -0.45 | 0.72 inst/cycle | -0.36 |

---

## Iteration 4

**Best so far**: 1.8864× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_site_eviction_policy_for_streaming_permutation | 1.425× | 1.886× | ✓ |  |
| B | fix_grid_lambda_and_large_xblock_yblock1_configs | 1.381× | 1.869× | ✓ |  |
| C | fix_hardcoded_launch_grid_to_enable_autotune_yblock1_selection | 1.419× | 1.886× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_site_e... | Δ | fix_grid_lambda_a... | Δ | fix_hardcoded_lau... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3170 ns | -2810 ns (-47.0%) | 3200 ns | -2780 ns (-46.5%) | 3170 ns | -2810 ns (-47.0%) |
| SM Throughput (%) | 16.57 % | 4.92 % | -11.65 pp | 5.16 % | -11.41 pp | 5.21 % | -11.36 pp |
| Memory Throughput (%) | 41.08 % | 11.43 % | -29.65 pp | 11.34 % | -29.74 pp | 11.47 % | -29.61 pp |
| DRAM Throughput (%) | 20.68 % | 2.63 % | -18.05 pp | 2.62 % | -18.06 pp | 2.64 % | -18.04 pp |
| Achieved Occupancy (%) | 55.81 % | 27.19 % | -28.62 pp | 28.13 % | -27.68 pp | 24.49 % | -31.32 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 71.59 % | 80.10 % | +8.51 pp | 79.79 % | +8.20 pp | 82.80 % | +11.21 pp |
| IPC Active | 1.08 inst/cycle | 0.59 inst/cycle | -0.49 | 0.62 inst/cycle | -0.46 | 0.63 inst/cycle | -0.45 |

---

## Iteration 5

**Best so far**: 1.8864× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | activate_autotune_via_lambda_grid_with_yblock1_configs | 1.371× | 1.869× | ✓ |  |
| B | large_xblock_yblock1_num_stages3_lambda_grid | 1.374× | 1.049× | ✓ |  |
| C | fix_static_grid_with_lambda_to_activate_autotune | 1.324× | 1.812× | ✓ |  |
| D | add_triton_autotune | 1.424× | 1.886× | ✓ |  |


### Metric Comparison

| Metric | Baseline | activate_autotune... | Δ | large_xblock_yblo... | Δ | fix_static_grid_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3200 ns | -2780 ns (-46.5%) | 5700 ns | -280 ns (-4.7%) | 3300 ns | -2680 ns (-44.8%) | 3170 ns | -2810 ns (-47.0%) |
| SM Throughput (%) | 16.57 % | 5.21 % | -11.36 pp | 39.66 % | +23.09 pp | 5.22 % | -11.35 pp | 5.09 % | -11.48 pp |
| Memory Throughput (%) | 41.08 % | 11.33 % | -29.75 pp | 29.84 % | -11.24 pp | 10.93 % | -30.15 pp | 11.25 % | -29.83 pp |
| DRAM Throughput (%) | 20.68 % | 2.62 % | -18.06 pp | 2.91 % | -17.77 pp | 2.54 % | -18.14 pp | 2.61 % | -18.07 pp |
| Achieved Occupancy (%) | 55.81 % | 24.79 % | -31.02 pp | 78.05 % | +22.24 pp | 28.67 % | -27.14 pp | 26.79 % | -29.02 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 2.97 | +2.23 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 71.59 % | 78.89 % | +7.30 pp | 25.33 % | -46.26 pp | 79.12 % | +7.53 pp | 80.81 % | +9.22 pp |
| IPC Active | 1.08 inst/cycle | 0.63 inst/cycle | -0.45 | 2.92 inst/cycle | +1.84 | 0.56 inst/cycle | -0.52 | 0.68 inst/cycle | -0.40 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     5950 ns  (1.005×)  — memory_coalescing_via_autotune_tile_swap
Iter 2     3260 ns  (1.834×)  — vectorized_loads_via_max_contiguous_multiple_of_hints
Iter 3     3170 ns  (1.886×)  — shared_memory_transpose_buffer_for_coalesced_stores

Final best: 3170 ns  (1.8864×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 2}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 4}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 32}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +39,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +48,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
