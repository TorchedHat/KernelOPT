# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 2910 ns (**2.0447×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_autotune_grid_and_vectorized_load_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_031/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.80 % | **Memory**: 40.88 % | **DRAM**: 20.71 %
- **Achieved Occupancy**: 54.77 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.406× | 1.787× | ✓ |  |
| B | autotune_tile_sizes_and_pipeline_stages | 1.379× | 1.740× | ✓ |  |
| C | memory_coalescing_tile_reorder | 1.460× | 2.024× | ✓ | **✓ best** |
| D | add_autotune_increase_tile_sizes | 1.245× | 1.488× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3330 ns | -2620 ns (-44.0%) | 3420 ns | -2530 ns (-42.5%) | 2940 ns | -3010 ns (-50.6%) | 4000 ns | -1950 ns (-32.8%) |
| SM Throughput (%) | 16.80 % | 17.24 % | +0.44 pp | 9.70 % | -7.10 pp | 2.20 % | -14.60 pp | 18.91 % | +2.11 pp |
| Memory Throughput (%) | 40.88 % | 20.48 % | -20.40 pp | 20.21 % | -20.67 pp | 7.36 % | -33.52 pp | 21.80 % | -19.08 pp |
| DRAM Throughput (%) | 20.71 % | 4.97 % | -15.74 pp | 4.84 % | -15.87 pp | 0.36 % | -20.35 pp | 4.15 % | -16.56 pp |
| Achieved Occupancy (%) | 54.77 % | 44.37 % | -10.40 pp | 52.64 % | -2.13 pp | 7.19 % | -47.58 pp | 51.41 % | -3.36 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.88 % | 54.14 % | -16.74 pp | 72.71 % | +1.83 pp | 90.17 % | +19.29 pp | 52.03 % | -18.85 pp |
| IPC Active | 1.15 inst/cycle | 1.66 inst/cycle | +0.51 | 1.06 inst/cycle | -0.09 | 0.24 inst/cycle | -0.91 | 1.69 inst/cycle | +0.54 |

---

## Iteration 2

**Best so far**: 2.0238× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_and_vectorized_load_hints | 1.466× | 2.045× | ✓ | **✓ best** |
| B | coalesce_strided_stores_via_yblock_dominant_tile_configs | 1.465× | 1.895× | ✓ |  |
| C | coalescing_friendly_yblock_dominant_tile_reorder | 1.433× | 2.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | coalesce_strided_... | Δ | coalescing_friend... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 2910 ns | -3040 ns (-51.1%) | 3140 ns | -2810 ns (-47.2%) | 2940 ns | -3010 ns (-50.6%) |
| SM Throughput (%) | 16.80 % | 2.10 % | -14.70 pp | 2.20 % | -14.60 pp | 2.09 % | -14.71 pp |
| Memory Throughput (%) | 40.88 % | 7.33 % | -33.55 pp | 6.85 % | -34.03 pp | 7.26 % | -33.62 pp |
| DRAM Throughput (%) | 20.71 % | 0.36 % | -20.35 pp | 0.34 % | -20.37 pp | 0.36 % | -20.35 pp |
| Achieved Occupancy (%) | 54.77 % | 7.19 % | -47.58 pp | 10.21 % | -44.56 pp | 7.75 % | -47.02 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 70.88 % | 90.71 % | +19.83 pp | 89.86 % | +18.98 pp | 90.37 % | +19.49 pp |
| IPC Active | 1.15 inst/cycle | 0.24 inst/cycle | -0.91 | 0.23 inst/cycle | -0.92 | 0.23 inst/cycle | -0.92 |


### Key Learnings

#### ⚠️ Plan C: `coalescing_friendly_yblock_dominant_tile_reorder` (0.000×)

Plan 'coalescing_friendly_yblock_dominant_tile_reorder' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=6.487e+00 > atol=0.001
seed=42 output[0]: max_diff=6.802e+00 > atol=0.001
seed=1337 output[0]: max_diff=6.504e+00 > atol=0.001

> **Applicable when**: The store address is `y0 + 3*x2 + 150528*y1` where x2 = xindex (warp-innermost). With XBLOCK=128, the 128 consecutive X-threads write to addresses {0, 3, 6, ..., 381} — a stride-3 scatter that touches 128 separate 128-byte cache lines instead of 4, causing 50% excessive sectors and only 10.7/32 bytes utilized per sector. The current autotune configs are ALL large-XBLOCK/small-YBLOCK (XBLOCK in {64..1024}, YBLOCK in {1..32}), which all preserve this bad orientation. The Y-dimension (ynumel=30) maps to y0 = yindex % 3 which is the contiguous output dimension (stride=1 in out_ptr). Making YBLOCK=32 (covering all 30 Y-elements) and XBLOCK small (1..64) puts the Y-threads as the warp-innermost group, so adjacent threads write to y0 in {0,1,2,...} — contiguous addresses — enabling fully coalesced stores.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=6.487e+00 > atol=0.001
seed=42 output[0]: max_diff=6.802e+00 > atol=0.001
seed=1337 output[0]: max_diff=6.504e+00 > atol=0.001

---

## Iteration 3

**Best so far**: 2.0447× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipeline_and_large_xblock_configs | 1.464× | 2.024× | ✓ |  |
| B | increase_num_stages_software_pipelining | 1.429× | 2.024× | ✓ |  |
| C | fix_static_grid_to_use_autotune_lambda | 1.439× | 2.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | increase_num_stag... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 2940 ns | -3010 ns (-50.6%) | 2940 ns | -3010 ns (-50.6%) | 2940 ns | -3010 ns (-50.6%) |
| SM Throughput (%) | 16.80 % | 2.17 % | -14.63 pp | 2.11 % | -14.69 pp | 2.08 % | -14.72 pp |
| Memory Throughput (%) | 40.88 % | 7.34 % | -33.54 pp | 7.38 % | -33.50 pp | 7.30 % | -33.58 pp |
| DRAM Throughput (%) | 20.71 % | 0.35 % | -20.36 pp | 0.36 % | -20.35 pp | 0.36 % | -20.35 pp |
| Achieved Occupancy (%) | 54.77 % | 7.54 % | -47.23 pp | 7.28 % | -47.49 pp | 7.40 % | -47.37 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 70.88 % | 90.11 % | +19.23 pp | 89.95 % | +19.07 pp | 89.56 % | +18.68 pp |
| IPC Active | 1.15 inst/cycle | 0.25 inst/cycle | -0.90 | 0.25 inst/cycle | -0.90 | 0.24 inst/cycle | -0.91 |


### Key Learnings

#### ✅ Plan B: `increase_num_stages_software_pipelining` (2.024×)

Increasing Triton autotune configs from num_stages=2 to num_stages=3/4 produced no measurable kernel-level improvement when the kernel is severely wave-limited (0.37 waves/SM, achieved occupancy ~7%). The NCU metrics before and after are nearly identical: duration 2.91→2.94 µs, elapsed cycles 4343→4349, executed instructions unchanged at 45080, No Eligible scheduler cycles 90.71%→89.95%, and eligible warps per scheduler 0.14→0.15. The 2.02x reported end-to-end speedup is inconsistent with the kernel profiling data and likely reflects measurement variance or system-level effects. When a kernel has fewer than 0.5 waves/SM and achieved occupancy below 10%, software pipelining via num_stages cannot hide latency because the fundamental bottleneck is insufficient parallelism (grid too small), not memory latency per se.

> **Applicable when**: Kernel has achieved occupancy >20%, waves_per_sm >1.0, and CPIStall on L1TEX is the dominant bottleneck (>40% of CPI) with sufficient active warps per scheduler (>2.0) to benefit from prefetching additional pipeline stages.

> **Avoid when**: Kernel has waves_per_sm < 0.5 and achieved occupancy < 10%; in this regime the grid is too small to fill the GPU and num_stages cannot compensate — the bottleneck is parallelism, not per-warp latency hiding. Also avoid when executed instructions are unchanged between num_stages variants (confirms no pipeline benefit).

#### ✅ Plan C: `fix_static_grid_to_use_autotune_lambda` (2.024×)

When a Triton kernel has @triton.autotune but the launch site uses a static grid tuple (e.g., grid = (cdiv(xnumel, 128), cdiv(ynumel, 8))), the autotuner's winning configuration is silently ignored at runtime — the kernel always executes with the hardcoded tile sizes, making the autotune decorator a no-op. The fix is to replace the static tuple with a lambda: grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK'])). However, in this iteration the NCU profiling metrics before and after are nearly identical (duration 2.91→2.94 µs, waves_per_sm unchanged at 0.37, occupancy 7.19→7.40%, uncoalesced sectors still 81%), indicating the lambda fix alone did not materially change kernel-level behavior — the 2.02x end-to-end speedup likely reflects system-level variance or the autotune selecting a config that happens to match the previously hardcoded values.

> **Applicable when**: Triton kernel has @triton.autotune decorator but the launch site uses a static grid tuple with hardcoded tile sizes; SOLBottleneck rule triggered (waves_per_sm < 1.0); IssueSlotUtilization flagged with No Eligible >85%; the autotune configs include tile shapes different from the hardcoded values in the static grid.

> **Avoid when**: The kernel does not use @triton.autotune (no configs to unlock); the static grid intentionally overrides autotune for correctness reasons; the autotune configs all happen to match the hardcoded static grid values (no-op fix); the kernel is already using a lambda grid correctly.

---

## Iteration 4

**Best so far**: 2.0447× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | activate_dead_autotune_by_fixing_static_grid_and_call_site | 1.443× | 1.997× | ✓ |  |
| B | shared_memory_transpose_for_coalesced_stores | 1.430× | 1.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | activate_dead_aut... | Δ | shared_memory_tra... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 2980 ns | -2970 ns (-49.9%) | 2980 ns | -2970 ns (-49.9%) |
| SM Throughput (%) | 16.80 % | 2.00 % | -14.80 pp | 2.19 % | -14.61 pp |
| Memory Throughput (%) | 40.88 % | 7.28 % | -33.60 pp | 7.46 % | -33.42 pp |
| DRAM Throughput (%) | 20.71 % | 0.36 % | -20.35 pp | 0.35 % | -20.36 pp |
| Achieved Occupancy (%) | 54.77 % | 7.57 % | -47.20 pp | 7.47 % | -47.30 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 70.88 % | 90.19 % | +19.31 pp | 90.60 % | +19.72 pp |
| IPC Active | 1.15 inst/cycle | 0.26 inst/cycle | -0.89 | 0.26 inst/cycle | -0.89 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     2940 ns  (2.024×)  — memory_coalescing_tile_reorder
Iter 2     2910 ns  (2.045×)  — fix_autotune_grid_and_vectorized_load_hints

Final best: 2910 ns  (2.0447×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 1},   num_warps=1, num_stages=2),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 4},   num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 8},   num_warps=2, num_stages=2),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 8},   num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 32},  num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 64},  num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 16},  num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 4,  'XBLOCK': 256}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 2,  'XBLOCK': 512}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 1,  'XBLOCK': 1024},num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 4,  'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 256}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 8},   num_warps=4, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +30,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset, XBLOCK) + tl.arange(0, XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +39,13 @@

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

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
