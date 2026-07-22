# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286590 ns — **Best latency**: 140930 ns (**2.0336×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_transpose_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_083/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286590 ns
- **SM Throughput**: 19.56 % | **Memory**: 78.74 % | **DRAM**: 72.00 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_pipeline_stages_for_transpose_kernel | 2.356× | 1.998× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.000× | 0.999× | ✓ |  |
| D | autotune_tile_sizes_for_transpose_coalescing | 2.366× | 2.034× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286590 ns | 143420 ns | -143170 ns (-50.0%) | 286750 ns | +160 ns (+0.1%) | 140930 ns | -145660 ns (-50.8%) |
| SM Throughput (%) | 19.56 % | 48.23 % | +28.67 pp | 19.08 % | -0.48 pp | 47.32 % | +27.76 pp |
| Memory Throughput (%) | 78.74 % | 42.11 % | -36.63 pp | 78.65 % | -0.09 pp | 41.31 % | -37.43 pp |
| DRAM Throughput (%) | 72.00 % | 34.35 % | -37.65 pp | 71.96 % | -0.04 pp | 34.95 % | -37.05 pp |
| Achieved Occupancy (%) | 94.01 % | 91.81 % | -2.20 pp | 93.37 % | -0.64 pp | 91.81 % | -2.20 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 24.00 register/thread | -1 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.04 % | 53.13 % | -26.91 pp | 80.62 % | +0.58 pp | 52.95 % | -27.09 pp |
| IPC Active | 0.79 inst/cycle | 1.86 inst/cycle | +1.07 | 0.78 inst/cycle | -0.01 | 1.88 inst/cycle | +1.09 |

---

## Iteration 2

**Best so far**: 2.0336× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_and_eviction_policy_fix | 2.351× | 1.958× | ✓ |  |
| B | vectorized_loads_via_tl_max_contiguous_and_multiple_of_hints | 2.358× | 2.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286590 ns | 146340 ns | -140250 ns (-48.9%) | 142300 ns | -144290 ns (-50.3%) |
| SM Throughput (%) | 19.56 % | 47.20 % | +27.64 pp | 47.31 % | +27.75 pp |
| Memory Throughput (%) | 78.74 % | 40.99 % | -37.75 pp | 41.35 % | -37.39 pp |
| DRAM Throughput (%) | 72.00 % | 32.89 % | -39.11 pp | 34.54 % | -37.46 pp |
| Achieved Occupancy (%) | 94.01 % | 91.95 % | -2.06 pp | 91.87 % | -2.14 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.04 % | 54.59 % | -25.45 pp | 53.21 % | -26.83 pp |
| IPC Active | 0.79 inst/cycle | 1.80 inst/cycle | +1.01 | 1.89 inst/cycle | +1.10 |


### Key Learnings

#### ❌ Plan A: `vectorized_stores_and_eviction_policy_fix` (1.958×)

Changing the load eviction policy from evict_last to evict_first and adding tl.max_contiguous/tl.multiple_of hints on the store index produced a slight regression (140.93us → 146.34us, ~3.8% slower). The profiling metrics are nearly identical before and after: DRAM throughput dropped marginally (34.95% → 32.89%), No Eligible cycles worsened slightly (52.95% → 54.59%), and eligible warps per scheduler decreased (3.40 → 3.28). The dominant bottleneck — IssueSlotUtilization with ~53% no-eligible cycles driven by memory latency — was not addressed by these store-side and eviction-policy changes, because the stall is caused by strided load latency (stride-128 pattern, 0% L1 hit rate), not by store throughput or L2 eviction pressure. When the kernel is already in a regime where tile shapes have been autotuned and the remaining stall is structural (strided loads with no L1 reuse), eviction policy tweaks and store vectorization hints provide no meaningful benefit.

> **Applicable when**: Never: this strategy did not improve performance. Do not apply evict_first + store contiguity hints as a standalone fix when the kernel is already latency-bound due to strided loads and the tile shape has already been autotuned.

> **Avoid when**: When the kernel is already in a post-autotune regime with ~53% no-eligible scheduler cycles driven by strided load latency (L1 hit rate ~0%, stride-128 pattern). Store-side hints and eviction policy changes cannot reduce load-side memory latency stalls. Also do not apply when DRAM throughput is already low (<40%) and L2 hit rate is moderate (~55%), indicating the bottleneck is latency not bandwidth.

---

## Iteration 3

**Best so far**: 2.0336× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_corner_turning_transpose | 2.352× | 2.017× | ✓ |  |
| C | add_num_stages_to_autotune_configs_for_latency_hiding | 2.352× | 1.967× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_cor... | Δ | add_num_stages_to... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286590 ns | 142110 ns | -144480 ns (-50.4%) | 145700 ns | -140890 ns (-49.2%) | 287010 ns | +420 ns (+0.1%) |
| SM Throughput (%) | 19.56 % | 48.47 % | +28.91 pp | 47.01 % | +27.45 pp | 19.91 % | +0.35 pp |
| Memory Throughput (%) | 78.74 % | 42.32 % | -36.42 pp | 40.85 % | -37.89 pp | 78.56 % | -0.18 pp |
| DRAM Throughput (%) | 72.00 % | 34.65 % | -37.35 pp | 33.04 % | -38.96 pp | 71.91 % | -0.09 pp |
| Achieved Occupancy (%) | 94.01 % | 89.69 % | -4.32 pp | 92.04 % | -1.97 pp | 94.01 % | +0.00 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.04 % | 56.19 % | -23.85 pp | 54.93 % | -25.11 pp | 79.76 % | -0.28 pp |
| IPC Active | 0.79 inst/cycle | 1.79 inst/cycle | +1.00 | 1.81 inst/cycle | +1.02 | 0.81 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan B: `shared_memory_corner_turning_transpose` (2.017×)

Attempting a shared-memory corner-turning rewrite on a Triton transpose kernel that was already autotuned produced essentially no improvement (142.3us → 142.11us, ~0.1% faster, well within noise). The profiling metrics are nearly identical before and after: No Eligible scheduler cycles actually worsened slightly (53.21% → 56.19%), eligible warps per scheduler dropped (3.38 → 2.72), and warp cycles per issued instruction increased (31.17 → 32.7). The diff reveals the actual change was not a true shared-memory corner-turn (no tl.zeros SRAM allocation was introduced); instead, only the tl.max_contiguous hint was moved from the load side to the store side and the eviction_policy='evict_last' was dropped. Without actual staging through shared memory, the stride-128 load pattern is structurally unchanged and the IssueSlotUtilization bottleneck (~53% no-eligible cycles) persists. When a strided transpose kernel is already post-autotune and the remaining stall is structural load latency, superficial index-hint rearrangements without true SRAM staging provide no benefit.

> **Applicable when**: Never: this strategy (moving contiguity hints without actual SRAM staging) did not improve performance. A true shared-memory corner-turn requires allocating a SRAM tile (e.g., via tl.zeros), loading into it coalesced along one axis, and storing from it coalesced along the other axis.

> **Avoid when**: When the kernel is already post-autotune with ~53% no-eligible scheduler cycles driven by stride-128 load latency (L1 hit rate ~0%). Moving tl.max_contiguous hints between load and store without introducing actual SRAM staging cannot reduce load-side memory latency stalls. Do not confuse index-hint rearrangement with true shared-memory corner-turning.

---

## Cumulative Progress

```
Baseline:  286590 ns  (1.000×)
Iter 1     140930 ns  (2.034×)  — autotune_tile_sizes_for_transpose_coalescing

Final best: 140930 ns  (2.0336×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=4),
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
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

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
