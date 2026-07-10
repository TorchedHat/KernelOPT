# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3360 ns (**1.7440×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_025/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 16.19 % | **Memory**: 41.99 % | **DRAM**: 21.14 %
- **Achieved Occupancy**: 54.64 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.381× | 1.744× | ✓ | **✓ best** |
| B | add_autotune_fix_uncoalesced_stores_via_tile_reshape | 1.376× | 1.744× | ✓ |  |
| D | add_autotune_increase_tile_sizes_and_warps | 1.222× | 1.376× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_autotune_fix_... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3360 ns | -2500 ns (-42.7%) | 3360 ns | -2500 ns (-42.7%) | 4260 ns | -1600 ns (-27.3%) |
| SM Throughput (%) | 16.19 % | 17.04 % | +0.85 pp | 17.20 % | +1.01 pp | 27.02 % | +10.83 pp |
| Memory Throughput (%) | 41.99 % | 20.28 % | -21.71 pp | 20.44 % | -21.55 pp | 21.46 % | -20.53 pp |
| DRAM Throughput (%) | 21.14 % | 4.93 % | -16.21 pp | 4.94 % | -16.20 pp | 3.89 % | -17.25 pp |
| Achieved Occupancy (%) | 54.64 % | 43.76 % | -10.88 pp | 45.74 % | -8.90 pp | 68.28 % | +13.64 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.19 % | 62.05 % | -9.14 pp | 55.01 % | -16.18 pp | 36.41 % | -34.78 pp |
| IPC Active | 1.16 inst/cycle | 1.68 inst/cycle | +0.52 | 1.56 inst/cycle | +0.40 | 2.43 inst/cycle | +1.27 |

---

## Iteration 2

**Best so far**: 1.744× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_load_evict_first | 1.292× | 1.760× | ✓ |  |
| B | add_num_stages_software_pipelining_to_autotune | 1.367× | 1.679× | ✓ |  |
| C | fix_autotune_launch_site_and_add_large_yblock_configs | 1.243× | 1.409× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | add_num_stages_so... | Δ | fix_autotune_laun... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3330 ns | -2530 ns (-43.2%) | 3490 ns | -2370 ns (-40.4%) | 4160 ns | -1700 ns (-29.0%) |
| SM Throughput (%) | 16.19 % | 17.46 % | +1.27 pp | 16.35 % | +0.16 pp | 27.32 % | +11.13 pp |
| Memory Throughput (%) | 41.99 % | 20.32 % | -21.67 pp | 19.51 % | -22.48 pp | 21.69 % | -20.30 pp |
| DRAM Throughput (%) | 21.14 % | 4.94 % | -16.20 pp | 4.74 % | -16.40 pp | 3.99 % | -17.15 pp |
| Achieved Occupancy (%) | 54.64 % | 43.91 % | -10.73 pp | 43.80 % | -10.84 pp | 67.84 % | +13.20 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.19 % | 54.09 % | -17.10 pp | 54.42 % | -16.77 pp | 37.93 % | -33.26 pp |
| IPC Active | 1.16 inst/cycle | 1.73 inst/cycle | +0.57 | 1.67 inst/cycle | +0.51 | 2.39 inst/cycle | +1.23 |


### Key Learnings

#### ✅ Plan A: `eviction_policy_streaming_load_evict_first` (1.760×)

For streaming kernels where each input element is read exactly once (zero temporal reuse), changing tl.load's eviction_policy from 'evict_last' to 'evict_first' signals the L2 to deprioritize loaded cache lines immediately after use. This frees L2 capacity for other traffic (e.g., strided stores that benefit from L2 buffering), reduces L2 slice workload imbalance, and modestly improves scheduler eligible warps per cycle. The profiling showed no-eligible-warp rate dropped from 62% to 54% and active warps per scheduler nearly doubled (5.94 → 10.22), though the overall duration improvement was marginal (~1%) since the dominant bottleneck (small grid, uncoalesced stores) remained.

> **Applicable when**: Triton kernel performs a streaming read where each input element is accessed exactly once (e.g., permutation, transpose, format conversion). L2 workload imbalance is present (max-above-avg > 15%), and the kernel has competing memory streams (e.g., strided stores that need L2 buffering). UncoalescedGlobalAccess and WorkloadImbalance (L2 Slices) rules are triggered.

> **Avoid when**: Input data is reused across iterations or tiles (e.g., reduction, matmul, convolution with filter reuse) — evict_first would evict data before reuse, causing cache misses. Also avoid when the kernel is already compute-bound or when L2 hit rate is low (< 50%), indicating data is not resident in L2 anyway.

#### ✅ Plan B: `add_num_stages_software_pipelining_to_autotune` (1.679×)

Expanding a Triton @triton.autotune config list to include num_stages=2 and num_stages=3 variants of all existing tile configs allows the autotuner to select software-pipelined configurations that can hide memory latency. However, for single-tile pointwise/permutation kernels (no inner loop), num_stages primarily affects compiler-level instruction scheduling rather than a traditional prefetch loop, and the benefit is limited or absent — in this case the NCU profiling shows nearly identical metrics before and after (elapsed cycles 4994 vs 5196, no-eligible rate 62% vs 54%), suggesting the 1.68x wall-time speedup was driven by the autotuner selecting a better tile shape from the expanded config space rather than software pipelining itself. The key causal signal is that eligible warps per scheduler improved slightly (1.76 → 2.08) and active warps increased (5.94 → 7.03), consistent with a better tile shape being selected.

> **Applicable when**: Triton autotune config list uses only num_stages=1 across all configs; kernel shows IssueSlotUtilization bottleneck with no-eligible-warp rate >50%; L2 hit rate is high (>70%) indicating memory latency is from L2 not DRAM; kernel is a pointwise or permutation op with no inner reduction loop.

> **Avoid when**: Kernel already explores multiple num_stages values in autotune; kernel has a tight inner loop (GEMM, reduction) where num_stages directly controls prefetch depth and the effect is well-understood; the config space expansion causes autotune overhead to dominate first-run latency in latency-sensitive deployments; NCU profiling shows compute-bound rather than memory-latency-bound behavior.

---

## Iteration 3

**Best so far**: 1.744× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_transpose_for_coalesced_stores | 1.125× | 0.139× | ✓ |  |
| B | fix_autotune_bypass_with_lambda_grid_and_full_y_configs | 1.362× | 1.776× | ✓ |  |
| C | fix_autotune_bypass_lambda_grid_and_coalescing_configs | 1.301× | 1.454× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | fix_autotune_bypa... | Δ | fix_autotune_bypa... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 42270 ns | +36410 ns (+621.3%) | 3300 ns | -2560 ns (-43.7%) | 4030 ns | -1830 ns (-31.2%) |
| SM Throughput (%) | 16.19 % | 38.01 % | +21.82 pp | 16.97 % | +0.78 pp | 26.74 % | +10.55 pp |
| Memory Throughput (%) | 41.99 % | 67.47 % | +25.48 pp | 20.83 % | -21.16 pp | 21.23 % | -20.76 pp |
| DRAM Throughput (%) | 21.14 % | 2.91 % | -18.23 pp | 5.02 % | -16.12 pp | 4.13 % | -17.01 pp |
| Achieved Occupancy (%) | 54.64 % | 12.38 % | -42.26 pp | 48.76 % | -5.88 pp | 69.01 % | +14.37 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 11.88 | +11.14 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.19 % | 56.10 % | -15.09 pp | 54.11 % | -17.08 pp | 36.97 % | -34.22 pp |
| IPC Active | 1.16 inst/cycle | 1.72 inst/cycle | +0.56 | 1.41 inst/cycle | +0.25 | 2.41 inst/cycle | +1.25 |


### Key Learnings

#### ❌ Plan A: `shared_memory_staged_transpose_for_coalesced_stores` (0.139×)

Applying tl.max_contiguous/tl.multiple_of hints to a stride-3 store address and expanding autotune configs to include YBLOCK=32 caused the autotuner to select a configuration with 131 KB of dynamic shared memory per block. This reduced Block Limit Shared Mem from 33 to 1 block per SM, collapsing theoretical occupancy from 100% to 12.5% and achieved occupancy from 43.9% to 12.4%. Despite waves/SM increasing from 0.74 to 11.88 (more total work), the severe occupancy collapse caused elapsed cycles to balloon from 4990 to 63333 — a 12.7x slowdown. The contiguity hints on a stride-3 store address did not resolve the uncoalesced access pattern (excess sectors remained ~52%), and the register count rose from 18 to 38, compounding the occupancy loss.

> **Applicable when**: Never — this is a negative result. Do not apply tl.max_contiguous/tl.multiple_of hints to stride-N (N>1) store addresses, and do not add YBLOCK configs that cause shared memory per block to exceed ~32KB when the kernel has many active blocks per SM.

> **Avoid when**: Always avoid: (1) adding autotune configs with XBLOCK*YBLOCK products so large they require >32KB shared memory per block on kernels where occupancy is already marginal; (2) applying tl.max_contiguous hints to the non-contiguous (strided) dimension of a scatter store — the hint does not fix the hardware-level stride and only confuses the compiler; (3) combining large YBLOCK (e.g., YBLOCK=32) with large XBLOCK (e.g., XBLOCK=1024) in autotune configs without verifying the resulting smem footprint stays within occupancy-safe bounds.

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3360 ns  (1.744×)  — vectorized_loads_with_contiguous_hints

Final best: 3360 ns  (1.7440×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +25,14 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

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
