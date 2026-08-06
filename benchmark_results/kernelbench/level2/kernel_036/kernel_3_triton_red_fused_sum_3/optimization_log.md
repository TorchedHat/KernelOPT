# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_sum_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7810 ns — **Best latency**: 3550 ns (**2.2000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`expand_autotune_with_xblock_1024_2048_and_num_warps_32_for_warp_level_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_sum_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_036/kernel_3_triton_red_fused_sum_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 32
- **Duration**: 7810 ns
- **SM Throughput**: 2.52 % | **Memory**: 11.04 % | **DRAM**: 11.04 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.30 inst/cycle | **No Eligible**: 92.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_grid_size_via_smaller_xblock_autotune | 1.468× | 2.284× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_grid_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7810 ns | 3420 ns | -4390 ns (-56.2%) |
| SM Throughput (%) | 2.52 % | 1.87 % | -0.65 pp |
| Memory Throughput (%) | 11.04 % | 6.36 % | -4.68 pp |
| DRAM Throughput (%) | 11.04 % | 6.36 % | -4.68 pp |
| Achieved Occupancy (%) | 6.25 % | 6.18 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 92.44 % | 91.64 % | -0.80 pp |
| IPC Active | 0.30 inst/cycle | 0.35 inst/cycle | +0.05 |

---

## Iteration 2

**Best so far**: 2.2836× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_to_pipeline_l1tex_latency | 1.426× | 2.182× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotuned_xblock | 1.458× | 2.140× | ✓ |  |
| C | vectorized_loads_with_evict_first_streaming_policy | 1.254× | 2.200× | ✓ |  |
| D | fix_uncoalesced_shared_memory_store_via_eviction_policy_and_num_warps_autotune | 1.368× | 2.158× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | fix_hardcoded_gri... | Δ | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7810 ns | 3580 ns | -4230 ns (-54.2%) | 3650 ns | -4160 ns (-53.3%) | 3550 ns | -4260 ns (-54.5%) | 3620 ns | -4190 ns (-53.6%) |
| SM Throughput (%) | 2.52 % | 1.92 % | -0.60 pp | 1.90 % | -0.62 pp | 1.90 % | -0.62 pp | 2.00 % | -0.52 pp |
| Memory Throughput (%) | 11.04 % | 6.03 % | -5.01 pp | 5.93 % | -5.11 pp | 6.10 % | -4.94 pp | 5.98 % | -5.06 pp |
| DRAM Throughput (%) | 11.04 % | 6.03 % | -5.01 pp | 5.93 % | -5.11 pp | 6.10 % | -4.94 pp | 5.98 % | -5.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.18 % | -0.07 pp | 6.19 % | -0.06 pp | 6.18 % | -0.07 pp | 6.18 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 92.44 % | 91.04 % | -1.40 pp | 90.93 % | -1.51 pp | 90.91 % | -1.53 pp | 90.72 % | -1.72 pp |
| IPC Active | 0.30 inst/cycle | 0.36 inst/cycle | +0.06 | 0.35 inst/cycle | +0.05 | 0.36 inst/cycle | +0.06 | 0.35 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan A: `increase_num_stages_to_pipeline_l1tex_latency` (2.182×)

Adding num_stages=2/3 variants and larger XBLOCK configs to the autotune search space yielded a 2.18x speedup, but the NCU profiling of the winning kernel shows nearly identical metrics before and after (elapsed cycles 5071→5343, duration 3.42→3.58 µs). The speedup is therefore attributable to the autotune selecting a better overall configuration from the expanded search space rather than to software pipelining hiding L1TEX latency — the CPIStall stall fraction actually increased (49.6%→57.9%) and elapsed cycles increased slightly in the profiled kernel. The real gain likely came from a different kernel in the workload being optimized, or from the autotune selecting a config that reduces total wall-clock time through better cache behavior across the full model, not from the num_stages change itself.

> **Applicable when**: Triton kernel with CPIStall on L1TEX as dominant stall (>40% of cycles), very low eligible warps per scheduler (<0.1), and a reduction loop with multiple iterations where software pipelining can overlap loads with compute. Expanding autotune config space with num_stages=2/3 and larger XBLOCK values is low-risk when the kernel body is unchanged.

> **Avoid when**: The reduction loop runs only a single iteration (R0_BLOCK == r0_numel and the range executes once) — in this case num_stages pipelining has no loop body to pipeline across and provides no benefit. Also avoid when the profiled kernel shows elapsed cycles increasing after the change, as seen here (5071→5343 cycles), indicating the winning autotune config is not the one being profiled.

---

## Iteration 3

**Best so far**: 2.2836× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_better_cache_reuse_and_sm_utilization | 1.243× | 2.257× | ✓ |  |
| B | fix_hardcoded_grid_with_lambda_and_expand_xblock_tile_configs | 1.571× | 2.158× | ✓ | **✓ best** |
| C | fix_hardcoded_grid_to_use_autotuned_xblock_via_lambda | 1.343× | 2.158× | ✓ |  |
| D | fix_hardcoded_grid_and_expand_xblock_configs_for_tile_parallelism | 1.446× | 2.200× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7810 ns | 3460 ns | -4350 ns (-55.7%) | 3620 ns | -4190 ns (-53.6%) | 3620 ns | -4190 ns (-53.6%) | 3550 ns | -4260 ns (-54.5%) |
| SM Throughput (%) | 2.52 % | 1.96 % | -0.56 pp | 1.97 % | -0.55 pp | 1.93 % | -0.59 pp | 1.97 % | -0.55 pp |
| Memory Throughput (%) | 11.04 % | 6.29 % | -4.75 pp | 6.03 % | -5.01 pp | 6.01 % | -5.03 pp | 6.09 % | -4.95 pp |
| DRAM Throughput (%) | 11.04 % | 6.29 % | -4.75 pp | 6.03 % | -5.01 pp | 6.01 % | -5.03 pp | 6.09 % | -4.95 pp |
| Achieved Occupancy (%) | 6.25 % | 6.18 % | -0.07 pp | 6.19 % | -0.06 pp | 6.19 % | -0.06 pp | 6.18 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 92.44 % | 91.21 % | -1.23 pp | 90.76 % | -1.68 pp | 91.02 % | -1.42 pp | 90.84 % | -1.60 pp |
| IPC Active | 0.30 inst/cycle | 0.34 inst/cycle | +0.04 | 0.32 inst/cycle | +0.02 | 0.34 inst/cycle | +0.04 | 0.35 inst/cycle | +0.05 |

---

## Iteration 4

**Best so far**: 2.2836× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_store_offset_linearization | 1.238× | 2.257× | ✓ |  |
| B | swap_accumulator_layout_to_eliminate_shared_memory_bank_conflicts | 1.451× | 2.219× | ✓ |  |
| C | fix_hardcoded_grid_launcher_to_use_lambda_meta_xblock | 1.558× | 2.182× | ✓ |  |
| D | fix_hardcoded_grid_lambda_to_respect_autotuned_xblock | 1.377× | 2.158× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | swap_accumulator_... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7810 ns | 3460 ns | -4350 ns (-55.7%) | 3520 ns | -4290 ns (-54.9%) | 3580 ns | -4230 ns (-54.2%) | 3620 ns | -4190 ns (-53.6%) |
| SM Throughput (%) | 2.52 % | 1.82 % | -0.70 pp | 1.87 % | -0.65 pp | 1.88 % | -0.64 pp | 1.87 % | -0.65 pp |
| Memory Throughput (%) | 11.04 % | 6.27 % | -4.77 pp | 6.15 % | -4.89 pp | 6.03 % | -5.01 pp | 6.03 % | -5.01 pp |
| DRAM Throughput (%) | 11.04 % | 6.27 % | -4.77 pp | 6.15 % | -4.89 pp | 6.03 % | -5.01 pp | 6.03 % | -5.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.19 % | -0.06 pp | 6.18 % | -0.07 pp | 6.19 % | -0.06 pp | 6.18 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 92.44 % | 91.01 % | -1.43 pp | 92.63 % | +0.19 pp | 90.94 % | -1.50 pp | 90.64 % | -1.80 pp |
| IPC Active | 0.30 inst/cycle | 0.32 inst/cycle | +0.02 | 0.36 inst/cycle | +0.06 | 0.34 inst/cycle | +0.04 | 0.35 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan A: `fix_shared_memory_bank_conflicts_via_store_offset_linearization` (2.257×)

When a Triton kernel computes a shared memory store offset using integer division and modulo (e.g., `(x1 // 2) * 512 + (x1 % 2) * 256 + x0`), the compiler cannot prove the access pattern is bank-conflict-free, and the resulting stride pattern causes systematic 3.0–3.5-way shared memory bank conflicts (NCU: UncoalescedSharedAccess, 62% excessive wavefronts). Replacing the non-linear expression with its algebraically equivalent linear form (`x1 * 256 + x0`) allows the compiler to generate a single multiply-add and verify bank-conflict-freedom, reducing executed instructions from 54464 to 52160 and warp cycles per executed instruction from 12.31 to 11.34. The per-kernel NCU duration was unchanged (3.46 µs) because the kernel is severely under-occupying the GPU (0.03 waves/SM), so the wall-clock speedup of 2.26x is attributable to the combined effect of this change plus the expanded autotune config space (XBLOCK=1024/2048 added in the same iteration).

> **Applicable when**: Triton kernel where the store offset to shared memory (or output tensor) is computed via integer division and modulo of the thread index, producing a non-linear expression that the compiler cannot simplify to a bank-conflict-free stride pattern. NCU reports UncoalescedSharedAccess with >50% excessive wavefronts and SharedMemoryConflicts with 3+ way bank conflicts on stores. The non-linear expression is algebraically equivalent to a simpler linear form.

> **Avoid when**: The non-linear offset expression is NOT algebraically equivalent to a linear form (i.e., the interleaving is intentional and encodes a non-trivial permutation). The store target is global memory rather than shared memory (bank conflicts only apply to shared memory). The kernel is already using a linear stride pattern. The bank conflict NCU rule is not triggered or reports <2-way conflicts.

#### ✅ Plan C: `fix_hardcoded_grid_launcher_to_use_lambda_meta_xblock` (2.182×)

This iteration applied three changes simultaneously: switching the launcher grid from a fixed tuple to a lambda using meta['XBLOCK'], changing the load eviction policy from 'evict_first' back to 'evict_last', and restructuring the autotune config space (adding num_stages=3 configs and small XBLOCK configs, removing some prior configs). The NCU-profiled kernel shows essentially unchanged or marginally worse per-kernel metrics (duration 3.46→3.58 µs, elapsed cycles 5120→5338, grid size still 64, occupancy still ~6.18%), indicating the 2.18x wall-clock speedup is attributable to autotune selecting a better configuration at the system level rather than to any of the individual code changes improving the profiled kernel's execution. The lambda grid fix is structurally correct and necessary for autotune to vary XBLOCK, but the profiled winning config still uses XBLOCK=128 (grid=64), so the grid coverage improvement was not realized in this run.

> **Applicable when**: Triton autotuned kernel where the launcher uses a fixed grid tuple (e.g., cdiv(xnumel, 128)) instead of a lambda, preventing autotune from varying XBLOCK. LaunchConfiguration NCU rule triggered with grid_size < sm_count and waves_per_sm < 0.1. The fix is structurally required for correctness when autotune selects XBLOCK != the hardcoded value.

> **Avoid when**: The autotune winning config always selects the same XBLOCK as the hardcoded grid value (no benefit from lambda). The kernel is already using a lambda grid. Per-kernel NCU metrics are the primary optimization target — this change showed no per-kernel improvement; the wall-clock speedup is a system-level effect from autotune config selection.

---

## Iteration 5

**Best so far**: 2.2836× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_r0_block_tile_size_for_better_l1_reuse_and_work_per_block | 1.369× | 2.257× | ✓ |  |
| B | increase_r0_block_tile_size_and_linearize_store_offset | 1.435× | 2.257× | ✓ |  |
| C | expand_autotune_with_xblock_1024_2048_and_num_warps_32_for_warp_level_parallelism | 1.582× | 2.200× | ✓ | **✓ best** |
| D | linearize_store_offset_and_expand_xblock_tile_configs | 1.369× | 2.200× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_r0_block... | Δ | increase_r0_block... | Δ | expand_autotune_w... | Δ | linearize_store_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7810 ns | 3460 ns | -4350 ns (-55.7%) | 3460 ns | -4350 ns (-55.7%) | 3550 ns | -4260 ns (-54.5%) | 3550 ns | -4260 ns (-54.5%) |
| SM Throughput (%) | 2.52 % | 1.89 % | -0.63 pp | 1.90 % | -0.62 pp | 1.88 % | -0.64 pp | 1.92 % | -0.60 pp |
| Memory Throughput (%) | 11.04 % | 6.26 % | -4.78 pp | 6.30 % | -4.74 pp | 6.09 % | -4.95 pp | 6.07 % | -4.97 pp |
| DRAM Throughput (%) | 11.04 % | 6.26 % | -4.78 pp | 6.30 % | -4.74 pp | 6.09 % | -4.95 pp | 6.07 % | -4.97 pp |
| Achieved Occupancy (%) | 6.25 % | 6.19 % | -0.06 pp | 6.19 % | -0.06 pp | 6.18 % | -0.07 pp | 6.18 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 92.44 % | 91.44 % | -1.00 pp | 91.83 % | -0.61 pp | 90.77 % | -1.67 pp | 91.01 % | -1.43 pp |
| IPC Active | 0.30 inst/cycle | 0.31 inst/cycle | +0.01 | 0.32 inst/cycle | +0.02 | 0.36 inst/cycle | +0.06 | 0.35 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  7810 ns  (1.000×)
Iter 1     3420 ns  (2.284×)  — increase_grid_size_via_smaller_xblock_autotune

Final best: 3550 ns  (2.2000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,52 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original num_stages=1 configs (unchanged)
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        # num_stages=2: pipeline global loads to hide L1TEX latency
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        # num_stages=3: deeper pipelining
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=3),
+        # Larger XBLOCK: more work per block, better L2 reuse
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8,  num_stages=3),
+        # NEW: small XBLOCK configs to maximize grid coverage and occupancy
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+        # NEW: large XBLOCK with high num_warps for warp-level parallelism to hide L1TEX latency
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 128}, num_warps=32, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 128}, num_warps=32, num_stages=2),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 128}, num_warps=32, num_stages=1),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048, 'R0_BLOCK': 128}, num_warps=32, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 8192
@@ -32,14 +78,13 @@

     store_offset = (x1 // 2) * 512 + (x1 % 2) * 256 + x0
     tl.store(out_ptr0 + store_offset, tmp2, mask=xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 8192
     r0_numel = 128
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_sum_3[grid](arg_0, arg_1, 8192, 128, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_sum_3[grid](arg_0, arg_1, 8192, 128)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
