# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3390 ns — **Best latency**: 4380 ns (**0.7740×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_5_triton_poi_fused__scaled_dot_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3390 ns
- **SM Throughput**: 10.05 % | **Memory**: 7.21 % | **DRAM**: 4.93 %
- **Achieved Occupancy**: 53.15 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 68.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_better_wave_utilization | 1.050× | 1.080× | ✓ |  |
| B | autotune_xblock_num_warps_num_stages | 1.030× | 1.009× | ✓ |  |
| C | autotune_xblock_increase_tile_size | 1.053× | 0.774× | ✓ | **✓ best** |
| D | autotune_xblock_for_underutilized_pointwise_kernel | 1.050× | 1.080× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_n... | Δ | autotune_xblock_i... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3140 ns | -250 ns (-7.4%) | 3360 ns | -30 ns (-0.9%) | 4380 ns | +990 ns (+29.2%) | 3140 ns | -250 ns (-7.4%) |
| SM Throughput (%) | 10.05 % | 12.98 % | +2.93 pp | 10.32 % | +0.27 pp | 41.65 % | +31.60 pp | 12.97 % | +2.92 pp |
| Memory Throughput (%) | 7.21 % | 7.30 % | +0.09 pp | 7.38 % | +0.17 pp | 20.71 % | +13.50 pp | 7.46 % | +0.25 pp |
| DRAM Throughput (%) | 4.93 % | 5.34 % | +0.41 pp | 4.97 % | +0.04 pp | 3.80 % | -1.13 pp | 5.33 % | +0.40 pp |
| Achieved Occupancy (%) | 53.15 % | 45.66 % | -7.49 pp | 52.15 % | -1.00 pp | 69.01 % | +15.86 pp | 45.77 % | -7.38 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 30.00 register/thread | +14 | 18.00 register/thread | +2 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 | 0.75 | +0.00 |
| No Eligible (%) | 68.16 % | 56.57 % | -11.59 pp | 66.65 % | -1.51 pp | 26.16 % | -42.00 pp | 57.76 % | -10.40 pp |
| IPC Active | 1.15 inst/cycle | 1.55 inst/cycle | +0.40 | 1.12 inst/cycle | -0.03 | 2.78 inst/cycle | +1.63 | 1.53 inst/cycle | +0.38 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_increase_tile_size` (0.774×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) selected a 256-thread block that nearly doubled register usage (16→30 registers/thread), which constrained occupancy and caused the Block Limit Registers and Block Limit Warps to drop from 32 to 8 blocks/SM. Although scheduler idle cycles improved (68%→26%) and compute SOL rose (10%→42%), the executed instruction count exploded 5.3x (277K→1.46M) due to uncoalesced global accesses (50% excessive sectors flagged by UncoalescedGlobalAccess rule) introduced by the larger tile stride pattern, and the kernel duration increased from 3.39µs to 4.38µs. When the original kernel is already register-efficient (16 regs/thread) and the bottleneck is grid size rather than per-CTA work, increasing XBLOCK can cause the autotuner to select a config that trades register pressure for throughput in a way that is net-negative due to uncoalesced access patterns at larger strides.

> **Applicable when**: Never: this specific strategy caused regression. In general, autotune over XBLOCK is applicable when the kernel has low register usage (<20 regs/thread), the grid is small (<1 wave), and the memory access pattern is coalesced so larger tiles don't introduce stride-induced uncoalesced accesses.

> **Avoid when**: The kernel uses complex multi-dimensional strided index arithmetic (e.g., x0 + stride*x1 + large_stride*x2) where increasing XBLOCK causes threads within a warp to access non-contiguous memory addresses, triggering UncoalescedGlobalAccess. Also avoid when the kernel already has low register count (≤16 regs/thread) and increasing block size would push registers/thread above ~24, reducing Block Limit Registers and capping occupancy. Do not apply when the instruction count increase from larger tiles outweighs the latency-hiding benefit.

#### ✅ Plan D: `autotune_xblock_for_underutilized_pointwise_kernel` (1.080×)

For small pointwise Triton kernels where the scheduler No-Eligible rate exceeds ~60% and Warp Cycles Per Issued Instruction is high (>20), the dominant bottleneck is insufficient independent memory operations per CTA to hide L1TEX latency. Adding @triton.autotune with larger XBLOCK configs (512–2048) and varied num_warps allows the runtime to select a tile size that gives each CTA more independent loads, reducing warp stall cycles and raising Issue Slots Busy. In this case, elapsed cycles dropped from 5037 to 4647 (~8%) and Warp Cycles Per Issued Instruction fell from 27.0 to 17.3, with Issue Slots Busy rising from 31% to 41%, despite waves_per_sm remaining at 0.75 (grid-size-limited).

> **Applicable when**: Triton pointwise kernel with: waves_per_sm < 1.5, Scheduler No-Eligible > 50%, Warp Cycles Per Issued Instruction > 20, CPIStall dominated by L1TEX latency, and XBLOCK fixed at a small value (e.g. 128) relative to total elements (>100K). The kernel body is purely elementwise with no shared memory or inter-warp dependencies.

> **Avoid when**: Kernel is already memory-bandwidth-bound (Memory SOL > 60%) or compute-bound (Compute SOL > 60%); increasing XBLOCK would not help and may hurt occupancy. Also avoid if the kernel uses shared memory sized to XBLOCK, as larger tiles would exhaust SMEM and reduce occupancy. Do not apply if the grid is already large (waves_per_sm > 2) — the bottleneck is elsewhere.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.047× | 1.018× | ✓ |  |
| B | persistent_tile_scheduling | 1.006× | 1.009× | ✓ |  |
| C | persistent_tile_scheduling | 0.734× | 0.583× | ✓ |  |
| D | persistent_tile_scheduling | 1.003× | 0.921× | ✓ |  |
| E | add_triton_autotune | 1.029× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3330 ns | -60 ns (-1.8%) | 3360 ns | -30 ns (-0.9%) | 5820 ns | +2430 ns (+71.7%) | 3680 ns | +290 ns (+8.6%) | 3390 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 10.05 % | 11.87 % | +1.82 pp | 12.33 % | +2.28 pp | 29.71 % | +19.66 pp | 20.00 % | +9.95 pp | 10.80 % | +0.75 pp |
| Memory Throughput (%) | 7.21 % | 7.27 % | +0.06 pp | 7.36 % | +0.15 pp | 16.29 % | +9.08 pp | 6.59 % | -0.62 pp | 7.16 % | -0.05 pp |
| DRAM Throughput (%) | 4.93 % | 5.03 % | +0.10 pp | 4.95 % | +0.02 pp | 2.86 % | -2.07 pp | 4.55 % | -0.38 pp | 4.94 % | +0.01 pp |
| Achieved Occupancy (%) | 53.15 % | 49.09 % | -4.06 pp | 55.00 % | +1.85 pp | 52.49 % | -0.66 pp | 56.54 % | +3.39 pp | 52.69 % | -0.46 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 26.00 register/thread | +10 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 68.16 % | 64.06 % | -4.10 pp | 63.42 % | -4.74 pp | 40.80 % | -27.36 pp | 51.96 % | -16.20 pp | 68.06 % | -0.10 pp |
| IPC Active | 1.15 inst/cycle | 1.29 inst/cycle | +0.14 | 1.44 inst/cycle | +0.29 | 2.28 inst/cycle | +1.13 | 1.85 inst/cycle | +0.70 | 1.13 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.583×)

Converting a single-tile-per-CTA Triton kernel to a persistent while-loop (grid=132 SMs, each CTA iterates over ~12 tiles) caused a severe regression: executed instructions exploded 4.4x (312K → 1.36M) and kernel duration nearly doubled (3.14µs → 5.82µs). Despite improved L1 hit rate (22% → 67%) and L2 hit rate (55% → 76%) from better cache reuse across persistent iterations, the while-loop branch overhead and serialized tile processing within each CTA dominated. The grid size remained at 1576 (the call site was not updated to 132), meaning the persistent loop ran on top of the original grid rather than replacing it, compounding the instruction count. Even if the grid had been corrected, the kernel body is too short (3.14µs, ~4650 cycles) for persistent scheduling to amortize its overhead — the per-tile loop control cost exceeds the latency-hiding benefit for such small, fast kernels.

> **Applicable when**: Never for this kernel. In general, persistent scheduling is applicable when: (1) the kernel duration is long (>50µs) so loop overhead is amortized, (2) there is significant inter-tile shared state (e.g., reduction accumulators, shared memory reuse) that benefits from keeping data in registers/L1 across tiles, and (3) the grid is large enough that wave fragmentation wastes >20% of SM time.

> **Avoid when**: The kernel is a short-duration elementwise pointwise kernel (<10µs). The while-loop branch overhead and serialized tile execution within each CTA will dominate over any wave-tail savings. Also avoid when the call site grid is not simultaneously reduced to match the persistent CTA count — leaving the original grid size causes each CTA to redundantly iterate over all tiles, multiplying instruction count by the tile-per-CTA count. Do not apply when the kernel has no inter-tile shared state (no shared memory, no register-resident accumulators) since there is no reuse benefit to justify the loop overhead.

#### ❌ Plan B: `persistent_tile_scheduling` (0.583×)

Converting a single-tile-per-CTA Triton kernel to a persistent while-loop (grid=132 SMs, each CTA iterates over ~12 tiles) caused a severe regression: executed instructions exploded 4.4x (312K → 1.36M) and kernel duration nearly doubled (3.14µs → 5.82µs). Despite improved L1 hit rate (22% → 67%) and L2 hit rate (55% → 76%) from better cache reuse across persistent iterations, the while-loop branch overhead and serialized tile processing within each CTA dominated. The grid size remained at 1576 (the call site was not updated to 132), meaning the persistent loop ran on top of the original grid rather than replacing it, compounding the instruction count. Even if the grid had been corrected, the kernel body is too short (3.14µs, ~4650 cycles) for persistent scheduling to amortize its overhead — the per-tile loop control cost exceeds the latency-hiding benefit for such small, fast kernels.

> **Applicable when**: Never for this kernel. In general, persistent scheduling is applicable when: (1) the kernel duration is long (>50µs) so loop overhead is amortized, (2) there is significant inter-tile shared state (e.g., reduction accumulators, shared memory reuse) that benefits from keeping data in registers/L1 across tiles, and (3) the grid is large enough that wave fragmentation wastes >20% of SM time.

> **Avoid when**: The kernel is a short-duration elementwise pointwise kernel (<10µs). The while-loop branch overhead and serialized tile execution within each CTA will dominate over any wave-tail savings. Also avoid when the call site grid is not simultaneously reduced to match the persistent CTA count — leaving the original grid size causes each CTA to redundantly iterate over all tiles, multiplying instruction count by the tile-per-CTA count. Do not apply when the kernel has no inter-tile shared state (no shared memory, no register-resident accumulators) since there is no reuse benefit to justify the loop overhead.

#### ❌ Plan C: `persistent_tile_scheduling` (0.583×)

Converting a single-tile-per-CTA Triton kernel to a persistent while-loop (grid=132 SMs, each CTA iterates over ~12 tiles) caused a severe regression: executed instructions exploded 4.4x (312K → 1.36M) and kernel duration nearly doubled (3.14µs → 5.82µs). Despite improved L1 hit rate (22% → 67%) and L2 hit rate (55% → 76%) from better cache reuse across persistent iterations, the while-loop branch overhead and serialized tile processing within each CTA dominated. The grid size remained at 1576 (the call site was not updated to 132), meaning the persistent loop ran on top of the original grid rather than replacing it, compounding the instruction count. Even if the grid had been corrected, the kernel body is too short (3.14µs, ~4650 cycles) for persistent scheduling to amortize its overhead — the per-tile loop control cost exceeds the latency-hiding benefit for such small, fast kernels.

> **Applicable when**: Never for this kernel. In general, persistent scheduling is applicable when: (1) the kernel duration is long (>50µs) so loop overhead is amortized, (2) there is significant inter-tile shared state (e.g., reduction accumulators, shared memory reuse) that benefits from keeping data in registers/L1 across tiles, and (3) the grid is large enough that wave fragmentation wastes >20% of SM time.

> **Avoid when**: The kernel is a short-duration elementwise pointwise kernel (<10µs). The while-loop branch overhead and serialized tile execution within each CTA will dominate over any wave-tail savings. Also avoid when the call site grid is not simultaneously reduced to match the persistent CTA count — leaving the original grid size causes each CTA to redundantly iterate over all tiles, multiplying instruction count by the tile-per-CTA count. Do not apply when the kernel has no inter-tile shared state (no shared memory, no register-resident accumulators) since there is no reuse benefit to justify the loop overhead.

#### ❌ Plan D: `persistent_tile_scheduling` (0.583×)

Converting a single-tile-per-CTA Triton kernel to a persistent while-loop (grid=132 SMs, each CTA iterates over ~12 tiles) caused a severe regression: executed instructions exploded 4.4x (312K → 1.36M) and kernel duration nearly doubled (3.14µs → 5.82µs). Despite improved L1 hit rate (22% → 67%) and L2 hit rate (55% → 76%) from better cache reuse across persistent iterations, the while-loop branch overhead and serialized tile processing within each CTA dominated. The grid size remained at 1576 (the call site was not updated to 132), meaning the persistent loop ran on top of the original grid rather than replacing it, compounding the instruction count. Even if the grid had been corrected, the kernel body is too short (3.14µs, ~4650 cycles) for persistent scheduling to amortize its overhead — the per-tile loop control cost exceeds the latency-hiding benefit for such small, fast kernels.

> **Applicable when**: Never for this kernel. In general, persistent scheduling is applicable when: (1) the kernel duration is long (>50µs) so loop overhead is amortized, (2) there is significant inter-tile shared state (e.g., reduction accumulators, shared memory reuse) that benefits from keeping data in registers/L1 across tiles, and (3) the grid is large enough that wave fragmentation wastes >20% of SM time.

> **Avoid when**: The kernel is a short-duration elementwise pointwise kernel (<10µs). The while-loop branch overhead and serialized tile execution within each CTA will dominate over any wave-tail savings. Also avoid when the call site grid is not simultaneously reduced to match the persistent CTA count — leaving the original grid size causes each CTA to redundantly iterate over all tiles, multiplying instruction count by the tile-per-CTA count. Do not apply when the kernel has no inter-tile shared state (no shared memory, no register-resident accumulators) since there is no reuse benefit to justify the loop overhead.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_streaming_eviction_for_l1tex_latency_hiding | 1.013× | 0.991× | ✓ |  |
| B | streaming_eviction_policy_for_large_strided_load | 1.027× | 1.000× | ✓ |  |
| C | fuse_fp32_add_to_fma_instruction | 0.964× | 0.890× | ✓ |  |
| D | revert_persistent_while_loop_to_single_tile_dispatch | 1.028× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | streaming_evictio... | Δ | fuse_fp32_add_to_... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3420 ns | +30 ns (+0.9%) | 3390 ns | +0 ns (+0.0%) | 3810 ns | +420 ns (+12.4%) | 3390 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 10.05 % | 13.20 % | +3.15 pp | 13.90 % | +3.85 pp | 20.18 % | +10.13 pp | 10.78 % | +0.73 pp |
| Memory Throughput (%) | 7.21 % | 7.09 % | -0.12 pp | 7.09 % | -0.12 pp | 6.40 % | -0.81 pp | 7.15 % | -0.06 pp |
| DRAM Throughput (%) | 4.93 % | 4.90 % | -0.03 pp | 4.90 % | -0.03 pp | 4.42 % | -0.51 pp | 4.94 % | +0.01 pp |
| Achieved Occupancy (%) | 53.15 % | 51.45 % | -1.70 pp | 51.76 % | -1.39 pp | 56.13 % | +2.98 pp | 53.03 % | -0.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 26.00 register/thread | +10 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 68.16 % | 61.53 % | -6.63 pp | 66.30 % | -1.86 pp | 53.39 % | -14.77 pp | 68.67 % | +0.51 pp |
| IPC Active | 1.15 inst/cycle | 1.47 inst/cycle | +0.32 | 1.48 inst/cycle | +0.33 | 1.79 inst/cycle | +0.64 | 1.16 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  3390 ns  (1.000×)

Final best: 4380 ns  (0.7740×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 201728
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x0 = (xindex % 512)
-    x1 = ((xindex // 512) % 197)
-    x2 = xindex // 100864
-    x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (512 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x4), tmp2, xmask)
-
-
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    num_sms = tl.num_programs(0)
+    pid = tl.program_id(0)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = (xindex % 512)
+        x1 = ((xindex // 512) % 197)
+        x2 = xindex // 100864
+        x4 = xindex
+        tmp0 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
+        tmp1 = tl.load(in_ptr1 + (512 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
+        tmp2 = tmp0 + tmp1
+        tl.store(out_ptr0 + (x4), tmp2, xmask)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
