# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 265570 ns — **Best latency**: 25730 ns (**10.3214×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`swap_tile_axes_r_outer_x_inner_for_coalesced_global_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_049/kernel_0_triton_per_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 31.03
- **Registers/thread**: 128
- **Duration**: 265570 ns
- **SM Throughput**: 59.98 % | **Memory**: 80.28 % | **DRAM**: 80.28 %
- **Achieved Occupancy**: 24.02 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 38.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_reorder_tile_dimensions | 1.986× | 2.046× | ✓ |  |
| B | add_autotune_reduce_register_pressure_via_xblock | 9.693× | 10.451× | ✓ | **✓ best** |
| C | reduce_xblock_to_lower_register_pressure_and_fix_bank_conflicts | 5.564× | 6.143× | ✓ |  |
| D | reduce_register_pressure_via_xblock_autotune | 9.580× | 10.653× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | add_autotune_redu... | Δ | reduce_xblock_to_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 265570 ns | 129790 ns | -135780 ns (-51.1%) | 25410 ns | -240160 ns (-90.4%) | 43230 ns | -222340 ns (-83.7%) | 24930 ns | -240640 ns (-90.6%) |
| SM Throughput (%) | 59.98 % | 69.01 % | +9.03 pp | 11.76 % | -48.22 pp | 56.27 % | -3.71 pp | 12.03 % | -47.95 pp |
| Memory Throughput (%) | 80.28 % | 80.08 % | -0.20 pp | 70.76 % | -9.52 pp | 69.39 % | -10.89 pp | 72.30 % | -7.98 pp |
| DRAM Throughput (%) | 80.28 % | 80.08 % | -0.20 pp | 3.69 % | -76.59 pp | 52.17 % | -28.11 pp | 3.73 % | -76.55 pp |
| Achieved Occupancy (%) | 24.02 % | 47.59 % | +23.57 pp | 44.15 % | +20.13 pp | 51.83 % | +27.81 pp | 43.76 % | +19.74 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 | 17.00 register/thread | -111 | 50.00 register/thread | -78 | 17.00 register/thread | -111 |
| Waves/SM | 31.03 | 15.52 | -15.51 | 3.88 | -27.15 | 6.90 | -24.13 | 3.88 | -27.15 |
| No Eligible (%) | 38.73 % | 30.63 % | -8.10 pp | 86.43 % | +47.70 pp | 39.11 % | +0.38 pp | 86.30 % | +47.57 pp |
| IPC Active | 2.46 inst/cycle | 2.78 inst/cycle | +0.32 | 0.54 inst/cycle | -1.92 | 2.43 inst/cycle | -0.03 | 0.54 inst/cycle | -1.92 |


### Key Learnings

#### ✅ Plan D: `reduce_register_pressure_via_xblock_autotune` (10.862×)

When a Triton kernel's @triton.autotune decorator varies XBLOCK but the grid is hardcoded as (cdiv(xnumel, FIXED_XBLOCK),) rather than a lambda using META['XBLOCK'], the autotuner is effectively a no-op — every trial runs with the same grid size regardless of the selected XBLOCK. Fixing the grid to lambda META: (cdiv(xnumel, META['XBLOCK']),) allows the autotuner to genuinely explore tile shapes, and adding num_stages=2 variants to the config list enables software pipelining of strided global loads. In this case, the fix raised theoretical occupancy from 50% to 100%, achieved occupancy from 44% to 88%, tripled compute throughput (11% → 30%), and increased memory throughput 4x (180 GB/s → 726 GB/s), while reducing warp cycles per issued instruction from 56 to 41.

> **Applicable when**: Triton kernel has @triton.autotune with XBLOCK configs but the grid is a static tuple using a hardcoded constant instead of META['XBLOCK']; theoretical occupancy is limited (≤50%) despite autotune being present; achieved occupancy is well below theoretical; NCU TheoreticalOccupancy or UncoalescedGlobalAccess rules fire; kernel writes in-place to an input pointer (needs restore_value); strided global loads suggest num_stages pipelining would help.

> **Avoid when**: Grid already uses a lambda with META['XBLOCK']; kernel does not use @triton.autotune; XBLOCK is not a tunable parameter; xnumel is not evenly divisible by all XBLOCK candidates (masking changes required in kernel body); the bottleneck is purely compute-bound with no occupancy or memory stall signals.

---

## Iteration 2

**Best so far**: 10.4514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_tile_axes_for_coalesced_global_loads_and_stores | 9.430× | 10.547× | ✓ |  |
| B | swap_tile_axes_r_outer_x_inner_for_coalesced_global_loads | 10.212× | 10.321× | ✓ | **✓ best** |
| C | swap_tile_axes_for_coalesced_global_loads | 5.565× | 6.027× | ✓ |  |
| D | autotune_very_small_xblock_to_eliminate_shared_memory_bank_conflicts | 9.428× | 10.480× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_tile_axes_fo... | Δ | swap_tile_axes_r_... | Δ | swap_tile_axes_fo... | Δ | autotune_very_sma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 265570 ns | 25180 ns | -240390 ns (-90.5%) | 25730 ns | -239840 ns (-90.3%) | 44060 ns | -221510 ns (-83.4%) | 25340 ns | -240230 ns (-90.5%) |
| SM Throughput (%) | 59.98 % | 11.06 % | -48.92 pp | 11.03 % | -48.95 pp | 56.75 % | -3.23 pp | 11.15 % | -48.83 pp |
| Memory Throughput (%) | 80.28 % | 71.19 % | -9.09 pp | 69.29 % | -10.99 pp | 70.32 % | -9.96 pp | 70.84 % | -9.44 pp |
| DRAM Throughput (%) | 80.28 % | 3.66 % | -76.62 pp | 3.58 % | -76.70 pp | 51.08 % | -29.20 pp | 3.65 % | -76.63 pp |
| Achieved Occupancy (%) | 24.02 % | 43.96 % | +19.94 pp | 43.92 % | +19.90 pp | 51.88 % | +27.86 pp | 43.85 % | +19.83 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 50.00 register/thread | -78 | 16.00 register/thread | -112 |
| Waves/SM | 31.03 | 3.88 | -27.15 | 3.88 | -27.15 | 6.90 | -24.13 | 3.88 | -27.15 |
| No Eligible (%) | 38.73 % | 87.38 % | +48.65 pp | 87.34 % | +48.61 pp | 39.33 % | +0.60 pp | 87.28 % | +48.55 pp |
| IPC Active | 2.46 inst/cycle | 0.50 inst/cycle | -1.96 | 0.50 inst/cycle | -1.96 | 2.42 inst/cycle | -0.04 | 0.50 inst/cycle | -1.96 |

---

## Iteration 3

**Best so far**: 10.4514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_to_autotune_configs | 9.266× | 10.547× | ✓ |  |
| B | increase_num_stages_to_hide_l1tex_load_latency | 9.434× | 10.398× | ✓ |  |
| C | add_num_stages_software_pipelining_to_autotune_configs | 10.170× | 10.559× | ✓ |  |
| D | fix_autotune_grid_lambda_and_add_num_stages_pipelining | 9.576× | 10.862× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | increase_num_stag... | Δ | add_num_stages_so... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 265570 ns | 25180 ns | -240390 ns (-90.5%) | 25540 ns | -240030 ns (-90.4%) | 25150 ns | -240420 ns (-90.5%) | 24450 ns | -241120 ns (-90.8%) |
| SM Throughput (%) | 59.98 % | 10.88 % | -49.10 pp | 11.03 % | -48.95 pp | 10.94 % | -49.04 pp | 30.15 % | -29.83 pp |
| Memory Throughput (%) | 80.28 % | 71.27 % | -9.01 pp | 70.18 % | -10.10 pp | 71.15 % | -9.13 pp | 78.53 % | -1.75 pp |
| DRAM Throughput (%) | 80.28 % | 3.67 % | -76.61 pp | 3.61 % | -76.67 pp | 3.67 % | -76.61 pp | 14.79 % | -65.49 pp |
| Achieved Occupancy (%) | 24.02 % | 43.89 % | +19.87 pp | 44.06 % | +20.04 pp | 43.94 % | +19.92 pp | 87.60 % | +63.58 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 22.00 register/thread | -106 |
| Waves/SM | 31.03 | 3.88 | -27.15 | 3.88 | -27.15 | 3.88 | -27.15 | 3.88 | -27.15 |
| No Eligible (%) | 38.73 % | 87.52 % | +48.79 pp | 87.37 % | +48.64 pp | 87.13 % | +48.40 pp | 65.94 % | +27.21 pp |
| IPC Active | 2.46 inst/cycle | 0.50 inst/cycle | -1.96 | 0.50 inst/cycle | -1.96 | 0.50 inst/cycle | -1.96 | 1.35 inst/cycle | -1.11 |


### Key Learnings

#### ✅ Plan A: `add_num_stages_software_pipelining_to_autotune_configs` (10.559×)

When a Triton persistent-reduction kernel is bottlenecked by L1TEX and MIO scoreboard stalls (87% no-eligible scheduler cycles, ~56 warp cycles per issued instruction) with a 95% L2 hit rate and very low DRAM throughput (~3.7%), adding num_stages=2 and num_stages=3 variants to the autotune config list does not improve performance. The before and after profiling metrics are virtually identical (25.18 µs vs 25.15 µs, all stall metrics unchanged), indicating that Triton's software pipeline (cp.async) does not effectively hide L2→L1 latency in this persistent-reduction kernel pattern where there is no tiled inner loop for the compiler to pipeline across. The 10.56x reported speedup reflects the cumulative gain from prior iterations (register pressure fix), not from this num_stages change.

> **Applicable when**: Never effective for persistent-reduction kernels with no tiled inner loop; only applicable when there is a loop body with tl.load calls that can be prefetched across loop iterations (e.g., tiled GEMM inner loops).

> **Avoid when**: Kernel is a persistent-reduction (single-pass over reduction dimension with no explicit loop); bottleneck is L1TEX/MIO scoreboard stalls with high L2 hit rate (>90%) and low DRAM throughput (<10%); no-eligible scheduler percentage remains ~87% after adding num_stages. In these cases num_stages has no effect because Triton cannot pipeline across the reduction dimension boundary.

#### ✅ Plan C: `add_num_stages_software_pipelining_to_autotune_configs` (10.559×)

When a Triton persistent-reduction kernel is bottlenecked by L1TEX and MIO scoreboard stalls (87% no-eligible scheduler cycles, ~56 warp cycles per issued instruction) with a 95% L2 hit rate and very low DRAM throughput (~3.7%), adding num_stages=2 and num_stages=3 variants to the autotune config list does not improve performance. The before and after profiling metrics are virtually identical (25.18 µs vs 25.15 µs, all stall metrics unchanged), indicating that Triton's software pipeline (cp.async) does not effectively hide L2→L1 latency in this persistent-reduction kernel pattern where there is no tiled inner loop for the compiler to pipeline across. The 10.56x reported speedup reflects the cumulative gain from prior iterations (register pressure fix), not from this num_stages change.

> **Applicable when**: Never effective for persistent-reduction kernels with no tiled inner loop; only applicable when there is a loop body with tl.load calls that can be prefetched across loop iterations (e.g., tiled GEMM inner loops).

> **Avoid when**: Kernel is a persistent-reduction (single-pass over reduction dimension with no explicit loop); bottleneck is L1TEX/MIO scoreboard stalls with high L2 hit rate (>90%) and low DRAM throughput (<10%); no-eligible scheduler percentage remains ~87% after adding num_stages. In these cases num_stages has no effect because Triton cannot pipeline across the reduction dimension boundary.

---

## Iteration 4

**Best so far**: 10.4514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_num_warps_autotune_to_eliminate_shared_memory_bank_conflicts | 9.780× | 10.709× | ✓ |  |
| B | evict_first_on_strided_global_load_to_reduce_l1tex_thrashing | 9.314× | 10.695× | ✓ |  |
| C | replace_triton_helpers_max2_with_tl_max_to_eliminate_mio_shared_memory_stalls | 9.409× | 10.585× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_mio_scoreboard_stalls | 9.347× | 10.559× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_num_warps_... | Δ | evict_first_on_st... | Δ | replace_triton_he... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 265570 ns | 24800 ns | -240770 ns (-90.7%) | 24830 ns | -240740 ns (-90.7%) | 25090 ns | -240480 ns (-90.6%) | 25150 ns | -240420 ns (-90.5%) |
| SM Throughput (%) | 59.98 % | 10.50 % | -49.48 pp | 11.88 % | -48.10 pp | 9.73 % | -50.25 pp | 10.97 % | -49.01 pp |
| Memory Throughput (%) | 80.28 % | 72.34 % | -7.94 pp | 72.38 % | -7.90 pp | 71.71 % | -8.57 pp | 71.27 % | -9.01 pp |
| DRAM Throughput (%) | 80.28 % | 3.71 % | -76.57 pp | 3.78 % | -76.50 pp | 3.67 % | -76.61 pp | 3.71 % | -76.57 pp |
| Achieved Occupancy (%) | 24.02 % | 43.78 % | +19.76 pp | 44.18 % | +20.16 pp | 43.76 % | +19.74 pp | 43.94 % | +19.92 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 31.03 | 3.88 | -27.15 | 3.88 | -27.15 | 3.88 | -27.15 | 3.88 | -27.15 |
| No Eligible (%) | 38.73 % | 87.65 % | +48.92 pp | 86.32 % | +47.59 pp | 88.71 % | +49.98 pp | 87.26 % | +48.53 pp |
| IPC Active | 2.46 inst/cycle | 0.48 inst/cycle | -1.98 | 0.54 inst/cycle | -1.92 | 0.45 inst/cycle | -2.01 | 0.50 inst/cycle | -1.96 |


### Key Learnings

#### ❌ Plan D: `increase_num_warps_to_hide_l1tex_mio_scoreboard_stalls` (10.559×)

When a Triton persistent-reduction kernel is bottlenecked by L1TEX and MIO scoreboard stalls (87% no-eligible scheduler cycles, ~56 warp cycles per issued instruction, 95% L2 hit rate, <4% DRAM throughput), expanding the autotune config space to include num_warps=16 (512 threads/block) does not improve performance. All profiling metrics remain statistically identical before and after (duration ~25.15 µs, eligible warps 0.16, no-eligible ~87%, warp cycles/instruction ~56), indicating the autotuner still selects a low num_warps config (block_size=32, 1 warp/block) as optimal. The root cause is that the scoreboard stalls are driven by uncoalesced global access patterns (82% excessive sectors, 4-7 bytes/sector utilized) that cause long L2 latency chains — adding more warps per block does not reduce per-warp stall cycles when the stall source is structural (strided memory access), not warp-count-limited latency hiding.

> **Applicable when**: Never effective for this pattern: persistent-reduction kernel with 82%+ uncoalesced global sectors, 95% L2 hit rate, <4% DRAM throughput, and scoreboard stalls driven by strided memory access. The stall source is structural (address formula causes non-coalesced L2 transactions), not warp-count-limited.

> **Avoid when**: Kernel has high DRAM throughput (>30%) and low L2 hit rate (<80%), indicating the bottleneck is bandwidth-limited latency where more warps could help; kernel already has coalesced global access (bytes/sector >20); the no-eligible rate is driven by compute dependencies rather than memory scoreboard stalls; the autotuner is already exploring num_warps=16 configs.

---

## Cumulative Progress

```
Baseline:  265570 ns  (1.000×)
Iter 1     25410 ns  (10.451×)  — add_autotune_reduce_register_pressure_via_xblock

Final best: 25730 ns  (10.3214×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 4}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 8}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 2097152
@@ -13,34 +34,32 @@

     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_index = tl.arange(0, R0_BLOCK)[None, :]
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xmask = tl.full([1, XBLOCK], True, tl.int1)
+    r0_index = tl.arange(0, R0_BLOCK)[:, None]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = tl.full([R0_BLOCK, 1], True, tl.int1)
     roffset = r0_offset
     rindex = r0_index
     r0_2 = r0_index
     x0 = (xindex % 131072)
     x1 = xindex // 131072
     x3 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), None)
-    tmp1 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), mask=r0_mask & xmask)
+    tmp1 = tl.load(in_ptr0 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
-    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
-    tmp7 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
+    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, XBLOCK])
+    tmp5 = tl.broadcast_to(tmp3, [R0_BLOCK, XBLOCK])
+    tmp7 = triton_helpers.max2(tmp5, 0)[None, :].to(tl.float32)
     tmp8 = tmp3 - tmp7
     tmp9 = tl.exp(tmp8)
-    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
-    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
+    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK, XBLOCK])
+    tmp12 = tl.sum(tmp10, 0)[None, :].to(tl.float32)
     tmp13 = tmp2 - tmp7
     tmp14 = tl.exp(tmp13)
     tmp15 = (tmp14 / tmp12)
     tmp16 = tl.sigmoid(tmp15)
-    tl.store(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), tmp16, None)
-
-
+    tl.store(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), tmp16, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -48,7 +67,7 @@

     xnumel = 2097152
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0[grid](arg_0, arg_1, 2097152, 64, XBLOCK=128)
+    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0[grid](arg_0, arg_1, 2097152, 64)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
