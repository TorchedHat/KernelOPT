# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8190 ns — **Best latency**: 7140 ns (**1.1471×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_evict_last_to_all_stencil_loads_for_l1_reuse`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_31_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 22
- **Duration**: 8190 ns
- **SM Throughput**: 52.37 % | **Memory**: 26.51 % | **DRAM**: 9.39 %
- **Achieved Occupancy**: 78.18 %
- **IPC Active**: 2.70 inst/cycle | **No Eligible**: 32.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.973× | 0.486× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.003× | 0.984× | ✓ |  |
| C | increase_xblock_with_autotune_to_eliminate_partial_wave_and_vectorize_loads | 0.994× | 0.951× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave | 1.006× | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | autotune_xblock_e... | Δ | increase_xblock_w... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 16860 ns | +8670 ns (+105.9%) | 8320 ns | +130 ns (+1.6%) | 8610 ns | +420 ns (+5.1%) | 8260 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 52.37 % | 71.43 % | +19.06 pp | 51.91 % | -0.46 pp | 57.24 % | +4.87 pp | 52.88 % | +0.51 pp |
| Memory Throughput (%) | 26.51 % | 25.97 % | -0.54 pp | 25.37 % | -1.14 pp | 30.93 % | +4.42 pp | 25.67 % | -0.84 pp |
| DRAM Throughput (%) | 9.39 % | 4.55 % | -4.84 pp | 9.27 % | -0.12 pp | 8.94 % | -0.45 pp | 9.35 % | -0.04 pp |
| Achieved Occupancy (%) | 78.18 % | 85.78 % | +7.60 pp | 75.63 % | -2.55 pp | 77.80 % | -0.38 pp | 78.17 % | -0.01 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.48 | 6.96 | +3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 32.85 % | 20.44 % | -12.41 pp | 31.84 % | -1.01 pp | 26.82 % | -6.03 pp | 34.87 % | +2.02 pp |
| IPC Active | 2.70 inst/cycle | 3.18 inst/cycle | +0.48 | 2.64 inst/cycle | -0.06 | 2.91 inst/cycle | +0.21 | 2.67 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_alu_latency | 1.002× | 0.600× | ✓ |  |
| C | reduce_alu_pressure_via_num_warps_reduction_and_index_hoisting | 1.091× | 1.137× | ✓ | **✓ best** |
| D | simplify_index_arithmetic_to_reduce_alu_pressure | 1.056× | 1.098× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | reduce_alu_pressu... | Δ | simplify_index_ar... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 13660 ns | +5470 ns (+66.8%) | 7200 ns | -990 ns (-12.1%) | 7460 ns | -730 ns (-8.9%) |
| SM Throughput (%) | 52.37 % | 64.70 % | +12.33 pp | 32.10 % | -20.27 pp | 48.49 % | -3.88 pp |
| Memory Throughput (%) | 26.51 % | 25.01 % | -1.50 pp | 28.15 % | +1.64 pp | 30.16 % | +3.65 pp |
| DRAM Throughput (%) | 9.39 % | 5.62 % | -3.77 pp | 10.73 % | +1.34 pp | 10.32 % | +0.93 pp |
| Achieved Occupancy (%) | 78.18 % | 83.22 % | +5.04 pp | 56.67 % | -21.51 pp | 73.78 % | -4.40 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 32.00 register/thread | +10 | 26.00 register/thread | +4 |
| Waves/SM | 3.48 | 6.96 | +3.48 | 1.74 | -1.74 | 3.48 | +0.00 |
| No Eligible (%) | 32.85 % | 27.95 % | -4.90 pp | 36.24 % | +3.39 pp | 32.61 % | -0.24 pp |
| IPC Active | 2.70 inst/cycle | 2.88 inst/cycle | +0.18 | 2.42 inst/cycle | -0.28 | 2.76 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_to_hide_alu_latency` (0.600×)

Increasing num_warps from 4 to 8 at XBLOCK=128 doubled the block size from 128 to 256 threads, which doubled the executed instruction count (3.1M → 6.3M) and increased kernel duration from 8.26 µs to 13.66 µs (0.6x speedup). Although the No-Eligible scheduler stall rate improved modestly (34.87% → 27.95%) and L1 hit rate jumped dramatically (1% → 47.6%), the doubled instruction volume overwhelmed any latency-hiding benefit. In Triton, num_warps directly sets threads_per_block = num_warps × 32, so doubling num_warps at a fixed XBLOCK means each thread processes fewer elements but the block is twice as large, doubling total work dispatched per grid.

> **Applicable when**: Never: this strategy caused regression. Avoid increasing num_warps beyond the natural XBLOCK/32 ratio in Triton pointwise kernels.

> **Avoid when**: When XBLOCK is fixed and num_warps > XBLOCK/32: in Triton, threads_per_block = num_warps × 32, so if num_warps × 32 > XBLOCK, each thread still processes exactly one element (XBLOCK elements / threads_per_block threads), but the block is larger, doubling total instructions dispatched. Do not apply when the kernel is already compute-bound (ALU >50%) and instruction count is the limiting factor. Do not apply when the scheduler No-Eligible rate is below 40% — the latency-hiding benefit is insufficient to offset the instruction volume increase.

#### ✅ Plan D: `simplify_index_arithmetic_to_reduce_alu_pressure` (1.098×)

When a stencil or pooling kernel computes per-neighbor boundary masks by combining multiple boolean expressions redundantly (e.g., recomputing row/col validity for each of 9 neighbors independently), replacing them with 6 precomputed boundary boolean vectors (row_m1_ok, row_ok, row_p1_ok, col_m1_ok, col_ok, col_p1_ok) and reusing them across all loads reduces executed instruction count (3.1M → 2.7M) and warp cycles per issued instruction (18.3 → 15.9 cycles), yielding ~10% speedup. The reduction in ALU pressure also slightly increases effective memory throughput (456 → 505 GB/s) despite a small occupancy drop from increased register usage (22 → 26 regs/thread). This pattern is applicable whenever multiple loads share the same boundary condition logic that is currently recomputed per-load.

> **Applicable when**: Kernel performs a stencil or pooling operation with multiple neighbor loads sharing boundary conditions; profiling shows HighPipeUtilization (ALU >60%) with moderate warp cycles per issued instruction (>15); executed instruction count is high relative to the number of loads; boundary masks are recomputed independently per load rather than shared.

> **Avoid when**: When the kernel is already memory-bound (memory throughput >70% of peak) rather than ALU-bound — reducing ALU ops will not help. Also avoid when the additional registers from precomputed mask variables would push register pressure high enough to significantly reduce block-limit-registers and occupancy (watch for >28 regs/thread with small block sizes).

---

## Iteration 3

**Best so far**: 1.1375× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_grid_lambda_and_add_num_warps1_config | 1.085× | 1.137× | ✓ |  |
| B | add_evict_last_cache_hint_to_neighbor_loads | 1.043× | 1.037× | ✓ |  |
| C | hoist_boundary_masks_and_balanced_tree_reduction | 1.082× | 0.684× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_grid_l... | Δ | add_evict_last_ca... | Δ | hoist_boundary_ma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 7200 ns | -990 ns (-12.1%) | 7900 ns | -290 ns (-3.5%) | 11970 ns | +3780 ns (+46.2%) |
| SM Throughput (%) | 52.37 % | 32.32 % | -20.05 pp | 56.51 % | +4.14 pp | 59.45 % | +7.08 pp |
| Memory Throughput (%) | 26.51 % | 28.01 % | +1.50 pp | 27.62 % | +1.11 pp | 28.78 % | +2.27 pp |
| DRAM Throughput (%) | 9.39 % | 10.67 % | +1.28 pp | 9.76 % | +0.37 pp | 6.43 % | -2.96 pp |
| Achieved Occupancy (%) | 78.18 % | 56.30 % | -21.88 pp | 78.08 % | -0.10 pp | 81.66 % | +3.48 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 24.00 register/thread | +2 | 26.00 register/thread | +4 |
| Waves/SM | 3.48 | 1.74 | -1.74 | 3.48 | +0.00 | 6.96 | +3.48 |
| No Eligible (%) | 32.85 % | 36.85 % | +4.00 pp | 21.32 % | -11.53 pp | 27.41 % | -5.44 pp |
| IPC Active | 2.70 inst/cycle | 2.47 inst/cycle | -0.23 | 3.08 inst/cycle | +0.38 | 2.88 inst/cycle | +0.18 |


### Key Learnings

#### ❌ Plan C: `hoist_boundary_masks_and_balanced_tree_reduction` (0.684×)

Applying boundary mask hoisting and balanced tree reduction alongside a narrowed autotune config set (only XBLOCK=128 with num_warps=8/16/4) caused the autotuner to select num_warps=8, yielding block_size=256 — quadrupling executed instructions (1.78M → 5.47M) and increasing duration from 7.2 µs to 11.97 µs. The mask hoisting and tree reduction changes are semantically correct and beneficial in isolation (confirmed by exp_002_simplify_index_arithmetic_reduce_alu at +9.8%), but the regression was entirely caused by the autotune config change: removing the previously winning num_warps=2 config forced selection of a higher-warp config that inflated instruction count. When bundling kernel body improvements with autotune config changes, the config change can dominate and mask or reverse the body improvement.

> **Applicable when**: Never apply this specific combination. The lesson is: when a previous iteration established a winning autotune config (e.g., num_warps=2), do not remove that config from the search space in subsequent iterations. Kernel body improvements should be applied while preserving the winning config set.

> **Avoid when**: Do not remove a previously winning autotune config (e.g., num_warps=2) when applying kernel body changes. Removing the winning config forces the autotuner to select a suboptimal config, which can dominate and reverse any body-level improvement. Specifically: do not narrow autotune configs to only high-num_warps options (≥4) for this kernel, as num_warps=8 at XBLOCK=128 produces block_size=256 and 3x more instructions than num_warps=2.

---

## Iteration 4

**Best so far**: 1.1375× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_block_size_to_cut_instruction_count_and_cpi_stalls | 1.082× | 1.147× | ✓ |  |
| C | reduce_xblock_and_num_warps_to_minimize_alu_instruction_count | 1.107× | 1.153× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_block_size... | Δ | reduce_xblock_and... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 7140 ns | -1050 ns (-12.8%) | 7100 ns | -1090 ns (-13.3%) |
| SM Throughput (%) | 52.37 % | 31.43 % | -20.94 pp | 24.19 % | -28.18 pp |
| Memory Throughput (%) | 26.51 % | 15.46 % | -11.05 pp | 15.49 % | -11.02 pp |
| DRAM Throughput (%) | 9.39 % | 5.40 % | -3.99 pp | 5.44 % | -3.95 pp |
| Achieved Occupancy (%) | 78.18 % | 49.12 % | -29.06 pp | 42.87 % | -35.31 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 26.00 register/thread | +4 |
| Waves/SM | 3.48 | 1.74 | -1.74 | 1.74 | -1.74 |
| No Eligible (%) | 32.85 % | 36.12 % | +3.27 pp | 47.77 % | +14.92 pp |
| IPC Active | 2.70 inst/cycle | 2.44 inst/cycle | -0.26 | 1.99 inst/cycle | -0.71 |

---

## Iteration 5

**Best so far**: 1.1535× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_last_to_all_stencil_loads_for_l1_reuse | 1.100× | 1.147× | ✓ |  |
| B | fix_grid_lambda_to_cover_full_xnumel_with_xblock64 | 1.110× | 1.168× | ✓ |  |
| C | fix_grid_lambda_and_add_num_warps1_config_to_eliminate_partial_wave | 1.125× | 1.147× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_evict_last_to... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 7140 ns | -1050 ns (-12.8%) | 7010 ns | -1180 ns (-14.4%) | 7140 ns | -1050 ns (-12.8%) |
| SM Throughput (%) | 52.37 % | 30.92 % | -21.45 pp | 12.11 % | -40.26 pp | 17.56 % | -34.81 pp |
| Memory Throughput (%) | 26.51 % | 15.27 % | -11.24 pp | 9.75 % | -16.76 pp | 15.44 % | -11.07 pp |
| DRAM Throughput (%) | 9.39 % | 5.40 % | -3.99 pp | 2.83 % | -6.56 pp | 5.41 % | -3.98 pp |
| Achieved Occupancy (%) | 78.18 % | 51.66 % | -26.52 pp | 17.06 % | -61.12 pp | 19.21 % | -58.97 pp |
| Registers/Thread | 22.00 register/thread | 26.00 register/thread | +4 | 28.00 register/thread | +6 | 36.00 register/thread | +14 |
| Waves/SM | 3.48 | 1.74 | -1.74 | 1.74 | -1.74 | 1.74 | -1.74 |
| No Eligible (%) | 32.85 % | 37.84 % | +4.99 pp | 71.69 % | +38.84 pp | 62.01 % | +29.16 pp |
| IPC Active | 2.70 inst/cycle | 2.29 inst/cycle | -0.41 | 0.90 inst/cycle | -1.80 | 1.43 inst/cycle | -1.27 |

---

## Cumulative Progress

```
Baseline:  8190 ns  (1.000×)
Iter 2     7200 ns  (1.137×)  — reduce_alu_pressure_via_num_warps_reduction_and_index_hoisting
Iter 4     7100 ns  (1.153×)  — reduce_xblock_and_num_warps_to_minimize_alu_instruction_count

Final best: 7140 ns  (1.1471×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64},  num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 940800
@@ -14,68 +24,37 @@

     x2 = ((xindex // 6720) % 14)
     x1 = ((xindex // 480) % 14)
     x6 = xindex
-    tmp0 = (-1) + x2
-    tmp1 = tl.full([1], 0, tl.int64)
-    tmp2 = tmp0 >= tmp1
-    tmp3 = tl.full([1], 14, tl.int64)
-    tmp4 = tmp0 < tmp3
-    tmp5 = tmp2 & tmp4
-    tmp6 = (-1) + x1
-    tmp7 = tmp6 >= tmp1
-    tmp8 = tmp6 < tmp3
-    tmp9 = tmp7 & tmp8
-    tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-7200) + x6), tmp10 & xmask, other=float("-inf"))
-    tmp12 = x1
-    tmp13 = tmp12 >= tmp1
-    tmp14 = tmp12 < tmp3
-    tmp15 = tmp13 & tmp14
-    tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-6720) + x6), tmp16 & xmask, other=float("-inf"))
-    tmp18 = tl.maximum(tmp11, tmp17)
-    tmp19 = 1 + x1
-    tmp20 = tmp19 >= tmp1
-    tmp21 = tmp19 < tmp3
-    tmp22 = tmp20 & tmp21
-    tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-6240) + x6), tmp23 & xmask, other=float("-inf"))
-    tmp25 = tl.maximum(tmp18, tmp24)
-    tmp26 = x2
-    tmp27 = tmp26 >= tmp1
-    tmp28 = tmp26 < tmp3
-    tmp29 = tmp27 & tmp28
-    tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-480) + x6), tmp30 & xmask, other=float("-inf"))
-    tmp32 = tl.maximum(tmp25, tmp31)
-    tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
-    tmp35 = tl.maximum(tmp32, tmp34)
-    tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (480 + x6), tmp36 & xmask, other=float("-inf"))
-    tmp38 = tl.maximum(tmp35, tmp37)
-    tmp39 = 1 + x2
-    tmp40 = tmp39 >= tmp1
-    tmp41 = tmp39 < tmp3
-    tmp42 = tmp40 & tmp41
-    tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (6240 + x6), tmp43 & xmask, other=float("-inf"))
-    tmp45 = tl.maximum(tmp38, tmp44)
-    tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (6720 + x6), tmp46 & xmask, other=float("-inf"))
-    tmp48 = tl.maximum(tmp45, tmp47)
-    tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (7200 + x6), tmp49 & xmask, other=float("-inf"))
-    tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, xmask)
 
+    # Precompute 6 boundary masks once and reuse across all 9 loads
+    row_m1_ok = x2 >= 1        # row-1 is valid (x2 >= 1 means x2-1 >= 0)
+    row_p1_ok = x2 < 13        # row+1 is valid (x2+1 < 14 means x2 < 13)
+    col_m1_ok = x1 >= 1        # col-1 is valid
+    col_p1_ok = x1 < 13        # col+1 is valid
 
+    # 9 neighbor loads with eviction_policy='evict_last' to improve L1 reuse
+    tmp0 = tl.load(in_ptr0 + (x6 - 7200), mask=xmask & row_m1_ok & col_m1_ok, other=float("-inf"), eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (x6 - 6720), mask=xmask & row_m1_ok,             other=float("-inf"), eviction_policy='evict_last')
+    tmp2 = tl.load(in_ptr0 + (x6 - 6240), mask=xmask & row_m1_ok & col_p1_ok, other=float("-inf"), eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr0 + (x6 - 480),  mask=xmask & col_m1_ok,             other=float("-inf"), eviction_policy='evict_last')
+    tmp4 = tl.load(in_ptr0 + (x6),        mask=xmask,                          other=float("-inf"), eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr0 + (x6 + 480),  mask=xmask & col_p1_ok,             other=float("-inf"), eviction_policy='evict_last')
+    tmp6 = tl.load(in_ptr0 + (x6 + 6240), mask=xmask & row_p1_ok & col_m1_ok, other=float("-inf"), eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr0 + (x6 + 6720), mask=xmask & row_p1_ok,             other=float("-inf"), eviction_policy='evict_last')
+    tmp8 = tl.load(in_ptr0 + (x6 + 7200), mask=xmask & row_p1_ok & col_p1_ok, other=float("-inf"), eviction_policy='evict_last')
+
+    # Balanced tree reduction to minimize dependency chain depth
+    result = tl.maximum(
+        tl.maximum(tl.maximum(tmp0, tmp1), tl.maximum(tmp2, tmp3)),
+        tl.maximum(tl.maximum(tmp4, tmp5), tl.maximum(tmp6, tl.maximum(tmp7, tmp8)))
+    )
+    tl.store(out_ptr0 + (x6), result, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 940800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_31[grid](arg_0, arg_1, 940800, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_31[grid](arg_0, arg_1, 940800)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
