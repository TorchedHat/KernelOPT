# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 7460 ns (**0.7895×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.90 % | **Memory**: 41.98 % | **DRAM**: 20.92 %
- **Achieved Occupancy**: 54.96 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.36 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.335× | 1.406× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_wave_saturation | 1.264× | 1.396× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4190 ns | -1700 ns (-28.9%) | 4220 ns | -1670 ns (-28.4%) |
| SM Throughput (%) | 15.90 % | 17.26 % | +1.36 pp | 19.77 % | +3.87 pp |
| Memory Throughput (%) | 41.98 % | 33.19 % | -8.79 pp | 22.85 % | -19.13 pp |
| DRAM Throughput (%) | 20.92 % | 15.76 % | -5.16 pp | 3.91 % | -17.01 pp |
| Achieved Occupancy (%) | 54.96 % | 55.34 % | +0.38 pp | 51.93 % | -3.03 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.36 % | 65.03 % | -6.33 pp | 55.19 % | -16.17 pp |
| IPC Active | 1.15 inst/cycle | 1.32 inst/cycle | +0.17 | 1.60 inst/cycle | +0.45 |


### Key Learnings

#### ❌ Plan D: `autotune_tile_sizes_for_wave_saturation` (0.267×)

Replacing a diverse autotune config set (XBLOCK 32–1024, YBLOCK 1–32) with a YBLOCK=32-only config set caused a 3.75x slowdown. The winning config allocated ~32 KB of dynamic shared memory per block, collapsing Block Limit Shared Mem from 33 to 6 blocks/SM and dropping theoretical occupancy from 100% to 37.5%. Although waves_per_SM increased from 0.74 to 1.98 (more grid coverage), the occupancy collapse and 5.2x instruction count increase (610K → 3.16M) dominated, extending duration from 4.03 µs to 22.05 µs. Eligible warps per scheduler fell from 1.94 to 0.65, worsening the IssueSlotUtilization bottleneck. Restricting autotune configs to a narrow tile family can cause the autotuner to select a config that is technically valid but pathologically bad for shared memory occupancy.

> **Applicable when**: Never apply this pattern — restricting Triton autotune configs to large YBLOCK values (e.g., YBLOCK=16/32 only) when the kernel allocates dynamic shared memory proportional to tile size causes the autotuner to select configs that collapse shared-memory occupancy.

> **Avoid when**: Always include small-YBLOCK configs (YBLOCK=1, 2, 4, 8) in the autotune search space alongside large-YBLOCK configs. When Block Limit Shared Mem drops below ~16 blocks/SM, theoretical occupancy falls below 50% and the kernel becomes severely latency-bound. Do not assume that larger YBLOCK always improves performance — the shared memory cost can dominate.

---

## Iteration 2

**Best so far**: 1.4057× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.360× | 1.426× | ✓ |  |
| B | fix_autotune_grid_lambda_and_add_num_stages_pipelining | 1.338× | 1.451× | ✓ |  |
| C | vectorized_loads_with_lambda_grid_and_yblock1_configs | 1.403× | 0.789× | ✓ | **✓ best** |
| D | fix_autotune_lambda_grid_for_yblock_alignment | 1.265× | 1.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ | fix_autotune_lamb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4130 ns | -1760 ns (-29.9%) | 4060 ns | -1830 ns (-31.1%) | 7460 ns | +1570 ns (+26.7%) | 4030 ns | -1860 ns (-31.6%) |
| SM Throughput (%) | 15.90 % | 16.84 % | +0.94 pp | 17.00 % | +1.10 pp | 30.73 % | +14.83 pp | 19.56 % | +3.66 pp |
| Memory Throughput (%) | 41.98 % | 33.48 % | -8.50 pp | 33.96 % | -8.02 pp | 48.62 % | +6.64 pp | 22.60 % | -19.38 pp |
| DRAM Throughput (%) | 20.92 % | 15.97 % | -4.95 pp | 16.15 % | -4.77 pp | 2.22 % | -18.70 pp | 4.11 % | -16.81 pp |
| Achieved Occupancy (%) | 54.96 % | 54.82 % | -0.14 pp | 54.63 % | -0.33 pp | 55.82 % | +0.86 pp | 49.18 % | -5.78 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 21.00 register/thread | -5 | 40.00 register/thread | +14 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 | 0.74 | +0.00 |
| No Eligible (%) | 71.36 % | 65.20 % | -6.16 pp | 65.07 % | -6.29 pp | 50.61 % | -20.75 pp | 52.14 % | -19.22 pp |
| IPC Active | 1.15 inst/cycle | 1.32 inst/cycle | +0.17 | 1.34 inst/cycle | +0.19 | 1.95 inst/cycle | +0.80 | 1.81 inst/cycle | +0.66 |

---

## Iteration 3

**Best so far**: 1.4057× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | store_eviction_policy_streaming_with_yblock32_saturation | 1.371× | 0.818× | ✓ |  |
| B | yblock32_covers_full_y_dimension_with_lambda_grid | 1.012× | 0.139× | ✓ |  |
| C | yblock32_covers_full_y_dimension_with_evict_first_streaming | 1.109× | 0.271× | ✓ |  |
| D | restore_low_register_pressure_configs_with_yblock32_coverage | 1.161× | 0.267× | ✓ |  |


### Metric Comparison

| Metric | Baseline | store_eviction_po... | Δ | yblock32_covers_f... | Δ | yblock32_covers_f... | Δ | restore_low_regis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 7200 ns | +1310 ns (+22.2%) | 42370 ns | +36480 ns (+619.4%) | 21730 ns | +15840 ns (+268.9%) | 22050 ns | +16160 ns (+274.4%) |
| SM Throughput (%) | 15.90 % | 30.81 % | +14.91 pp | 38.21 % | +22.31 pp | 36.64 % | +20.74 pp | 22.39 % | +6.49 pp |
| Memory Throughput (%) | 41.98 % | 48.09 % | +6.11 pp | 67.82 % | +25.84 pp | 50.66 % | +8.68 pp | 51.56 % | +9.58 pp |
| DRAM Throughput (%) | 20.92 % | 2.29 % | -18.63 pp | 2.90 % | -18.02 pp | 5.66 % | -15.26 pp | 5.58 % | -15.34 pp |
| Achieved Occupancy (%) | 54.96 % | 55.83 % | +0.87 pp | 12.38 % | -42.58 pp | 66.03 % | +11.07 pp | 33.60 % | -21.36 pp |
| Registers/Thread | 26.00 register/thread | 40.00 register/thread | +14 | 38.00 register/thread | +12 | 32.00 register/thread | +6 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 11.88 | +11.14 | 1.98 | +1.24 | 1.98 | +1.24 |
| No Eligible (%) | 71.36 % | 50.07 % | -21.29 pp | 56.23 % | -15.13 pp | 55.54 % | -15.82 pp | 64.55 % | -6.81 pp |
| IPC Active | 1.15 inst/cycle | 1.98 inst/cycle | +0.83 | 1.73 inst/cycle | +0.58 | 1.76 inst/cycle | +0.61 | 1.42 inst/cycle | +0.27 |


### Key Learnings

#### ❌ Plan B: `yblock32_covers_full_y_dimension_with_lambda_grid` (0.139×)

Adding YBLOCK=32 configs with large XBLOCK values (128–2048) to a Triton autotune list caused the winning config to allocate 131 KB of dynamic shared memory per block, collapsing Block Limit Shared Mem from 33 to 1 block/SM and theoretical occupancy from 100% to 12.5%. Although warp cycles per issued instruction improved (16.4→4.6) and waves_per_SM increased (0.74→11.88), the catastrophic occupancy collapse (49%→12%) and 21x instruction count increase (610K→12.8M) caused a 7x slowdown (4.03→42.37 µs). The root cause is that large XBLOCK×YBLOCK tile products require proportionally large shared memory allocations that exceed per-SM shared memory limits when multiple blocks are scheduled. When shared memory per block exceeds ~50 KB on H100/H200, Block Limit Shared Mem drops to 1, serializing all warps on each SM.

> **Applicable when**: Never — adding large XBLOCK×YBLOCK tile configs (product > ~4096 elements) to Triton autotune for kernels that allocate shared memory proportional to tile size causes Block Limit Shared Mem to collapse to 1, destroying occupancy and causing severe slowdowns even when warp cycles per instruction improve.

> **Avoid when**: Always avoid autotune configs where XBLOCK×YBLOCK×dtype_bytes exceeds ~50 KB per block on H100/H200. Specifically, XBLOCK=256×YBLOCK=32×4 bytes = 32 KB is borderline; XBLOCK=512×YBLOCK=32×4 bytes = 64 KB will collapse Block Limit Shared Mem to 1. Monitor Dynamic Shared Memory Per Block in profiling — if it exceeds 50 KB, theoretical occupancy will be severely limited regardless of other improvements.

#### ❌ Plan C: `yblock32_covers_full_y_dimension_with_evict_first_streaming` (0.271×)

Replacing autotune configs to force YBLOCK=32 (covering all 30 Y elements in one block) and switching the load eviction policy to 'evict_first' caused a 3.7x slowdown (4.03 µs → 21.73 µs). The winning config allocated 32.77 KB of dynamic shared memory per block (up from 2.05 KB), which limited theoretical occupancy to 75% (down from 100%) via the 'Block Limit Shared Mem' constraint (6 blocks/SM max), and increased executed instructions by 7.2x (610K → 4.4M). Despite waves_per_SM improving from 0.74 to 1.98 and L1 hit rate rising from 0% to 60.5%, the massive shared memory footprint and instruction count explosion dominated, making the kernel 5.4x slower. The CPIStall rule appeared in the after profile (8.6 cycles MIO stall, 35.7% of warp cycles) due to shared memory pressure, which was absent before.

> **Applicable when**: Never — this specific combination of YBLOCK=32 configs for a kernel with small ynumel (30) causes the Triton compiler to allocate large dynamic shared memory per block (~32 KB), severely limiting block occupancy via shared memory constraints and inflating instruction count by 7x.

> **Avoid when**: Always avoid YBLOCK values that cause dynamic shared memory per block to exceed ~4 KB in Triton kernels targeting H100/H200, as this triggers the 'Block Limit Shared Mem' occupancy constraint and caps theoretical occupancy below 100%. Even if the resulting waves_per_SM improves (0.74 → 1.98), the shared memory bottleneck and instruction count explosion will dominate. The evict_first change alone is not the cause — the shared memory explosion from large YBLOCK is the primary regression driver.

---

## Iteration 4

**Best so far**: 1.4057× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_low_register_pressure_configs_with_lambda_grid | 1.242× | 1.426× | ✓ |  |
| B | revert_large_yblock_shared_memory_regression_with_small_yblock_lambda_grid | 1.266× | 1.095× | ✓ |  |
| C | reduce_shared_memory_via_small_yblock_configs_with_lambda_grid | 1.234× | 1.017× | ✓ |  |
| D | eliminate_shared_memory_to_unlock_occupancy | 1.254× | 0.271× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_low_regis... | Δ | revert_large_yblo... | Δ | reduce_shared_mem... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4130 ns | -1760 ns (-29.9%) | 5380 ns | -510 ns (-8.7%) | 5790 ns | -100 ns (-1.7%) | 21700 ns | +15810 ns (+268.4%) |
| SM Throughput (%) | 15.90 % | 11.95 % | -3.95 pp | 35.66 % | +19.76 pp | 23.41 % | +7.51 pp | 21.64 % | +5.74 pp |
| Memory Throughput (%) | 41.98 % | 29.07 % | -12.91 pp | 36.54 % | -5.44 pp | 34.50 % | -7.48 pp | 49.91 % | +7.93 pp |
| DRAM Throughput (%) | 20.92 % | 15.00 % | -5.92 pp | 12.24 % | -8.68 pp | 11.34 % | -9.58 pp | 5.68 % | -15.24 pp |
| Achieved Occupancy (%) | 54.96 % | 30.26 % | -24.70 pp | 74.26 % | +19.30 pp | 51.24 % | -3.72 pp | 33.56 % | -21.40 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 38.00 register/thread | +12 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 1.48 | +0.74 | 0.99 | +0.25 | 1.98 | +1.24 |
| No Eligible (%) | 71.36 % | 72.25 % | +0.89 pp | 38.05 % | -33.31 pp | 54.95 % | -16.41 pp | 64.40 % | -6.96 pp |
| IPC Active | 1.15 inst/cycle | 1.03 inst/cycle | -0.12 | 2.41 inst/cycle | +1.26 | 1.80 inst/cycle | +0.65 | 1.42 inst/cycle | +0.27 |


### Key Learnings

#### ✅ Plan A: `restore_low_register_pressure_configs_with_lambda_grid` (1.426×)

When a Triton autotune config list includes large XBLOCK values (≥512) with YBLOCK=1, the compiler unrolls the inner loop aggressively, inflating register usage (e.g., 21→40 registers/thread) and capping theoretical occupancy at 75% (down from 100%) on H100/H200. Removing all XBLOCK≥512 and YBLOCK=1 configs and restricting the search space to moderate XBLOCK values (64–256) with YBLOCK≥4 drops register usage back to ≤26/thread, restores 100% theoretical occupancy, and reduces executed instructions by ~4.5x (1.79M→398K), yielding a 1.43x speedup (7.2→4.13 µs). The lambda grid fix (replacing a fixed grid tuple with a lambda referencing meta['XBLOCK']/meta['YBLOCK']) ensures the actual CTA grid adapts to the winning tile size.

> **Applicable when**: Triton kernel with @triton.autotune where the config list includes XBLOCK≥512 or YBLOCK=1 entries, and profiling shows registers_per_thread≥40, theoretical_occupancy≤75%, executed_instructions≥1.5M, and TheoreticalOccupancy or IssueSlotUtilization NCU rules are triggered. The kernel body has moderate complexity (index arithmetic with modulo/division) that the compiler unrolls aggressively at large tile sizes.

> **Avoid when**: Do not apply if the kernel already uses XBLOCK≤256 with YBLOCK≥4 and registers_per_thread≤26. Also avoid if the problem size is large enough that waves_per_SM>>1 even with large XBLOCK (in that case, larger tiles may improve efficiency without register pressure issues). Note that reducing XBLOCK also reduces waves_per_SM (here 0.99→0.37), so this trade-off only pays off when the register pressure reduction outweighs the wave-count reduction.

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     4190 ns  (1.406×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 7460 ns  (0.7895×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    # YBLOCK=1: each CTA handles one y-row -> y0 is constant per CTA -> no y-scatter in stores
+    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8),
+    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8),
+    # Existing multi-row configs kept
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=4),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -20,9 +39,7 @@

     y0 = (yindex % 3)
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +47,7 @@

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
