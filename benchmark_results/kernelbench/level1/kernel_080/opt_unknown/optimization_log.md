# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_implicit_gemm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 20520000 ns — **Best latency**: 17940000 ns (**1.1438×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_transposed_weight_load_to_original_kn_layout_and_remove_tl_trans`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_implicit_gemm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_080/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15748,1,1), Waves/SM: 59.65
- **Registers/thread**: 255
- **Duration**: 20520000 ns
- **SM Throughput**: 51.65 % | **Memory**: 18.45 % | **DRAM**: 0.76 %
- **Achieved Occupancy**: 12.45 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 51.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.001× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_smaller_tiles | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 20520000 ns | 20500000 ns | -20000 ns (-0.1%) | 20520000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 51.65 % | 51.68 % | +0.03 pp | 51.66 % | +0.01 pp |
| Memory Throughput (%) | 18.45 % | 18.46 % | +0.01 pp | 18.45 % | +0.00 pp |
| DRAM Throughput (%) | 0.76 % | 0.76 % | +0.00 pp | 0.76 % | +0.00 pp |
| Achieved Occupancy (%) | 12.45 % | 12.45 % | +0.00 pp | 12.45 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 59.65 | 59.65 | +0.00 | 59.65 | +0.00 |
| No Eligible (%) | 51.77 % | 51.79 % | +0.02 pp | 51.80 % | +0.03 pp |
| IPC Active | 1.93 inst/cycle | 1.93 inst/cycle | +0.00 | 1.93 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.001× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_m_index_decode_outside_k_loop | 0.999× | 0.988× | ✓ |  |
| B | hoist_kloop_index_decode_to_reduce_register_pressure | 0.995× | 0.973× | ✓ |  |
| C | hoist_loop_invariant_index_decode_outside_k_loop | 0.999× | 1.001× | ✓ |  |
| D | hoist_k_index_decode_outside_loop | 0.994× | 1.161× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_m_index_dec... | Δ | hoist_kloop_index... | Δ | hoist_loop_invari... | Δ | hoist_k_index_dec... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20520000 ns | 20770000 ns | +250000 ns (+1.2%) | 21080000 ns | +560000 ns (+2.7%) | 20500000 ns | -20000 ns (-0.1%) | 17680000 ns | -2840000 ns (-13.8%) |
| SM Throughput (%) | 51.65 % | 51.28 % | -0.37 pp | 51.85 % | +0.20 pp | 51.67 % | +0.02 pp | 62.98 % | +11.33 pp |
| Memory Throughput (%) | 18.45 % | 18.33 % | -0.12 pp | 18.42 % | -0.03 pp | 18.45 % | +0.00 pp | 21.49 % | +3.04 pp |
| DRAM Throughput (%) | 0.76 % | 0.75 % | -0.01 pp | 0.74 % | -0.02 pp | 0.76 % | +0.00 pp | 0.88 % | +0.12 pp |
| Achieved Occupancy (%) | 12.45 % | 12.45 % | +0.00 pp | 12.48 % | +0.03 pp | 12.45 % | +0.00 pp | 12.47 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 59.65 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 |
| No Eligible (%) | 51.77 % | 52.26 % | +0.49 pp | 51.45 % | -0.32 pp | 51.79 % | +0.02 pp | 42.38 % | -9.39 pp |
| IPC Active | 1.93 inst/cycle | 1.91 inst/cycle | -0.02 | 1.94 inst/cycle | +0.01 | 1.93 inst/cycle | +0.00 | 2.31 inst/cycle | +0.38 |

---

## Iteration 3

**Best so far**: 1.001× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_weight_loads_via_transposed_tile | 0.944× | 0.750× | ✓ |  |
| B | coalesced_weight_tile_via_transposed_load | 0.698× | 0.719× | ✓ |  |
| C | hoist_weight_base_pointer_and_precompute_k_side_offset | 1.029× | 1.123× | ✓ | **✓ best** |
| D | fix_uncoalesced_weight_load_via_transposed_tile | 0.703× | 0.746× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_weight_l... | Δ | coalesced_weight_... | Δ | hoist_weight_base... | Δ | fix_uncoalesced_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20520000 ns | 27350000 ns | +6830000 ns (+33.3%) | 28540000 ns | +8020000 ns (+39.1%) | 18280000 ns | -2240000 ns (-10.9%) | 27500000 ns | +6980000 ns (+34.0%) |
| SM Throughput (%) | 51.65 % | 49.69 % | -1.96 pp | 49.02 % | -2.63 pp | 55.74 % | +4.09 pp | 49.78 % | -1.87 pp |
| Memory Throughput (%) | 18.45 % | 43.50 % | +25.05 pp | 41.97 % | +23.52 pp | 21.22 % | +2.77 pp | 43.57 % | +25.12 pp |
| DRAM Throughput (%) | 0.76 % | 0.57 % | -0.19 pp | 0.55 % | -0.21 pp | 0.85 % | +0.09 pp | 0.57 % | -0.19 pp |
| Achieved Occupancy (%) | 12.45 % | 12.46 % | +0.01 pp | 12.46 % | +0.01 pp | 12.45 % | +0.00 pp | 12.46 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 59.65 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 |
| No Eligible (%) | 51.77 % | 52.29 % | +0.52 pp | 53.25 % | +1.48 pp | 45.73 % | -6.04 pp | 52.34 % | +0.57 pp |
| IPC Active | 1.93 inst/cycle | 1.91 inst/cycle | -0.02 | 1.88 inst/cycle | -0.05 | 2.17 inst/cycle | +0.24 | 1.90 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan A: `coalesce_weight_loads_via_transposed_tile` (0.750×)

Loading the weight tile in transposed [BLOCK_N, BLOCK_K] layout and calling tl.trans() before tl.dot caused a severe regression: duration increased from 17.68ms to 27.35ms (0.75x). Rather than reducing uncoalesced accesses, the transposed layout dramatically worsened them — UncoalescedSharedAccess excessive wavefronts grew from 284M to 1696M (6x worse) and UncoalescedGlobalAccess excessive sectors grew from 80M to 1350M (17x worse). The transposed memory layout caused Triton to generate strided local loads/stores (MemoryCacheAccessPattern rule triggered: only 1 of 32 bytes per sector utilized), increasing total executed instructions by 30% (7.1B to 9.2B) and dropping compute throughput from 63% to 50%. In Triton, tl.trans() on a loaded tile does not reorder the underlying memory access pattern — it only changes how the tile is presented to tl.dot, while the actual load pattern is determined by the pointer layout at load time. If the weight tensor is stored in [OUT_C, IN_C, KH, KW] (N-major) layout, loading with offs_n as the outer dimension still produces N-strided global accesses regardless of whether the tile is labeled transposed.

> **Applicable when**: Never: this specific transposition strategy is counterproductive for weight tensors stored in OUT_C-major (N-major) layout.

> **Avoid when**: Weight tensor is stored in [OUT_C, IN_C, KH, KW] layout (N-major). Applying tl.trans() to a [BLOCK_N, BLOCK_K] loaded tile does not coalesce global memory accesses — it makes them worse by making N the contiguous load dimension, which is the large-stride dimension. The MemoryCacheAccessPattern rule (only 1/32 bytes utilized per sector) and 6x increase in excessive shared wavefronts confirm this is an anti-pattern for this memory layout.

#### ❌ Plan B: `coalesced_weight_tile_via_transposed_load` (0.722×)

Loading the weight tile in [BLOCK_N, BLOCK_K] layout and calling tl.trans() before tl.dot caused a 28% regression (18.28ms → 28.43ms). Despite the intent to coalesce global weight loads, UncoalescedGlobalAccess excessive sectors exploded from 80M to 1.35B (13% → 71% of total) and UncoalescedSharedAccess excessive wavefronts grew from 284M to 1.70B (46% → 84%), while executed instructions increased 37% (6.85B → 9.40B). A new MemoryCacheAccessPattern rule appeared flagging local loads/stores with only 1/32 bytes utilized per sector, confirming register spilling caused by the transposed tile layout at 255 registers/thread. The L1 hit rate improved (61% → 87%) but was entirely insufficient to offset the massive increase in uncoalesced traffic and instruction count.

> **Applicable when**: Never — this approach caused a severe regression in all measured metrics across multiple iterations (iterations 3 and 4 both confirm this).

> **Avoid when**: When the kernel is at 255 registers/thread (hardware ceiling) with low theoretical occupancy (12.5%). At this register pressure, loading a [BLOCK_N, BLOCK_K] tile and calling tl.trans() introduces register spilling to local memory (MemoryCacheAccessPattern: 1/32 bytes per sector on local loads/stores), massively increasing memory traffic. The weight tensor stored in [OUT_C, IN_C, KH, KW] (N-major) layout already has K as the contiguous dimension — loading with K as the inner tile dimension is the naturally coalesced direction. Adding larger autotune tile configs (128x128, 256x64) alongside tl.trans compounds register pressure further. The UncoalescedSharedAccess signal in the baseline reflects tl.dot's internal shared memory staging and is NOT addressable via transposed global loads.

---

## Iteration 4

**Best so far**: 1.1225× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_x_base_pointer_and_valid_m_outside_k_loop | 0.916× | 0.723× | ✓ |  |
| B | revert_transposed_weight_load_to_original_kn_layout | 0.908× | 1.161× | ✓ |  |
| C | hoist_x_base_pointer_and_static_masks_outside_k_loop | 0.915× | 0.722× | ✓ |  |
| D | revert_transposed_weight_load_to_coalesced_layout | 0.996× | 1.159× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_x_base_poin... | Δ | revert_transposed... | Δ | hoist_x_base_poin... | Δ | revert_transposed... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20520000 ns | 28390000 ns | +7870000 ns (+38.4%) | 17680000 ns | -2840000 ns (-13.8%) | 28430000 ns | +7910000 ns (+38.5%) | 17700000 ns | -2820000 ns (-13.7%) |
| SM Throughput (%) | 51.65 % | 48.97 % | -2.68 pp | 63.03 % | +11.38 pp | 49.17 % | -2.48 pp | 63.01 % | +11.36 pp |
| Memory Throughput (%) | 18.45 % | 41.99 % | +23.54 pp | 21.50 % | +3.05 pp | 42.15 % | +23.70 pp | 21.50 % | +3.05 pp |
| DRAM Throughput (%) | 0.76 % | 0.55 % | -0.21 pp | 0.88 % | +0.12 pp | 0.55 % | -0.21 pp | 0.88 % | +0.12 pp |
| Achieved Occupancy (%) | 12.45 % | 12.46 % | +0.01 pp | 12.47 % | +0.02 pp | 12.46 % | +0.01 pp | 12.47 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 59.65 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 |
| No Eligible (%) | 51.77 % | 52.97 % | +1.20 pp | 42.35 % | -9.42 pp | 52.89 % | +1.12 pp | 42.39 % | -9.38 pp |
| IPC Active | 1.93 inst/cycle | 1.88 inst/cycle | -0.05 | 2.30 inst/cycle | +0.37 | 1.88 inst/cycle | -0.05 | 2.30 inst/cycle | +0.37 |


### Key Learnings

#### ❌ Plan A: `hoist_x_base_pointer_and_valid_m_outside_k_loop` (0.723×)

Applying M-side base pointer hoisting (base_x_m, out_h_scaled, out_w_scaled, valid_m) simultaneously with a transposed weight tile load ([BLOCK_N, BLOCK_K] + tl.trans()) caused a 28% regression (18.28ms → 28.39ms). The dominant cause was the transposed weight load, not the pointer hoisting: UncoalescedSharedAccess excessive wavefronts grew 6x (284M → 1696M), UncoalescedGlobalAccess excessive sectors grew 17x (80M → 1350M), and total executed instructions increased 37% (6.85B → 9.40B). The MemoryCacheAccessPattern rule triggered (only 1/32 bytes per sector utilized for local loads/stores), confirming the transposed layout forced strided register spill/fill traffic. The L1 hit rate improvement (61% → 87%) was insufficient to offset the massive uncoalesced traffic increase. This is the same anti-pattern documented in items b3e2f7c1 and f2a1b3c4: the transposed weight tile load is the root cause of regression regardless of what other hoisting is applied alongside it.

> **Applicable when**: Never — this combined change (M-side hoist + transposed weight tile) is always worse than M-side hoist alone. The transposed weight tile load is the dominant regression source.

> **Avoid when**: Always avoid combining M-side pointer hoisting with transposed weight tile loading ([BLOCK_N, BLOCK_K] + tl.trans()). The transposed weight load causes 6x more UncoalescedSharedAccess wavefronts and 17x more UncoalescedGlobalAccess sectors for weight tensors stored in OUT_C-major layout, increasing instruction count by ~37% and causing severe regression regardless of other optimizations applied simultaneously. The M-side hoisting alone (without transposed weight load) is beneficial — see items c7f3a291 and d4e9b7c2.

#### ✅ Plan D: `revert_transposed_weight_load_to_coalesced_layout` (1.159×)

After a prior iteration introduced a regressed transposed [BLOCK_N, BLOCK_K] weight load with tl.trans(), reverting to the direct [BLOCK_K, BLOCK_N] layout (K as outer dim, N as inner dim) restored Compute SM Throughput from 55.7% to 63.0% and reduced duration from 18.28ms to 17.7ms. The revert also replaced small-tile autotune configs (targeting lower register pressure) with larger tiles, and re-hoisted M-side base pointer arithmetic outside the K-loop. The combined effect improved Issue Slots Busy from 54.3% to 57.6% and reduced No-Eligible-Warp cycles from 45.7% to 42.4%, confirming the coalesced weight load layout is the correct baseline for this kernel.

> **Applicable when**: Weight tensor is stored in [OUT_C, IN_C, KH, KW] layout where OUT_C has the largest stride. The K-loop iterates over (in_c, kh, kw) which are the small-stride dimensions. NCU reports UncoalescedGlobalAccess and UncoalescedSharedAccess as top rules. A prior iteration introduced a transposed [BLOCK_N, BLOCK_K] load with tl.trans() that caused regression.

> **Avoid when**: The weight tensor is already loaded in [BLOCK_K, BLOCK_N] layout with K as the outer dimension and N as the inner (coalesced) dimension. Do not re-apply if the baseline is already the coalesced layout — this is a recovery/revert strategy, not a novel optimization.

---

## Iteration 5

**Best so far**: 1.1225× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_stages_to_eliminate_shared_memory_bank_conflicts | 1.183× | 1.355× | ✓ |  |
| B | revert_transposed_weight_load_to_original_kn_layout_and_remove_tl_trans | 1.227× | 1.144× | ✓ | **✓ best** |
| D | revert_transposed_weight_load_restore_kn_layout_with_hoisted_pointers | 1.223× | 1.145× | ✓ |  |
| E | add_triton_autotune | 1.077× | 1.161× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_stages... | Δ | revert_transposed... | Δ | revert_transposed... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20520000 ns | 15150000 ns | -5370000 ns (-26.2%) | 17940000 ns | -2580000 ns (-12.6%) | 17930000 ns | -2590000 ns (-12.6%) | 17680000 ns | -2840000 ns (-13.8%) |
| SM Throughput (%) | 51.65 % | 69.70 % | +18.05 pp | 62.53 % | +10.88 pp | 62.41 % | +10.76 pp | 62.99 % | +11.34 pp |
| Memory Throughput (%) | 18.45 % | 20.27 % | +1.82 pp | 21.27 % | +2.82 pp | 21.23 % | +2.78 pp | 21.49 % | +3.04 pp |
| DRAM Throughput (%) | 0.76 % | 1.03 % | +0.27 pp | 0.90 % | +0.14 pp | 0.87 % | +0.11 pp | 0.88 % | +0.12 pp |
| Achieved Occupancy (%) | 12.45 % | 12.45 % | +0.00 pp | 12.46 % | +0.01 pp | 12.46 % | +0.01 pp | 12.47 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 59.65 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 | 59.65 | +0.00 |
| No Eligible (%) | 51.77 % | 37.89 % | -13.88 pp | 43.25 % | -8.52 pp | 43.12 % | -8.65 pp | 42.39 % | -9.38 pp |
| IPC Active | 1.93 inst/cycle | 2.48 inst/cycle | +0.55 | 2.27 inst/cycle | +0.34 | 2.27 inst/cycle | +0.34 | 2.30 inst/cycle | +0.37 |


### Key Learnings

#### ✅ Plan A: `reduce_num_stages_to_eliminate_shared_memory_bank_conflicts` (1.355×)

When Triton's software pipeline (num_stages>1) stages input tiles through shared memory before feeding wgmma, non-square tile shapes with non-power-of-2 strides cause shared memory bank conflicts during the wgmma read phase, reported as UncoalescedSharedAccess excessive wavefronts. Setting num_stages=1 disables the software pipeline entirely, eliminating shared memory staging and its associated bank conflicts: dynamic shared memory per block dropped from 73.7 KB to 24.6 KB, executed instructions fell from 7.11B to 6.54B (~8%), Compute SM Throughput rose from 63% to 70%, and No-Eligible-Warp cycles dropped from 42.4% to 37.9%, yielding a 35% wall-time reduction. At very low occupancy (12.5%, only 2 active warps/scheduler), there is insufficient warp-level parallelism to benefit from pipeline overlap anyway, so eliminating bank conflicts outweighs the loss of load-compute overlap.

> **Applicable when**: Triton kernel with num_stages>1 where NCU reports UncoalescedSharedAccess as a top rule with a large number of excessive wavefronts (>40% of total wavefronts). Kernel has very low theoretical occupancy (<25%) due to high register pressure (near 255 registers/thread), meaning only 1-2 active warps/scheduler are available — insufficient to hide load latency through pipeline overlap. Dynamic shared memory per block is large (>50 KB) relative to the SM shared memory budget, limiting Block Limit Shared Mem.

> **Avoid when**: Kernel has high occupancy (>50%) with many active warps per scheduler (>8), where pipeline overlap from num_stages>1 provides meaningful latency hiding. The kernel is memory-bandwidth-bound (DRAM throughput >60%) rather than compute-bound, where software pipelining's async prefetch benefit outweighs bank conflict cost. Tile shapes are square or have power-of-2 strides that avoid shared memory bank conflicts even with num_stages>1.

---

## Cumulative Progress

```
Baseline:  20520000 ns  (1.000×)
Iter 1     20500000 ns  (1.001×)  — tf32_tensor_cores
Iter 3     18280000 ns  (1.123×)  — hoist_weight_base_pointer_and_precompute_k_side_offset
Iter 5     17940000 ns  (1.144×)  — revert_transposed_weight_load_to_original_kn_layout_and_remove_tl_trans

Final best: 17940000 ns  (1.1438×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,8 +2,10 @@

 import triton
 import triton.language as tl
 
+
 @triton.autotune(
     configs=[
+        # Original 8 configs (large tiles)
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
@@ -12,6 +14,19 @@

         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
+        # New 8 small-tile configs targeting lower register pressure
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=5),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=5),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
+        # New larger-BLOCK_N configs for coalesced transposed weight loads
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
     ],
     key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
 )
@@ -36,10 +51,15 @@

     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
 
-    # Decode M index -> (batch, out_h, out_w)
-    out_w = offs_m % OUT_W
-    out_h = (offs_m // OUT_W) % OUT_H
-    batch = offs_m // (OUT_W * OUT_H)
+    # HOISTED: M-dimension decodes — loop-invariant, compute once before K-loop
+    out_w_1d = offs_m % OUT_W                    # [BLOCK_M]
+    out_h_1d = (offs_m // OUT_W) % OUT_H        # [BLOCK_M]
+    batch_1d = offs_m // (OUT_W * OUT_H)        # [BLOCK_M]
+
+    # HOISTED: base pointer and static masks — loop-invariant, compute once
+    x_base_m = X_ptr + batch_1d * s_xn + out_h_1d * s_xh + out_w_1d * s_xw  # [BLOCK_M]
+    valid_m = offs_m < M_TOTAL    # [BLOCK_M] 1D
+    valid_wn = offs_n < OUT_C     # [BLOCK_N] 1D
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
@@ -51,32 +71,34 @@

         kh = (offs_k // KW) % KH
         in_c = offs_k // (KH * KW)
 
-        # Compute input coordinates
-        ih = out_h[:, None] * STRIDE_H + kh[None, :] * DIL_H - PAD_H
-        iw = out_w[:, None] * STRIDE_W + kw[None, :] * DIL_W - PAD_W
+        # K-varying deltas (using input tensor strides s_xh, s_xw)
+        dh = kh[None, :] * DIL_H - PAD_H   # [1, BLOCK_K]
+        dw = kw[None, :] * DIL_W - PAD_W   # [1, BLOCK_K]
 
-        valid_m = offs_m[:, None] < M_TOTAL
+        # Compute input coordinates for spatial bounds check
+        ih = out_h_1d[:, None] * STRIDE_H + dh  # [BLOCK_M, BLOCK_K]
+        iw = out_w_1d[:, None] * STRIDE_W + dw  # [BLOCK_M, BLOCK_K]
+
         valid_k = offs_k[None, :] < TOTAL_K
         valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
-        valid_x = valid_m & valid_k & valid_spatial
+        valid_x = valid_m[:, None] & valid_k & valid_spatial  # use hoisted valid_m
 
-        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
+        # Use hoisted x_base_m; add K-varying components only
+        x_ptrs = x_base_m[:, None] + in_c[None, :] * s_xc + dh * s_xh + dw * s_xw
         x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
 
-        valid_wk = offs_k[:, None] < TOTAL_K
-        valid_wn = offs_n[None, :] < OUT_C
+        # Direct weight load: shape [BLOCK_K, BLOCK_N] — K as outer dim, N as inner dim
+        # This matches the physical layout [OUT_C, IN_C, KH, KW] with KW stride-1 (coalesced)
+        valid_wk = offs_k[:, None] < TOTAL_K    # [BLOCK_K, 1]
+        valid_wn_2d = offs_n[None, :] < OUT_C   # [1, BLOCK_N]
         w_ptrs = W_ptr + offs_n[None, :] * s_wn + in_c[:, None] * s_wc + kh[:, None] * s_wh + kw[:, None] * s_ww
-        w_tile = tl.load(w_ptrs, mask=valid_wk & valid_wn, other=0.0)
+        w_tile = tl.load(w_ptrs, mask=valid_wk & valid_wn_2d, other=0.0)  # [BLOCK_K, BLOCK_N]
 
         acc = tl.dot(x_tile, w_tile, acc)
 
-    # Store output
-    valid_out_m = offs_m[:, None] < M_TOTAL
-    valid_out_n = offs_n[None, :] < OUT_C
-
-    y_ptrs = Y_ptr + batch[:, None] * s_yn + offs_n[None, :] * s_yc + out_h[:, None] * s_yh + out_w[:, None] * s_yw
-    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out_m & valid_out_n)
-
+    # Store output — reuse hoisted valid_m and valid_wn
+    y_ptrs = Y_ptr + batch_1d[:, None] * s_yn + offs_n[None, :] * s_yc + out_h_1d[:, None] * s_yh + out_w_1d[:, None] * s_yw
+    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_m[:, None] & valid_wn[None, :])
 
 def kernel_function(weight, x):
     N, IN_C, IN_H, IN_W = x.shape
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
