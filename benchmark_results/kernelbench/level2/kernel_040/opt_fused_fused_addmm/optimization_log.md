# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_scale_residual_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31820000 ns — **Best latency**: 15020000 ns (**2.1185×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tf32_tensor_cores_with_ieee_precision_emulation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_scale_residual_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_040/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 15.52
- **Registers/thread**: 255
- **Duration**: 31820000 ns
- **SM Throughput**: 43.11 % | **Memory**: 61.34 % | **DRAM**: 1.41 %
- **Achieved Occupancy**: 12.38 %
- **IPC Active**: 1.77 inst/cycle | **No Eligible**: 55.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_hopper_register_pressure_reduction | 0.736× | 0.736× | ✓ |  |
| C | eviction_policies_for_streaming_output_and_weight_loads | 1.001× | 0.999× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_autotune_configs | 0.983× | 0.337× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_hopper_... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31820000 ns | 43260000 ns | +11440000 ns (+36.0%) | 31850000 ns | +30000 ns (+0.1%) | 94490000 ns | +62670000 ns (+197.0%) |
| SM Throughput (%) | 43.11 % | 30.66 % | -12.45 pp | 43.09 % | -0.02 pp | 14.83 % | -28.28 pp |
| Memory Throughput (%) | 61.34 % | 73.91 % | +12.57 pp | 61.34 % | +0.00 pp | 98.24 % | +36.90 pp |
| DRAM Throughput (%) | 1.41 % | 3.02 % | +1.61 pp | 1.36 % | -0.05 pp | 1.07 % | -0.34 pp |
| Achieved Occupancy (%) | 12.38 % | 12.50 % | +0.12 pp | 12.38 % | +0.00 pp | 24.93 % | +12.55 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 96.00 register/thread | -159 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 15.52 | +0.00 | 62.06 | +46.54 |
| No Eligible (%) | 55.82 % | 68.38 % | +12.56 pp | 55.80 % | -0.02 pp | 85.13 % | +29.31 pp |
| IPC Active | 1.77 inst/cycle | 1.26 inst/cycle | -0.51 | 1.77 inst/cycle | +0.00 | 0.59 inst/cycle | -1.18 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_weight_tile_transposed_load_for_coalesced_access | 1.001× | 0.998× | ✓ | **✓ best** |
| B | fix_w_tile_layout_to_eliminate_shared_memory_bank_conflicts | 0.736× | 0.736× | ✓ |  |
| C | large_tile_autotune_to_eliminate_smem_bank_conflicts | 0.983× | 0.938× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_weight_tile_t... | Δ | fix_w_tile_layout... | Δ | large_tile_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31820000 ns | 31880000 ns | +60000 ns (+0.2%) | 43220000 ns | +11400000 ns (+35.8%) | 33940000 ns | +2120000 ns (+6.7%) |
| SM Throughput (%) | 43.11 % | 43.12 % | +0.01 pp | 30.69 % | -12.42 pp | 36.34 % | -6.77 pp |
| Memory Throughput (%) | 61.34 % | 61.38 % | +0.04 pp | 73.97 % | +12.63 pp | 80.43 % | +19.09 pp |
| DRAM Throughput (%) | 1.41 % | 1.35 % | -0.06 pp | 3.05 % | +1.64 pp | 1.29 % | -0.12 pp |
| Achieved Occupancy (%) | 12.38 % | 12.41 % | +0.03 pp | 12.50 % | +0.12 pp | 12.50 % | +0.12 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 | 242.00 register/thread | -13 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 55.82 % | 55.78 % | -0.04 pp | 68.35 % | +12.53 pp | 62.61 % | +6.79 pp |
| IPC Active | 1.77 inst/cycle | 1.77 inst/cycle | +0.00 | 1.27 inst/cycle | -0.50 | 1.50 inst/cycle | -0.27 |


### Key Learnings

#### ❌ Plan B: `fix_w_tile_layout_to_eliminate_shared_memory_bank_conflicts` (0.736×)

Swapping the W tile pointer layout from [BLOCK_K, BLOCK_N] to [BLOCK_N, BLOCK_K] and using tl.trans() in tl.dot was intended to make K the fast-varying axis and eliminate shared memory bank conflicts. However, the profiling metrics after the change are nearly identical to before: bank conflicts remain at 8.1-way, uncoalesced shared wavefronts stay at 68%, warp cycles per instruction unchanged at 6.32, and duration unchanged at ~43ms. The tl.trans() approach does not change how Triton stages the tile into shared memory — the compiler still lays out the tile in the same conflicting pattern, so the pointer index swap alone is insufficient to resolve the bank conflict pattern in this kernel.

> **Applicable when**: Never: this specific approach (pointer index swap + tl.trans()) does not resolve shared memory bank conflicts in Triton GEMM kernels when the root cause is the compiler's shared memory staging layout, not the global memory access order.

> **Avoid when**: When shared memory bank conflicts are the dominant bottleneck (SharedMemoryConflicts rule triggered, bank_conflict_way > 5, UncoalescedSharedAccess excessive_wavefronts_pct > 50%): swapping W tile pointer indices and using tl.trans() in tl.dot does not change the compiler's shared memory layout and leaves bank conflicts unchanged. The fix must target the shared memory layout directly (e.g., padding, different tile shapes, or restructuring the accumulation loop).

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_with_ieee_precision_emulation | 3.728× | 2.119× | ✓ | **✓ best** |
| B | break_smem_bank_conflicts_via_non_power_of_two_block_k | 0.981× | 0.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | break_smem_bank_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 31820000 ns | 15020000 ns | -16800000 ns (-52.8%) | 33950000 ns | +2130000 ns (+6.7%) |
| SM Throughput (%) | 43.11 % | 29.54 % | -13.57 pp | 36.31 % | -6.80 pp |
| Memory Throughput (%) | 61.34 % | 58.13 % | -3.21 pp | 80.37 % | +19.03 pp |
| DRAM Throughput (%) | 1.41 % | 5.18 % | +3.77 pp | 1.29 % | -0.12 pp |
| Achieved Occupancy (%) | 12.38 % | 12.38 % | +0.00 pp | 12.50 % | +0.12 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 242.00 register/thread | -13 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 31.03 | +15.51 |
| No Eligible (%) | 55.82 % | 75.77 % | +19.95 pp | 62.61 % | +6.79 pp |
| IPC Active | 1.77 inst/cycle | 0.97 inst/cycle | -0.80 | 1.50 inst/cycle | -0.27 |

---

## Iteration 4

**Best so far**: 2.1185× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_async_pipeline | 2.574× | 1.269× | ✓ |  |
| C | tf32x3_tensor_cores_ozaki_precision_emulation | 3.296× | 3.325× | ✓ |  |
| D | add_triton_autotune | 3.724× | 2.123× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | tf32x3_tensor_cor... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31820000 ns | 25080000 ns | -6740000 ns (-21.2%) | 9570000 ns | -22250000 ns (-69.9%) | 14990000 ns | -16830000 ns (-52.9%) |
| SM Throughput (%) | 43.11 % | 17.80 % | -25.31 pp | 46.39 % | +3.28 pp | 29.47 % | -13.64 pp |
| Memory Throughput (%) | 61.34 % | 34.85 % | -26.49 pp | 43.12 % | -18.22 pp | 58.23 % | -3.11 pp |
| DRAM Throughput (%) | 1.41 % | 1.77 % | +0.36 pp | 4.57 % | +3.16 pp | 5.26 % | +3.85 pp |
| Achieved Occupancy (%) | 12.38 % | 6.25 % | -6.13 pp | 12.49 % | +0.11 pp | 12.36 % | -0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 209.00 register/thread | -46 | 255.00 register/thread | +0 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 31.03 | +15.51 | 15.52 | +0.00 |
| No Eligible (%) | 55.82 % | 85.28 % | +29.46 pp | 68.43 % | +12.61 pp | 75.76 % | +19.94 pp |
| IPC Active | 1.77 inst/cycle | 0.59 inst/cycle | -1.18 | 1.26 inst/cycle | -0.51 | 0.97 inst/cycle | -0.80 |


### Key Learnings

#### ✅ Plan B: `tma_loads_via_block_ptr_async_pipeline` (3.257×)

Replacing scalar pointer arithmetic with tl.make_block_ptr() + tl.advance() while simultaneously loading W as [BLOCK_K, BLOCK_N] (eliminating tl.trans()) and adding cache_modifier='.cs'/'.cg' on stores/bias loads produced nearly identical profiling metrics: duration increased slightly from 9.57ms to 9.77ms, registers/thread rose from 209 to 214, instruction count increased from 2.12B to 2.13B, and all scheduler/occupancy metrics were essentially unchanged. The block pointer migration did not improve eligible warps per scheduler (0.41 both), warp cycles per issued instruction (6.33 vs 6.43), or compute SM throughput (46.4% vs 45.4%). The dominant bottlenecks (CPIStall at ~35%, IssueSlotUtilization at ~55% potential speedup, TheoreticalOccupancy at 12.5%) remain unresolved by this change. The reported 3.26x speedup is relative to the original scalar FFMA baseline, not the immediately preceding kernel — the actual delta from this iteration's change is a slight regression (~2%).

> **Applicable when**: Triton GEMM kernel on Hopper already using tf32x3 and num_warps=8 (block_size=256) with scalar pointer arithmetic in the K-loop; CPIStall and IssueSlotUtilization are the dominant bottlenecks; kernel is register-limited at ~200+ regs/thread with 12.5% theoretical occupancy.

> **Avoid when**: When the kernel has already applied the W-tile layout fix (loading [BLOCK_K, BLOCK_N] directly without tl.trans()) and is already using num_warps=8 — migrating to block pointers in this state yields no measurable improvement and may slightly increase register pressure (209→214 regs/thread) and instruction count. Do not apply expecting further speedup beyond what tf32x3 + W-tile layout + num_warps=8 already achieved.

#### ✅ Plan C: `tf32x3_tensor_cores_ozaki_precision_emulation` (3.325×)

Combining input_precision='tf32x3' in tl.dot with a corrected W-tile pointer layout (loading [BLOCK_K, BLOCK_N] directly instead of [BLOCK_N, BLOCK_K] + tl.trans()) and increasing block size from 128 to 256 threads (num_warps=8) reduced duration from 15.02ms to 9.57ms (3.325x over the original scalar FFMA baseline). The W-tile layout fix eliminated the need for tl.trans(), reducing instruction count from 2.58B to 2.12B and improving compute SM throughput from 29.5% to 46.4%. Doubling the block size doubled waves_per_sm (15.5→31.0), improving scheduler eligible warps from 0.28 to 0.41 and reducing warp cycles per issued instruction from 8.2 to 6.3.

> **Applicable when**: tl.dot uses allow_tf32=False or input_precision='tf32x3' with tl.trans(); registers_per_thread near 255; TheoreticalOccupancy and IssueSlotUtilization rules triggered; eligible_warps_per_scheduler < 0.35; waves_per_sm < 20; compute SM throughput < 35%; correctness tolerance atol/rtol >= 1e-3.

> **Avoid when**: Correctness requires exact FP32 bit-reproducibility; Triton version < 2.1; kernel is already memory-bandwidth bound (DRAM throughput > 80%); W matrix layout requires transposition that cannot be absorbed into pointer arithmetic.

---

## Iteration 5

**Best so far**: 2.1185× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_larger_block_k_and_fewer_warps | 3.724× | 1.587× | ✓ |  |
| B | eliminate_tl_trans_via_direct_block_k_block_n_w_tile_layout | 3.717× | 3.257× | ✓ |  |
| C | reduce_num_stages_to_relieve_smem_block_limit | 3.381× | 1.809× | ✓ |  |
| D | add_triton_autotune | 3.723× | 3.332× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | eliminate_tl_tran... | Δ | reduce_num_stages... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31820000 ns | 20050000 ns | -11770000 ns (-37.0%) | 9770000 ns | -22050000 ns (-69.3%) | 17590000 ns | -14230000 ns (-44.7%) | 9550000 ns | -22270000 ns (-70.0%) |
| SM Throughput (%) | 43.11 % | 22.14 % | -20.97 pp | 45.35 % | +2.24 pp | 25.17 % | -17.94 pp | 46.38 % | +3.27 pp |
| Memory Throughput (%) | 61.34 % | 38.60 % | -22.74 pp | 41.99 % | -19.35 pp | 34.81 % | -26.53 pp | 43.11 % | -18.23 pp |
| DRAM Throughput (%) | 1.41 % | 3.60 % | +2.19 pp | 4.47 % | +3.06 pp | 2.60 % | +1.19 pp | 4.57 % | +3.16 pp |
| Achieved Occupancy (%) | 12.38 % | 6.25 % | -6.13 pp | 12.49 % | +0.11 pp | 6.25 % | -6.13 pp | 12.49 % | +0.11 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 214.00 register/thread | -41 | 255.00 register/thread | +0 | 209.00 register/thread | -46 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 55.82 % | 85.01 % | +29.19 pp | 68.93 % | +13.11 pp | 84.75 % | +28.93 pp | 68.41 % | +12.59 pp |
| IPC Active | 1.77 inst/cycle | 0.60 inst/cycle | -1.17 | 1.24 inst/cycle | -0.53 | 0.61 inst/cycle | -1.16 | 1.26 inst/cycle | -0.51 |


### Key Learnings

#### ❌ Plan A: `reduce_register_pressure_via_larger_block_k_and_fewer_warps` (1.587×)

Increasing BLOCK_K from 32 to 64 and reducing num_warps from 8 to 4 was intended to reduce register pressure and improve occupancy, but the autotuner selected a config that doubled dynamic SMEM per block from 98 to 196 KB, shifting the occupancy limiter from Block_Limit_Registers (1 block/SM) to Block_Limit_Shared_Mem (1 block/SM) while also halving block size from 256 to 128 threads. This cut theoretical occupancy from 12.5% to 6.25%, worsened no-eligible cycles from 68% to 85%, and triggered MemoryCacheAccessPattern warnings for uncoalesced local loads/stores (35-37% speedup potential lost). The net result was a 1.59x slowdown (duration 9.57ms → 20.05ms) because the SMEM increase from larger K-staging buffers consumed all available SMEM headroom, making SMEM the new binding constraint instead of registers.

> **Applicable when**: Never apply this specific pattern when dynamic SMEM per block is already near the per-SM SMEM limit. Increasing BLOCK_K doubles the SMEM required for K-dimension staging buffers, which can shift the occupancy limiter from registers to SMEM.

> **Avoid when**: When dynamic SMEM per block is already >= 64 KB and Block_Limit_Shared_Mem <= 2: increasing BLOCK_K (e.g., 32→64) doubles SMEM staging buffer requirements, making SMEM the new occupancy bottleneck. Also avoid when reducing num_warps (8→4) halves block size, since this reduces waves_per_sm coverage and worsens scheduler starvation. The combination of BLOCK_K=64 + num_warps=4 is particularly dangerous when the kernel already uses software pipelining (num_stages>1) because SMEM usage scales as BLOCK_K × BLOCK_M × num_stages.

#### ❌ Plan C: `reduce_num_stages_to_relieve_smem_block_limit` (1.809×)

Replacing autotune configs with num_stages=4-5 (BLOCK_K=32, ~65-98 KB SMEM) with configs using num_stages=2-3 and BLOCK_K=32-64 (targeting ~65-131 KB SMEM) caused the autotune to select a config with 131 KB dynamic SMEM per block, which reduced Block_Limit_Shared_Mem from 2 to 1 block/SM. Theoretical occupancy dropped from 12.5% to 6.25% (4 active warps/SM vs 8), scheduler no-eligible cycles rose from 68% to 85%, and eligible warps/scheduler fell from 0.41 to 0.15. Compute throughput dropped from 46% to 25% and duration increased from 9.57ms to 17.59ms — a 1.84x regression. The new configs also introduced uncoalesced local loads/stores (MemoryCacheAccessPattern rule, 33.7% speedup potential) not present before, suggesting register spills to local memory from the 255 registers/thread cap.

> **Applicable when**: Never apply this specific pattern. When the kernel is already at block_limit_smem=2 with BLOCK_K=32 configs, introducing BLOCK_K=64 configs into the autotune search space risks selecting a config that exceeds the SMEM budget and drops block_limit_smem to 1.

> **Avoid when**: When the kernel already achieves block_limit_shared_mem >= 2 with the current autotune configs (BLOCK_K=32, num_stages=3-5); when registers_per_thread is at or near 255 (hardware max), as reducing num_stages does not free registers and may cause the compiler to select larger BLOCK_K configs that increase SMEM; when the current occupancy limiter is Block_Limit_Registers rather than Block_Limit_Shared_Mem. Adding BLOCK_K=64 configs alongside BLOCK_K=32 configs is dangerous because autotune may select the BLOCK_K=64 config if it appears faster in isolation, even though it reduces block_limit_smem from 2 to 1.

---

## Cumulative Progress

```
Baseline:  31820000 ns  (1.000×)
Iter 3     15020000 ns  (2.119×)  — tf32_tensor_cores_with_ieee_precision_emulation

Final best: 15020000 ns  (2.1185×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -8,16 +8,19 @@

 # Which simplifies to: out = linear_out * (1 + scale)
 # where scale = 0.5, so out = linear_out * 1.5
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # BLOCK_K=32 with varied num_stages to explore pipeline depth
+        # (non-power-of-2 BLOCK_K rejected by tl.arange in this Triton version)
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -79,7 +82,7 @@

             other=0.0
         )
 
-        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
+        acc = tl.dot(x_tile, w_tile, acc, input_precision='tf32x3')
 
         X_ptrs += BLOCK_K * stride_xk
         W_ptrs += BLOCK_K * stride_wk
@@ -99,7 +102,6 @@

         acc.to(tl.float32),
         mask=mask_m[:, None] & mask_n[None, :]
     )
-
 
 def kernel_function(p_matmul_weight, p_matmul_bias, x, scaling_factor=0.5):
     M, K = x.shape
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
