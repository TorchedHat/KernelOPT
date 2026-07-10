# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_mul_hardtanh_gelu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17490000 ns — **Best latency**: 5600000 ns (**3.1232×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`prune_autotune_to_small_tiles_and_fix_block_ptr_order`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_mul_hardtanh_gelu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_053/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 3.88
- **Registers/thread**: 255
- **Duration**: 17490000 ns
- **SM Throughput**: 37.47 % | **Memory**: 59.13 % | **DRAM**: 4.20 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.56 inst/cycle | **No Eligible**: 61.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_register_pressure_reduction | 0.950× | 0.661× | ✓ |  |
| C | eviction_policies_for_streaming_gemm_tiles | 0.999× | 1.001× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles_and_more_warps | 1.001× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_re... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17490000 ns | 26470000 ns | +8980000 ns (+51.3%) | 17480000 ns | -10000 ns (-0.1%) | 17510000 ns | +20000 ns (+0.1%) |
| SM Throughput (%) | 37.47 % | 24.63 % | -12.84 pp | 37.50 % | +0.03 pp | 37.51 % | +0.04 pp |
| Memory Throughput (%) | 59.13 % | 92.82 % | +33.69 pp | 59.15 % | +0.02 pp | 59.21 % | +0.08 pp |
| DRAM Throughput (%) | 4.20 % | 1.08 % | -3.12 pp | 4.09 % | -0.11 pp | 4.11 % | -0.09 pp |
| Achieved Occupancy (%) | 12.46 % | 18.34 % | +5.88 pp | 12.50 % | +0.04 pp | 12.50 % | +0.04 pp |
| Registers/Thread | 255.00 register/thread | 168.00 register/thread | -87 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 3.88 | 10.34 | +6.46 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 61.01 % | 74.64 % | +13.63 pp | 60.98 % | -0.03 pp | 61.03 % | +0.02 pp |
| IPC Active | 1.56 inst/cycle | 1.01 inst/cycle | -0.55 | 1.56 inst/cycle | +0.00 | 1.56 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32x3_precision_tensor_cores_for_correctness_safe_mma | 2.438× | 1.796× | ✓ | **✓ best** |
| C | fix_shared_memory_bank_conflicts_via_block_ptr_order | 0.952× | 0.661× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32x3_precision_... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17490000 ns | 9740000 ns | -7750000 ns (-44.3%) | 26450000 ns | +8960000 ns (+51.2%) |
| SM Throughput (%) | 37.47 % | 23.07 % | -14.40 pp | 24.66 % | -12.81 pp |
| Memory Throughput (%) | 59.13 % | 38.56 % | -20.57 pp | 92.94 % | +33.81 pp |
| DRAM Throughput (%) | 4.20 % | 2.57 % | -1.63 pp | 1.08 % | -3.12 pp |
| Achieved Occupancy (%) | 12.46 % | 6.25 % | -6.21 pp | 18.34 % | +5.88 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 168.00 register/thread | -87 |
| Waves/SM | 3.88 | 7.76 | +3.88 | 10.34 | +6.46 |
| No Eligible (%) | 61.01 % | 84.15 % | +23.14 pp | 74.64 % | +13.63 pp |
| IPC Active | 1.56 inst/cycle | 0.63 inst/cycle | -0.93 | 1.01 inst/cycle | -0.55 |


### Key Learnings

#### ✅ Plan A: `tf32x3_precision_tensor_cores_for_correctness_safe_mma` (3.123×)

Combining `input_precision='tf32x3'` in `tl.dot` with `tl.make_block_ptr` / `tl.advance` for tile loads, and expanding the autotune config space to include smaller tiles (BLOCK_M/N=64, BLOCK_K up to 128), reduced registers/thread from 255 to 134 and tripled theoretical occupancy from 6.25% to 18.75%. The causal chain is: TMA block pointers allow the compiler to reduce register pressure (255→134 regs/thread), which relaxes Block Limit Registers from 2 to 3 blocks/SM; combined with the reduced shared memory footprint from the autotuner selecting smaller tiles (196 KB→65 KB/block), Block Limit Shared Mem also rose to 3, yielding 12 active warps/SM vs. 4 before. This drove Compute SM Throughput from 23% to 47.7% and Issue Slots Busy from 15.85% to 44.66%, cutting duration from 9.74ms to 5.6ms (1.74x on top of the prior tf32x3 gain, 3.12x overall). Note: total executed instructions increased (1.07B→1.73B) because the 4× grid expansion (1024→4096 blocks) from smaller tiles adds more total work, but the occupancy and IPC gains more than compensate.

> **Applicable when**: Triton GEMM kernel is register-bound (>200 regs/thread) causing low occupancy (<10%) and high no-eligible-warp rate (>70%), with scalar pointer arithmetic and large autotune tile configs. The kernel already uses or can adopt tf32x3 precision. Registers/thread is the primary occupancy limiter (Block Limit Registers < Block Limit Warps). Switching to tl.make_block_ptr + tl.advance allows the compiler to reduce register pressure by eliminating per-thread pointer arithmetic, freeing registers and enabling the autotuner to select smaller tiles that fit more blocks/SM.

> **Avoid when**: Kernel is already using tl.make_block_ptr and register count is already low (<150 regs/thread). Do not apply when the kernel is memory-bandwidth bound (DRAM throughput >80%) rather than occupancy/compute bound — the register reduction benefit will not translate to speedup. Avoid when the autotune search space does not include smaller tile configs alongside the block pointer migration, as smaller tiles alone (without tf32x3) previously caused a 0.66x regression by amplifying shared memory bank conflicts (see exp_tma_block_ptr_order_fix_regression_002). The combination of tf32x3 + TMA block ptrs + smaller tiles is required; any subset may regress.

---

## Iteration 3

**Best so far**: 1.7957× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_for_occupancy_recovery | — | fail | ✓ |  |
| B | autotune_smem_aware_configs_for_occupancy_recovery | 2.772× | 2.222× | ✓ |  |
| C | tf32_tensor_cores_via_input_precision_tf32x3 | — | fail | ✓ |  |
| D | tf32x3_tensor_cores_for_correctness_safe_mma_with_larger_block_k | 3.705× | 3.123× | ✓ | **✓ best** |
| E | add_triton_autotune | 3.108× | 1.779× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_smem_awa... | Δ | tf32x3_tensor_cor... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17490000 ns | 7870000 ns | -9620000 ns (-55.0%) | 5600000 ns | -11890000 ns (-68.0%) | 9830000 ns | -7660000 ns (-43.8%) |
| SM Throughput (%) | 37.47 % | 28.79 % | -8.68 pp | 47.70 % | +10.23 pp | 23.09 % | -14.38 pp |
| Memory Throughput (%) | 59.13 % | 55.14 % | -3.99 pp | 56.02 % | -3.11 pp | 38.25 % | -20.88 pp |
| DRAM Throughput (%) | 4.20 % | 3.96 % | -0.24 pp | 5.95 % | +1.75 pp | 2.55 % | -1.65 pp |
| Achieved Occupancy (%) | 12.46 % | 12.38 % | -0.08 pp | 18.33 % | +5.87 pp | 6.25 % | -6.21 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 134.00 register/thread | -121 | 255.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 10.34 | +6.46 | 7.76 | +3.88 |
| No Eligible (%) | 61.01 % | 75.77 % | +14.76 pp | 55.34 % | -5.67 pp | 84.15 % | +23.14 pp |
| IPC Active | 1.56 inst/cycle | 0.97 inst/cycle | -0.59 | 1.79 inst/cycle | +0.23 | 0.63 inst/cycle | -0.93 |


### Key Learnings

#### ✅ Plan B: `autotune_smem_aware_configs_for_occupancy_recovery` (2.222×)

When a Triton GEMM kernel is bottlenecked by Block Limit Shared Mem=1 (only one block fits per SM due to excessive dynamic SMEM per block), adding autotune configs with smaller BLOCK_K and fewer pipeline stages dramatically reduces SMEM consumption and allows multiple blocks per SM. In this case, reducing dynamic SMEM from ~197KB to ~66KB per block (via BLOCK_K=32, num_stages=2) raised Block Limit Shared Mem from 1 to 3, doubling theoretical occupancy from 6.25% to 12.5%, active warps per scheduler from 1.0 to 1.99, and eligible warps per scheduler from 0.16 to 0.28, while memory throughput rose from 126 to 194 GB/s. The SMEM budget formula is: num_stages × (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × dtype_bytes; targeting SMEM/block ≤ floor(total_smem_per_sm / desired_blocks_per_sm) is the key design constraint.

> **Applicable when**: Triton GEMM kernel where: (1) Block Limit Shared Mem = 1 (single block per SM due to SMEM), (2) TheoreticalOccupancy < 10%, (3) Eligible Warps Per Scheduler < 0.25, (4) No Eligible cycles > 75%, (5) autotune configs all use large BLOCK_K (≥64) with many pipeline stages (≥3). SMEM formula: num_stages × (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × dtype_bytes must exceed total_smem_per_sm / desired_blocks.

> **Avoid when**: Kernel is already register-bound (Block Limit Registers < Block Limit Shared Mem) — reducing SMEM won't help if registers are the binding constraint. Also avoid when BLOCK_K reduction causes excessive K-loop iterations that overwhelm the compute pipeline (e.g., K is very large and BLOCK_K=32 means 256+ iterations with high loop overhead). Do not apply when the kernel is already memory-bandwidth bound (DRAM Throughput > 60%) rather than occupancy-latency bound.

---

## Iteration 4

**Best so far**: 3.1232× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_smem_via_fewer_pipeline_stages_for_occupancy | 3.691× | 3.123× | ✓ |  |
| B | block_ptr_migration_to_eliminate_local_memory_register_spills | 3.227× | 1.935× | ✓ |  |
| C | tma_block_ptr_register_reduction_for_occupancy | 3.324× | 1.931× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_smem_via_f... | Δ | block_ptr_migrati... | Δ | tma_block_ptr_reg... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17490000 ns | 5600000 ns | -11890000 ns (-68.0%) | 9040000 ns | -8450000 ns (-48.3%) | 9060000 ns | -8430000 ns (-48.2%) |
| SM Throughput (%) | 37.47 % | 47.90 % | +10.43 pp | 25.22 % | -12.25 pp | 25.14 % | -12.33 pp |
| Memory Throughput (%) | 59.13 % | 56.25 % | -2.88 pp | 34.96 % | -24.17 pp | 34.87 % | -24.26 pp |
| DRAM Throughput (%) | 4.20 % | 5.95 % | +1.75 pp | 2.17 % | -2.03 pp | 2.16 % | -2.04 pp |
| Achieved Occupancy (%) | 12.46 % | 18.33 % | +5.87 pp | 6.25 % | -6.21 pp | 6.25 % | -6.21 pp |
| Registers/Thread | 255.00 register/thread | 134.00 register/thread | -121 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 3.88 | 10.34 | +6.46 | 7.76 | +3.88 | 7.76 | +3.88 |
| No Eligible (%) | 61.01 % | 55.33 % | -5.68 pp | 84.52 % | +23.51 pp | 84.47 % | +23.46 pp |
| IPC Active | 1.56 inst/cycle | 1.79 inst/cycle | +0.23 | 0.62 inst/cycle | -0.94 | 0.62 inst/cycle | -0.94 |

---

## Iteration 5

**Best so far**: 3.1232× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | gelu_fast_approximation_to_eliminate_nonfused_fp32_instructions | 3.657× | 3.146× | ✓ |  |
| B | output_block_ptr_store_with_num_stages_1_configs | 3.349× | 1.754× | ✓ |  |
| C | prune_autotune_to_small_tiles_and_fix_block_ptr_order | 2.959× | 3.123× | ✓ |  |


### Metric Comparison

| Metric | Baseline | gelu_fast_approxi... | Δ | output_block_ptr_... | Δ | prune_autotune_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17490000 ns | 5560000 ns | -11930000 ns (-68.2%) | 9970000 ns | -7520000 ns (-43.0%) | 5600000 ns | -11890000 ns (-68.0%) |
| SM Throughput (%) | 37.47 % | 49.08 % | +11.61 pp | 22.60 % | -14.87 pp | 49.11 % | +11.64 pp |
| Memory Throughput (%) | 59.13 % | 55.76 % | -3.37 pp | 35.18 % | -23.95 pp | 56.48 % | -2.65 pp |
| DRAM Throughput (%) | 4.20 % | 6.04 % | +1.84 pp | 2.31 % | -1.89 pp | 5.96 % | +1.76 pp |
| Achieved Occupancy (%) | 12.46 % | 18.33 % | +5.87 pp | 6.25 % | -6.21 pp | 18.33 % | +5.87 pp |
| Registers/Thread | 255.00 register/thread | 134.00 register/thread | -121 | 255.00 register/thread | +0 | 134.00 register/thread | -121 |
| Waves/SM | 3.88 | 10.34 | +6.46 | 7.76 | +3.88 | 10.34 | +6.46 |
| No Eligible (%) | 61.01 % | 55.64 % | -5.37 pp | 84.82 % | +23.81 pp | 55.82 % | -5.19 pp |
| IPC Active | 1.56 inst/cycle | 1.77 inst/cycle | +0.21 | 0.61 inst/cycle | -0.95 | 1.77 inst/cycle | +0.21 |


### Key Learnings

#### ❌ Plan B: `output_block_ptr_store_with_num_stages_1_configs` (1.754×)

Replacing scalar output store with tl.make_block_ptr and adding num_stages=1 autotune configs caused a catastrophic regression: registers/thread jumped from 134 back to 255, dynamic SMEM/block tripled from 65KB to 197KB, theoretical occupancy collapsed from 18.75% to 6.25%, and duration increased from 5.6ms to 9.97ms (0.57x). The causal chain is that the autotuner selected a num_stages=1 config (likely BLOCK_M=128, BLOCK_N=128, BLOCK_K=32) which, despite low SMEM in theory, forced the compiler to spill aggressively to local memory (MemoryCacheAccessPattern rule: only 1.0 of 32 bytes/sector utilized for local loads/stores), and the output block pointer change did not reduce register pressure — instead the compiler allocated more registers for the TMA descriptor state. The net effect was that num_stages=1 eliminated software pipelining, causing the scheduler to stall 84.82% of cycles waiting for memory, far worse than the 55.33% stall rate of the prior kernel.

> **Applicable when**: Never — this is a confirmed regression pattern. Adding num_stages=1 configs to a Triton GEMM autotune space that already has a well-tuned num_stages>=2 baseline causes the autotuner to select a config that eliminates software pipelining, dramatically increasing memory stalls and register spills.

> **Avoid when**: Do not add num_stages=1 autotune configs to a Triton GEMM kernel that is already achieving 18%+ theoretical occupancy with num_stages>=2 configs. The elimination of software pipelining at num_stages=1 causes the scheduler no-eligible rate to jump from ~55% to ~85%, negating any SMEM-per-block savings. Do not replace a working scalar output store with tl.make_block_ptr when registers/thread is already at 134 — the output store is not the register pressure bottleneck at that point, and the TMA descriptor overhead may increase register usage instead.

---

## Cumulative Progress

```
Baseline:  17490000 ns  (1.000×)
Iter 2     9740000 ns  (1.796×)  — tf32x3_precision_tensor_cores_for_correctness_safe_mma
Iter 3     5600000 ns  (3.123×)  — tf32x3_tensor_cores_for_correctness_safe_mma_with_larger_block_k

Final best: 5600000 ns  (3.1232×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,16 +4,38 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
+        # --- Existing 16 configs (unchanged) ---
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
+        # --- 11 new configs with num_stages=2 to reduce SMEM footprint ---
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 256}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -44,38 +66,40 @@

     pid_m = group_id * GROUP_M + (pid % group_size)
     pid_n = (pid % width) // group_size
 
-    # Row/col offsets
+    # Row/col offsets (still needed for bias and output store)
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
 
-    # Pointers
-    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
-    # Weight is stored as [N, K] (out_features x in_features), we access w[n, k]
-    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
+    # TMA block pointers
+    # x is [M, K], load [BLOCK_M, BLOCK_K] tile, row-major (K is fast dim)
+    x_block_ptr = tl.make_block_ptr(
+        base=x_ptr,
+        shape=(M, K),
+        strides=(stride_xm, stride_xk),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # Weight is [N, K]; view as [K, N] to load [BLOCK_K, BLOCK_N] tile
+    # strides=(stride_wk, stride_wn) = (1, K) for the [K, N] transposed view
+    w_block_ptr = tl.make_block_ptr(
+        base=w_ptr,
+        shape=(K, N),
+        strides=(stride_wk, stride_wn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
 
     # Accumulate
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
-        mask_k = offs_k < k_remaining
-
-        x_tile = tl.load(
-            x_ptrs,
-            mask=(offs_m[:, None] < M) & (mask_k[None, :]),
-            other=0.0
-        )
-        w_tile = tl.load(
-            w_ptrs,
-            mask=(mask_k[:, None]) & (offs_n[None, :] < N),
-            other=0.0
-        )
-
-        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
-
-        x_ptrs += BLOCK_K * stride_xk
-        w_ptrs += BLOCK_K * stride_wk
+        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(x_tile, w_tile, acc=acc, input_precision='tf32x3')
+        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
+        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))
 
     # Load bias
     bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
@@ -98,7 +122,6 @@

     out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
     mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     tl.store(out_ptrs, gelu_out.to(tl.float32), mask=mask_out)
-
 
 def kernel_function(p_gemm_weight, p_gemm_bias, x):
     # p_gemm_weight: [out_features, in_features] = [8192, 8192]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
