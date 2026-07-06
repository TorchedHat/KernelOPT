# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_nt_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10850000 ns — **Best latency**: 422620 ns (**25.6732×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_nt_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_017/synthesis/kernel_0_triton_tem_fused_mm_permute_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (256,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 10850000 ns
- **SM Throughput**: 28.85 % | **Memory**: 42.55 % | **DRAM**: 1.55 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.19 inst/cycle | **No Eligible**: 70.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 17.936× | 25.673× | ✓ | **✓ best** |
| B | tma_loads_eliminate_inline_transpose_via_block_ptr | 0.997× | 0.586× | ✓ |  |
| C | eviction_policies_and_register_pressure_reduction | 1.000× | 0.973× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_configs | 0.999× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_elimina... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10850000 ns | 422620 ns | -10427380 ns (-96.1%) | 18510000 ns | +7660000 ns (+70.6%) | 11150000 ns | +300000 ns (+2.8%) | 11200000 ns | +350000 ns (+3.2%) |
| SM Throughput (%) | 28.85 % | 92.43 % | +63.58 pp | 19.16 % | -9.69 pp | 27.90 % | -0.95 pp | 28.07 % | -0.78 pp |
| Memory Throughput (%) | 42.55 % | 72.46 % | +29.91 pp | 34.26 % | -8.29 pp | 41.16 % | -1.39 pp | 41.40 % | -1.15 pp |
| DRAM Throughput (%) | 1.55 % | 19.47 % | +17.92 pp | 13.11 % | +11.56 pp | 1.49 % | -0.06 pp | 1.49 % | -0.06 pp |
| Achieved Occupancy (%) | 12.50 % | 12.18 % | -0.32 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 202.00 register/thread | -53 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 70.29 % | 67.66 % | -2.63 pp | 80.91 % | +10.62 pp | 71.06 % | +0.77 pp | 71.34 % | +1.05 pp |
| IPC Active | 1.19 inst/cycle | 1.29 inst/cycle | +0.10 | 0.77 inst/cycle | -0.42 | 1.15 inst/cycle | -0.04 | 1.16 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 25.6732× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_b_tile_padding | 17.918× | 25.605× | ✓ |  |
| B | fp16_cast_dot_operands_for_tensor_core_and_bank_conflict_reduction | — | fail | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_smem_staging_with_padding | 0.999× | 0.990× | ✓ |  |
| D | fix_shared_memory_bank_conflicts_via_block_ptr_order | 0.994× | 0.573× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | eliminate_shared_... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10850000 ns | 423740 ns | -10426260 ns (-96.1%) | 10960000 ns | +110000 ns (+1.0%) | 18940000 ns | +8090000 ns (+74.6%) |
| SM Throughput (%) | 28.85 % | 92.15 % | +63.30 pp | 28.65 % | -0.20 pp | 18.22 % | -10.63 pp |
| Memory Throughput (%) | 42.55 % | 72.22 % | +29.67 pp | 42.27 % | -0.28 pp | 32.52 % | -10.03 pp |
| DRAM Throughput (%) | 1.55 % | 19.16 % | +17.61 pp | 1.51 % | -0.04 pp | 12.80 % | +11.25 pp |
| Achieved Occupancy (%) | 12.50 % | 12.18 % | -0.32 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 202.00 register/thread | -53 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 70.29 % | 67.68 % | -2.61 pp | 70.48 % | +0.19 pp | 81.27 % | +10.98 pp |
| IPC Active | 1.19 inst/cycle | 1.29 inst/cycle | +0.10 | 1.18 inst/cycle | -0.01 | 0.76 inst/cycle | -0.43 |


### Key Learnings

#### ❌ Plan A: `eliminate_shared_memory_bank_conflicts_via_b_tile_padding` (25.605×)

Restructuring the B tile pointer from (BLOCK_N, BLOCK_K) to (BLOCK_K, BLOCK_N) to eliminate `tl.trans(b)` in the tl.dot call produced no measurable improvement: duration changed by only +1.12 µs (422.62 → 423.74 µs, effectively flat), and the SharedMemoryConflicts rule still reported 4.8-way average bank conflicts (36,554 conflicts, 11.57% of wavefronts) — nearly identical to before (4.7-way, 36,850 conflicts, 11.85%). All other metrics (compute SOL, occupancy, warp cycles per instruction, scheduler eligibility) were statistically unchanged. The kernel is already at 92%+ compute SOL and 95%+ Tensor pipeline utilization; at this ceiling, the bank conflict overhead is already absorbed by the Tensor Core pipeline and does not manifest as measurable latency. The NCU-reported 8.84% speedup estimate was not realized because the kernel is compute-bound, not shared-memory-load-bound.

> **Applicable when**: Never demonstrated to help in a compute-bound Triton matmul kernel already at >90% compute SOL and >90% Tensor pipeline utilization. The NCU SharedMemoryConflicts rule may overestimate the benefit when the kernel is Tensor Core throughput-limited.

> **Avoid when**: When the kernel is already compute-bound (Compute SOL > 90%, Tensor pipeline > 90%): eliminating tl.trans via B pointer reshape does not reduce latency because the shared-memory bank conflict overhead is hidden by the Tensor Core pipeline. Do not apply when NCU SharedMemoryConflicts reports <15% of wavefronts as conflicted and the kernel is simultaneously at >90% compute SOL — the conflict cost is already absorbed.

---

## Iteration 3

**Best so far**: 25.6732× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_tail_via_wave_quantized_grid | 17.925× | 25.962× | ✓ |  |
| B | grid_wave_alignment_via_group_m_and_tile_configs | 1.001× | 0.464× | ✓ |  |
| C | restore_tf32_and_prune_autotune_configs_to_high_performance_subset | 16.121× | 18.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | grid_wave_alignme... | Δ | restore_tf32_and_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10850000 ns | 417920 ns | -10432080 ns (-96.1%) | 23400000 ns | +12550000 ns (+115.7%) | 602370 ns | -10247630 ns (-94.4%) |
| SM Throughput (%) | 28.85 % | 92.75 % | +63.90 pp | 15.12 % | -13.73 pp | 63.19 % | +34.34 pp |
| Memory Throughput (%) | 42.55 % | 61.23 % | +18.68 pp | 28.87 % | -13.68 pp | 55.60 % | +13.05 pp |
| DRAM Throughput (%) | 1.55 % | 18.13 % | +16.58 pp | 16.54 % | +14.99 pp | 13.71 % | +12.16 pp |
| Achieved Occupancy (%) | 12.50 % | 12.49 % | -0.01 pp | 12.50 % | +0.00 pp | 12.21 % | -0.29 pp |
| Registers/Thread | 255.00 register/thread | 188.00 register/thread | -67 | 255.00 register/thread | +0 | 218.00 register/thread | -37 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 0.97 | -0.97 | 1.94 | +0.00 |
| No Eligible (%) | 70.29 % | 73.22 % | +2.93 pp | 84.40 % | +14.11 pp | 73.14 % | +2.85 pp |
| IPC Active | 1.19 inst/cycle | 1.07 inst/cycle | -0.12 | 0.62 inst/cycle | -0.57 | 1.07 inst/cycle | -0.12 |


### Key Learnings

#### ❌ Plan B: `grid_wave_alignment_via_group_m_and_tile_configs` (0.464×)

Adding large-tile autotune configs (BLOCK_M=256, BLOCK_N=256; BLOCK_M=512, BLOCK_N=128) intended to reduce grid waves from 1.94 to 0.97 caused a ~2.15x slowdown: duration jumped from 0.42ms to 23.4ms, compute SOL collapsed from 92.15% to 15.12%, and executed instructions exploded from 91M to 2.78B (~30x more). The autotune selected a large-tile config that pushed registers/thread from 202 to 255, dropping Block Limit Registers from 2 to 1, which also triggered uncoalesced local loads/stores (MemoryCacheAccessPattern rule: only 1 of 32 bytes utilized per sector) — a sign of register spilling to local memory. Critically, the diff also accidentally re-introduced `allow_tf32=False` in the tl.dot call, reverting the 25x speedup from iteration 1; the instruction count explosion is consistent with scalar FFMA replacing Tensor Core MMA. Wave-alignment via larger tiles is counterproductive when register pressure is already at the ceiling (255 regs/thread) and allow_tf32=True must be preserved.

> **Applicable when**: Never: adding large-tile configs (BLOCK_M>=256, BLOCK_N>=256) to autotune when registers/thread is already at 255 causes register spilling (MemoryCacheAccessPattern: 1/32 bytes utilized per sector) and the autotune selects the catastrophically worse config. Wave-alignment via tile enlargement is only safe when register headroom exists.

> **Avoid when**: Always avoid when: (1) registers/thread is already at or near 255 — larger tiles push into register spilling territory; (2) allow_tf32=True is already enabled — any diff that touches tl.dot risks accidentally disabling it; (3) the kernel is already compute-bound at >90% Tensor Core utilization — wave-alignment savings are dwarfed by the cost of register spilling or TF32 regression; (4) Block Limit Registers is already 1 or 2 — there is no occupancy headroom to absorb larger tiles.

---

## Cumulative Progress

```
Baseline:  10850000 ns  (1.000×)
Iter 1     422620 ns  (25.673×)  — tf32_tensor_cores

Final best: 422620 ns  (25.6732×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -49,8 +50,8 @@

 
     # Pointers to A and B tiles
     a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
-    # B is (N, K), we want B[offs_n, offs_k] for the NT matmul
-    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
+    # B is (N, K), load as (BLOCK_K, BLOCK_N) to avoid tl.trans and bank conflicts
+    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
@@ -61,13 +62,12 @@

         a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
         a = tl.load(a_ptrs, mask=a_mask, other=0.0)
 
-        # Load B tile: (BLOCK_N, BLOCK_K)
-        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
+        # Load B tile: (BLOCK_K, BLOCK_N) — no transpose needed
+        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
         b = tl.load(b_ptrs, mask=b_mask, other=0.0)
 
-        # acc += A_tile @ B_tile.T  =>  tl.dot(a, b.T)
-        # b is (BLOCK_N, BLOCK_K), transpose to (BLOCK_K, BLOCK_N)
-        acc = tl.dot(a, tl.trans(b), acc=acc, allow_tf32=False)
+        # acc += A_tile @ B_tile  =>  (M,K) @ (K,N) = (M,N), no tl.trans
+        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
 
         a_ptrs += BLOCK_K * stride_ak
         b_ptrs += BLOCK_K * stride_bk
@@ -78,7 +78,6 @@

     c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
     tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)
-
 
 def kernel_function(a, b):
     # a: (M, K), b: (N, K)  =>  output: (M, N) = a @ b.T
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
