# Kernel Optimization Log

## Summary

- **Kernel**: `linear_mish_mish_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8610000 ns — **Best latency**: 8590000 ns (**1.0023×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`eviction_policies_for_streaming_gemm_tiles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `linear_mish_mish_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_029/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 8610000 ns
- **SM Throughput**: 38.43 % | **Memory**: 60.11 % | **DRAM**: 3.76 %
- **Achieved Occupancy**: 12.30 %
- **IPC Active**: 1.58 inst/cycle | **No Eligible**: 60.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_reduce_register_pressure_hopper | 0.694× | 0.185× | ✓ |  |
| C | eviction_policies_for_streaming_gemm_tiles | 1.001× | 1.002× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_smaller_tiles_and_num_warps | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_reduce_... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8610000 ns | 46480000 ns | +37870000 ns (+439.8%) | 8590000 ns | -20000 ns (-0.2%) | 8620000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 38.43 % | 8.10 % | -30.33 pp | 38.43 % | +0.00 pp | 38.49 % | +0.06 pp |
| Memory Throughput (%) | 60.11 % | 16.32 % | -43.79 pp | 60.12 % | +0.01 pp | 60.14 % | +0.03 pp |
| DRAM Throughput (%) | 3.76 % | 12.91 % | +9.15 pp | 3.66 % | -0.10 pp | 3.66 % | -0.10 pp |
| Achieved Occupancy (%) | 12.30 % | 6.25 % | -6.05 pp | 12.44 % | +0.14 pp | 12.47 % | +0.17 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 60.40 % | 91.32 % | +30.92 pp | 60.45 % | +0.05 pp | 60.41 % | +0.01 pp |
| IPC Active | 1.58 inst/cycle | 0.35 inst/cycle | -1.23 | 1.59 inst/cycle | +0.01 | 1.58 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0023× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_smaller_tiles_reduce_register_spill | 1.000× | 1.001× | ✓ |  |
| C | revert_tma_block_ptrs_to_scalar_pointer_arithmetic | 0.648× | 0.193× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_smaller_... | Δ | revert_tma_block_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8610000 ns | 8600000 ns | -10000 ns (-0.1%) | 44600000 ns | +35990000 ns (+418.0%) |
| SM Throughput (%) | 38.43 % | 38.52 % | +0.09 pp | 8.57 % | -29.86 pp |
| Memory Throughput (%) | 60.11 % | 60.27 % | +0.16 pp | 17.44 % | -42.67 pp |
| DRAM Throughput (%) | 3.76 % | 3.64 % | -0.12 pp | 14.58 % | +10.82 pp |
| Achieved Occupancy (%) | 12.30 % | 12.43 % | +0.13 pp | 6.25 % | -6.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 3.88 | +1.94 |
| No Eligible (%) | 60.40 % | 60.40 % | +0.00 pp | 90.93 % | +30.53 pp |
| IPC Active | 1.58 inst/cycle | 1.59 inst/cycle | +0.01 | 0.36 inst/cycle | -1.22 |

---

## Iteration 3

**Best so far**: 1.0023× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | maxnreg_cap_to_break_register_deadlock_and_improve_occupancy | 0.799× | 0.161× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_w_tile_transpose_and_group_m_sweep | 0.988× | 1.001× | ✓ |  |
| C | shrink_tile_sizes_to_reduce_shared_memory_and_recover_occupancy | 0.983× | 0.654× | ✓ |  |
| D | autotune_small_tile_low_smem_recover_occupancy | 0.945× | 0.192× | ✓ |  |


### Metric Comparison

| Metric | Baseline | maxnreg_cap_to_br... | Δ | fix_shared_memory... | Δ | shrink_tile_sizes... | Δ | autotune_small_ti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8610000 ns | 53570000 ns | +44960000 ns (+522.2%) | 8600000 ns | -10000 ns (-0.1%) | 13170000 ns | +4560000 ns (+53.0%) | 44860000 ns | +36250000 ns (+421.0%) |
| SM Throughput (%) | 38.43 % | 11.13 % | -27.30 pp | 38.52 % | +0.09 pp | 25.07 % | -13.36 pp | 8.46 % | -29.97 pp |
| Memory Throughput (%) | 60.11 % | 24.17 % | -35.94 pp | 60.28 % | +0.17 pp | 93.23 % | +33.12 pp | 17.23 % | -42.88 pp |
| DRAM Throughput (%) | 3.76 % | 17.41 % | +13.65 pp | 3.76 % | +0.00 pp | 1.11 % | -2.65 pp | 14.47 % | +10.71 pp |
| Achieved Occupancy (%) | 12.30 % | 16.97 % | +4.67 pp | 12.43 % | +0.13 pp | 18.15 % | +5.85 pp | 6.25 % | -6.05 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 160.00 register/thread | -95 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.29 | -0.65 | 1.94 | +0.00 | 5.17 | +3.23 | 3.88 | +1.94 |
| No Eligible (%) | 60.40 % | 88.89 % | +28.49 pp | 60.30 % | -0.10 pp | 74.05 % | +13.65 pp | 90.93 % | +30.53 pp |
| IPC Active | 1.58 inst/cycle | 0.44 inst/cycle | -1.14 | 1.58 inst/cycle | +0.00 | 1.04 inst/cycle | -0.54 | 0.36 inst/cycle | -1.22 |


### Key Learnings

#### ❌ Plan A: `maxnreg_cap_to_break_register_deadlock_and_improve_occupancy` (0.161×)

Capping registers via Triton's maxnreg parameter (255→128 regs/thread) successfully reduced Block Limit Registers from 2 to 4 blocks/SM and raised theoretical occupancy from 12.5% to 18.75%, but caused a catastrophic 6.2x slowdown (8.6ms→53.57ms). The compiler responded to the register cap by massively expanding the stack (1024→2432 bytes/thread) and spilling to local memory, increasing executed instructions by 33% and causing CPIStall to appear at 83.9% of cycles waiting on L1TEX local memory loads. Scheduler no-eligible cycles worsened from 60.4% to 88.89% and warp cycles per issued instruction increased nearly 5x (5.02→24.7), indicating the spill traffic completely overwhelmed any occupancy benefit. When a compute-heavy epilogue (e.g., multiple transcendental functions like exp/log/tanh) drives register demand above the cap, the compiler cannot reorganize computation and instead generates uncoalesced local memory spill traffic that is far more expensive than the original register pressure.

> **Applicable when**: maxnreg may help when: (1) register pressure is moderate and the kernel has simple arithmetic (few transcendentals), (2) the compiler can reorganize computation rather than spill, (3) the stack size does not grow significantly under the cap, and (4) the occupancy gain is large enough to offset any spill overhead.

> **Avoid when**: Do NOT apply maxnreg when the kernel contains register-intensive transcendental operations (exp, log, tanh, etc.) applied multiple times in the epilogue. In this regime, the compiler cannot reduce register usage through reorganization and instead generates massive local memory spill traffic (stack size grows 2-3x), causing CPIStall to dominate at >80% of cycles and warp cycles/instruction to increase 5x. The occupancy gain (12.5%→18.75%) is completely overwhelmed by the spill penalty. Also avoid when the kernel is already register-bound at 255 regs/thread with a compute-heavy epilogue — the register demand is irreducible.

#### ❌ Plan C: `shrink_tile_sizes_to_reduce_shared_memory_and_recover_occupancy` (0.654×)

Reducing autotune tile sizes (BLOCK_M/BLOCK_N/BLOCK_K) to lower shared memory per block successfully improved theoretical occupancy from 12.5% to 18.75% and reduced registers/thread from 255 to 160, but caused a 1.53x slowdown (8.6ms → 13.17ms). The smaller tiles dramatically worsened shared memory bank conflicts from 5.4-way to 12.0-way and increased uncoalesced shared wavefronts from 56% to 76%, saturating L1/TEX throughput at 96.54% (up from 62.08%). Despite more active warps per scheduler (1.99 → 2.9), warp cycles per issued instruction more than doubled (5.02 → 11.19) due to the severe bank conflict stalls, and compute throughput actually fell from 38.52% to 25.07%. The occupancy gain was entirely negated by the increased memory access inefficiency from smaller, more conflict-prone tiles.

> **Applicable when**: Never: this strategy (shrinking tiles to recover occupancy) backfired when the kernel already had moderate occupancy (12.5%) and the real bottleneck was shared memory bank conflicts. Smaller tiles increased bank conflict severity rather than reducing it.

> **Avoid when**: Do not shrink GEMM tile sizes when shared memory bank conflicts are already present (5+ way conflicts reported by NCU SharedMemoryConflicts rule). Smaller tiles reduce arithmetic intensity per tile and can worsen bank conflict patterns (e.g., smaller BLOCK_N increases stride-based conflicts in W tile staging), saturating L1/TEX throughput and more than doubling warp stall cycles. The occupancy gain from smaller tiles is insufficient to compensate when bank conflicts dominate stall time. Also avoid when the kernel is already achieving >35% compute SOL — the existing tile size is likely well-matched to the arithmetic intensity.

#### ❌ Plan D: `autotune_small_tile_low_smem_recover_occupancy` (0.192×)

Adding small-tile Triton autotune configs (BLOCK_M/N=32-64, BLOCK_K=32, num_stages=1-2) alongside large-tile configs with BLOCK_K=64-128 and num_stages=3-4 caused autotune to select a large-tile config with higher BLOCK_K and more stages, increasing dynamic shared memory per block from 65 KB to 196 KB. This drove Block Limit Shared Mem from 3 to 1 block/SM, halving theoretical occupancy from 12.5% to 6.25%, and worsening scheduler idle from 60% to 91% (eligible warps dropped from 0.67 to 0.09 per scheduler). The kernel duration increased 5.2x (8.6ms → 44.86ms). When the existing autotune config set already includes large-tile configs that dominate the search space, adding small-tile configs does not guarantee autotune selects them — if the large-tile configs have higher SMEM footprints, autotune may still select them and the net effect is a regression.

> **Applicable when**: Never apply when the autotune config set includes large-tile configs with BLOCK_K>=64 and num_stages>=3, as these produce SMEM footprints >100KB that autotune may prefer over small-tile configs. This anti-pattern is especially harmful when the kernel is already at 255 registers/thread.

> **Avoid when**: Do not add large-tile configs (BLOCK_K>=64, num_stages>=3-4) to the autotune search space when the kernel is already register-bound at 255 registers/thread and shared memory is a secondary occupancy limiter. The autotune may select these large-tile configs, tripling SMEM and making Block Limit Shared Mem the binding occupancy constraint. If the goal is to recover occupancy via small tiles, the large-tile configs must be removed or constrained, not merely supplemented.

---

## Cumulative Progress

```
Baseline:  8610000 ns  (1.000×)
Iter 1     8590000 ns  (1.002×)  — eviction_policies_for_streaming_gemm_tiles

Final best: 8590000 ns  (1.0023×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,16 +4,27 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
+        # --- Existing large-tile configs (kept) ---
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # --- NEW small-tile configs targeting <128 registers/thread ---
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=2),
     ],
     key=['M', 'N', 'K'],
 )
@@ -58,12 +69,14 @@

         x_tile = tl.load(
             X_ptrs,
             mask=(offs_m[:, None] < M) & (mask_k[None, :]),
-            other=0.0
+            other=0.0,
+            eviction_policy='evict_first'  # streaming: each X tile used once per CTA
         )
         w_tile = tl.load(
             W_ptrs,
             mask=(mask_k[:, None]) & (offs_n[None, :] < N),
-            other=0.0
+            other=0.0,
+            eviction_policy='evict_first'  # streaming: each W tile used once per CTA
         )
 
         acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
@@ -72,7 +85,12 @@

         W_ptrs += BLOCK_K * stride_wk
 
     # Add bias
-    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
+    bias = tl.load(
+        B_ptr + offs_n,
+        mask=offs_n < N,
+        other=0.0,
+        eviction_policy='evict_last'  # bias reused across all M-dim CTAs sharing same N column
+    )
     acc = acc + bias[None, :]
 
     # Apply mish: x * tanh(softplus(x))
@@ -98,8 +116,12 @@

     # Store result
     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     Y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
-    tl.store(Y_ptrs, mish2.to(tl.float32), mask=out_mask)
-
+    tl.store(
+        Y_ptrs,
+        mish2.to(tl.float32),
+        mask=out_mask,
+        eviction_policy='evict_first'  # output written once, no reuse
+    )
 
 def kernel_function(p_linear_weight, p_linear_bias, x):
     M, K = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
